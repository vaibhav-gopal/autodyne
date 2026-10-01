//! Turns any `autodyne` [`Parameterized`] processor's parameters into NIH-plug parameters.
//!
//! [`ParamBridge`] builds one `FloatParam` per `ParamInfo` (range, log scaling, default, autodyne's
//! own value formatting) and implements NIH-plug's `Params`, so a plugin exposes exactly the
//! processor's parameters with no hand-written parameter struct. Call [`ParamBridge::apply`] at the
//! start of each process block to push host changes into the processor; it only touches parameters
//! that changed and never allocates.

use std::sync::Arc;

use autodyne::params::{ParamInfo, ParamScale, Parameterized};
use nih_plug::prelude::*;

/// NIH-plug parameters mirroring a processor's `Parameterized` interface.
pub struct ParamBridge {
    params: Vec<FloatParam>,
    ids: Vec<String>,
    /// host-facing group per parameter: the owning stage's name, or "" when every parameter has the
    /// same owner (a single processor needs no grouping)
    groups: Vec<String>,
    /// last value pushed to the processor, per parameter (NaN = never pushed)
    applied: Vec<std::sync::atomic::AtomicU32>,
}

impl ParamBridge {
    /// One host parameter per parameter of `processor`, starting at the processor's current values.
    /// Ids repeated across the stages of a chain get a numeric suffix to stay unique, and a chain's
    /// parameters are grouped by stage.
    pub fn new(processor: &dyn Parameterized) -> Self {
        let mut params = Vec::new();
        let mut ids: Vec<String> = Vec::new();
        let mut groups: Vec<String> = Vec::new();
        for index in 0..processor.param_count() {
            let info = processor.param_info(index).expect("index below param_count");
            let current = processor.get_param(index).unwrap_or(info.default);
            params.push(float_param(&info, current));
            let mut id = info.id.to_string();
            let mut n = 2;
            while ids.contains(&id) {
                id = format!("{}_{n}", info.id);
                n += 1;
            }
            ids.push(id);
            groups.push(processor.param_group(index).unwrap_or_default().to_string());
        }
        if groups.iter().all(|g| *g == groups[0]) {
            groups.iter_mut().for_each(String::clear);
        }
        let applied = (0..params.len()).map(|_| std::sync::atomic::AtomicU32::new(f32::NAN.to_bits())).collect();
        Self { params, ids, groups, applied }
    }

    /// Pushes every host value that changed since the last call into `processor`.
    /// Call at the start of each process block (it doesn't allocate).
    pub fn apply(&self, processor: &mut dyn Parameterized) {
        use std::sync::atomic::Ordering::Relaxed;
        for (index, (param, applied)) in self.params.iter().zip(&self.applied).enumerate() {
            let value = param.value();
            if f32::from_bits(applied.load(Relaxed)) != value {
                // values come from the host already inside the range, so this can't fail
                let _ = processor.set_param(index, value as f64);
                applied.store(value.to_bits(), Relaxed);
            }
        }
    }

    /// Forces the next `apply` to push every value (e.g. after rebuilding the processor).
    pub fn invalidate(&self) {
        self.applied.iter().for_each(|a| a.store(f32::NAN.to_bits(), std::sync::atomic::Ordering::Relaxed));
    }

    /// The host-facing parameter at `index`.
    pub fn param(&self, index: usize) -> &FloatParam {
        &self.params[index]
    }

    pub fn len(&self) -> usize {
        self.params.len()
    }

    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }
}

/// The NIH-plug parameter for one autodyne parameter.
fn float_param(info: &ParamInfo, value: f64) -> FloatParam {
    let (min, max) = (info.min as f32, info.max as f32);
    let range = match info.scale {
        ParamScale::Log if info.min > 0.0 && info.max > info.min => {
            // a skew that puts the geometric mean at the middle of the control, like a log scale
            let middle = ((info.min * info.max).sqrt() as f32 - min) / (max - min);
            FloatRange::Skewed { min, max, factor: 0.5f32.log(middle) }
        }
        _ => FloatRange::Linear { min, max },
    };
    let display = *info;
    FloatParam::new(info.name, value as f32, range)
        .with_value_to_string(Arc::new(move |v| display.format(v as f64)))
        .with_string_to_value(Arc::new(move |text| display.parse(text).map(|v| display.clamp(v) as f32)))
}

// SAFETY: every `ParamPtr` points into `self.params`, a Vec that is never resized or reallocated
// after construction, so the pointers stay valid for as long as the bridge (held in an Arc by the
// plugin) lives, which is what NIH-plug requires.
unsafe impl Params for ParamBridge {
    fn param_map(&self) -> Vec<(String, ParamPtr, String)> {
        self.params.iter().zip(&self.ids).zip(&self.groups).map(|((p, id), group)| (id.clone(), p.as_ptr(), group.clone())).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autodyne::reverb::Reverb;

    #[test]
    fn mirrors_ranges_defaults_and_formatting() {
        let reverb = Reverb::<f32>::new(48_000.0);
        let bridge = ParamBridge::new(&reverb);
        assert_eq!(bridge.len(), reverb.param_count());
        let decay = bridge.param(1); // decay_s: 0.05 .. 30 s, log
        assert_eq!(decay.value(), reverb.decay());
        // a log parameter's control midpoint is the geometric mean of its range
        let middle = (0.05f32 * 30.0).sqrt();
        assert!((decay.preview_plain(0.5) - middle).abs() < 1e-3 * middle);
        assert_eq!(decay.to_string(), "1.80 s");
    }

    #[test]
    fn apply_pushes_only_changes() {
        let mut reverb = Reverb::<f32>::new(48_000.0);
        let bridge = ParamBridge::new(&reverb);
        bridge.apply(&mut reverb);
        assert_eq!(reverb.decay(), 1.8);
        reverb.set_decay(5.0); // changed directly, not through the host
        bridge.apply(&mut reverb);
        assert_eq!(reverb.decay(), 5.0, "unchanged host values are not re-sent");
        bridge.invalidate();
        bridge.apply(&mut reverb);
        assert_eq!(reverb.decay(), 1.8, "after invalidate everything is re-sent");
    }

    #[test]
    fn parses_typed_values_in_display_units() {
        let reverb = Reverb::<f32>::new(48_000.0);
        let bridge = ParamBridge::new(&reverb);
        let by_id = |id: &str| bridge.param(reverb.param_index(id).unwrap());
        assert_eq!(by_id("decay_s").string_to_normalized_value("2.5 s"), Some(by_id("decay_s").preview_normalized(2.5)));
        let mix = by_id("mix");
        assert_eq!(mix.string_to_normalized_value("50%"), Some(mix.preview_normalized(0.5)));
        assert_eq!(mix.string_to_normalized_value("abc"), None);
        // the host's value -> text -> value -> text round trip is stable
        let text = mix.normalized_value_to_string(0.0101, true);
        let back = mix.string_to_normalized_value(&text).unwrap();
        assert_eq!(mix.normalized_value_to_string(back, true), text);
    }

    #[test]
    fn duplicate_ids_in_chains_get_suffixes() {
        use autodyne::gain::Gain;
        let chain = (Gain::<f32>::new(1.0, 0.0, 48_000.0), Gain::<f32>::new(1.0, 0.0, 48_000.0));
        let bridge = ParamBridge::new(&chain);
        let ids: Vec<String> = bridge.param_map().into_iter().map(|(id, _, _)| id).collect();
        assert_eq!(ids, ["gain_db", "gain_db_2"]);
    }

    #[test]
    fn chains_are_grouped_by_stage() {
        use autodyne::gain::Gain;
        let groups = |bridge: ParamBridge| bridge.param_map().into_iter().map(|(_, _, g)| g).collect::<Vec<_>>();
        let single = Reverb::<f32>::new(48_000.0);
        assert!(groups(ParamBridge::new(&single)).iter().all(String::is_empty), "one stage: no groups");
        let chain = (Gain::<f32>::new(1.0, 0.0, 48_000.0), Reverb::<f32>::new(48_000.0));
        let g = groups(ParamBridge::new(&chain));
        assert_eq!((g[0].as_str(), g[1].as_str()), ("Gain", "Reverb"));
    }
}
