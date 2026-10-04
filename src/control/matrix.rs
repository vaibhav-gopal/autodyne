//! A modulation matrix: routes control signals (LFOs, envelopes, velocity, macros, ...) to any
//! parameter of any processor.

use std::ops::Range;

use thiserror::Error;

use crate::channels::{AudioBuffer, MultiProcessor};
use crate::params::{ParamError, ParamInfo, ParamUnit, Parameterized, RAMP_STEP};
use crate::processor::Processor;
use crate::units::Float;

/// Most route slots a [`Modulated`] processor can have.
pub const MAX_ROUTES: usize = 16;

const DEPTH_IDS: [&str; MAX_ROUTES] = [
    "mod_1_depth", "mod_2_depth", "mod_3_depth", "mod_4_depth", "mod_5_depth", "mod_6_depth", "mod_7_depth", "mod_8_depth",
    "mod_9_depth", "mod_10_depth", "mod_11_depth", "mod_12_depth", "mod_13_depth", "mod_14_depth", "mod_15_depth", "mod_16_depth",
];
const DEPTH_NAMES: [&str; MAX_ROUTES] = [
    "Mod 1 depth", "Mod 2 depth", "Mod 3 depth", "Mod 4 depth", "Mod 5 depth", "Mod 6 depth", "Mod 7 depth", "Mod 8 depth",
    "Mod 9 depth", "Mod 10 depth", "Mod 11 depth", "Mod 12 depth", "Mod 13 depth", "Mod 14 depth", "Mod 15 depth", "Mod 16 depth",
];

/// One connection: `sources[source]` (times `sources[via]`, if any) moves parameter `destination`.
/// Its depth is a separate, automatable parameter of the [`Modulated`] processor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Route {
    pub source: usize,
    /// a parameter index of the wrapped processor
    pub destination: usize,
    /// another source scaling this route, e.g. the mod wheel controlling vibrato depth
    pub via: Option<usize>,
}

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteError {
    #[error("no route slot {0}")]
    UnknownSlot(usize),
    #[error("no modulation source {0}")]
    UnknownSource(usize),
    #[error("no parameter {0} to modulate")]
    UnknownDestination(usize),
}

/// A processor whose parameters can be modulated.
///
/// Each parameter's value is its **base** (what the host or user set, through this wrapper's
/// `Parameterized` interface) plus the sum of its routes' `depth * source` in **normalized** units:
/// a depth of 0.1 moves a parameter by a tenth of its range (for a log frequency, a tenth of its
/// octaves), and the result is clamped to the range. The wrapper exposes the wrapped processor's
/// parameters followed by one depth parameter per route slot (`mod_1_depth`, ...), so hosts can
/// automate modulation amounts too.
///
/// Modulation is applied at control rate: [`run`](Self::run) updates the sources and parameters
/// every [`RAMP_STEP`] samples. Coefficient parameters (frequencies, Q, rates, times) follow within
/// one step; a parameter the processor smooths itself ([`Smoothing::Internal`](crate::params::Smoothing),
/// e.g. gains and mixes) also passes each modulated value through that 20 ms ramp, which softens
/// fast modulation of it. Allocates only in `new`.
#[derive(Debug, Clone)]
pub struct Modulated<P> {
    inner: P,
    infos: Vec<ParamInfo>,
    /// plain values set through the wrapper
    base: Vec<f64>,
    /// plain values last written into the wrapped processor
    pushed: Vec<f64>,
    /// per-parameter normalized offsets (scratch for `apply`)
    offsets: Vec<f64>,
    sources: Vec<f64>,
    routes: Vec<Option<Route>>,
    depths: Vec<f64>,
    /// display names of the depth parameters
    depth_names: Vec<&'static str>,
}

impl<P: Parameterized> Modulated<P> {
    /// `sources` modulation inputs and `route_slots` routes (at most [`MAX_ROUTES`]).
    pub fn new(inner: P, sources: usize, route_slots: usize) -> Self {
        assert!(route_slots <= MAX_ROUTES, "at most {MAX_ROUTES} route slots");
        let infos: Vec<ParamInfo> = (0..inner.param_count()).map(|i| inner.param_info(i).expect("index below param_count")).collect();
        let base: Vec<f64> = infos.iter().enumerate().map(|(i, info)| inner.get_param(i).unwrap_or(info.default)).collect();
        Self {
            pushed: base.clone(),
            offsets: vec![0.0; infos.len()],
            base,
            infos,
            inner,
            sources: vec![0.0; sources],
            routes: vec![None; route_slots],
            depths: vec![0.0; route_slots],
            depth_names: DEPTH_NAMES[..route_slots].to_vec(),
        }
    }
    pub fn inner(&self) -> &P {
        &self.inner
    }
    /// The wrapped processor. Parameters changed on it directly are overwritten by the next
    /// `apply`; call [`sync`](Self::sync) to adopt them as bases instead.
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }
    pub fn into_inner(self) -> P {
        self.inner
    }
    /// Adopts the wrapped processor's current values as the bases.
    pub fn sync(&mut self) {
        for i in 0..self.infos.len() {
            if let Some(v) = self.inner.get_param(i) {
                self.base[i] = v;
                self.pushed[i] = v;
            }
        }
    }

    /// Sets one source's value (typically -1..1 for LFOs, 0..1 for envelopes and controllers).
    /// Out-of-range sources are ignored.
    pub fn set_source(&mut self, source: usize, value: f64) {
        if let Some(s) = self.sources.get_mut(source) {
            *s = value;
        }
    }
    pub fn sources(&self) -> &[f64] {
        &self.sources
    }
    pub fn sources_mut(&mut self) -> &mut [f64] {
        &mut self.sources
    }

    pub fn route_slots(&self) -> usize {
        self.routes.len()
    }
    pub fn route(&self, slot: usize) -> Option<Route> {
        self.routes.get(slot).copied().flatten()
    }
    /// Connects (or with `None`, disconnects) a route slot. Never allocates, so routes can change
    /// while audio runs.
    pub fn set_route(&mut self, slot: usize, route: Option<Route>) -> Result<(), RouteError> {
        if slot >= self.routes.len() {
            return Err(RouteError::UnknownSlot(slot));
        }
        if let Some(r) = route {
            if r.source >= self.sources.len() {
                return Err(RouteError::UnknownSource(r.source));
            }
            if let Some(v) = r.via.filter(|&v| v >= self.sources.len()) {
                return Err(RouteError::UnknownSource(v));
            }
            if r.destination >= self.infos.len() {
                return Err(RouteError::UnknownDestination(r.destination));
            }
        }
        self.routes[slot] = route;
        Ok(())
    }
    /// Route depth in normalized units, -1..1 (clamped).
    pub fn set_depth(&mut self, slot: usize, depth: f64) -> Result<(), RouteError> {
        let d = self.depths.get_mut(slot).ok_or(RouteError::UnknownSlot(slot))?;
        *d = depth.clamp(-1.0, 1.0);
        Ok(())
    }
    pub fn depth(&self, slot: usize) -> Option<f64> {
        self.depths.get(slot).copied()
    }
    /// Names a route's depth parameter for hosts and GUIs, e.g. "LFO > cutoff" (its id stays
    /// `mod_N_depth`, so saved states keep working when routes are renamed).
    pub fn set_depth_name(&mut self, slot: usize, name: &'static str) -> Result<(), RouteError> {
        let n = self.depth_names.get_mut(slot).ok_or(RouteError::UnknownSlot(slot))?;
        *n = name;
        Ok(())
    }

    /// A parameter's current value including modulation (for a GUI's modulation indicator).
    pub fn modulated_value(&self, index: usize) -> Option<f64> {
        self.pushed.get(index).copied()
    }

    /// Computes every parameter's modulated value from the current sources and routes, and writes
    /// those that changed into the wrapped processor.
    pub fn apply(&mut self) {
        self.offsets.iter_mut().for_each(|o| *o = 0.0);
        for (route, &depth) in self.routes.iter().zip(&self.depths) {
            if let Some(r) = route {
                let via = r.via.map_or(1.0, |v| self.sources[v]);
                self.offsets[r.destination] += depth * self.sources[r.source] * via;
            }
        }
        for i in 0..self.infos.len() {
            let value = self.combined(i);
            if value != self.pushed[i] {
                // the value is inside the range by construction, so this can't fail
                let _ = self.inner.set_param(i, value);
                self.pushed[i] = value;
            }
        }
    }

    fn combined(&self, i: usize) -> f64 {
        let offset = self.offsets[i];
        if offset == 0.0 {
            return self.base[i];
        }
        let info = &self.infos[i];
        info.from_normalized((info.to_normalized(self.base[i]) + offset).clamp(0.0, 1.0))
    }

    /// Drives a block of `frames` samples in control steps of [`RAMP_STEP`]: before each step,
    /// `update_sources(sources, step_len)` refreshes the source values (advance LFOs, read
    /// envelopes), the modulation is applied, and `render(processor, range)` produces that part of
    /// the block.
    pub fn run(
        &mut self,
        frames: usize,
        mut update_sources: impl FnMut(&mut [f64], usize),
        mut render: impl FnMut(&mut P, Range<usize>),
    ) {
        let mut pos = 0;
        while pos < frames {
            let len = RAMP_STEP.min(frames - pos);
            update_sources(&mut self.sources, len);
            self.apply();
            render(&mut self.inner, pos..pos + len);
            pos += len;
        }
    }
}

impl<P: Parameterized> Parameterized for Modulated<P> {
    fn param_count(&self) -> usize {
        self.infos.len() + self.routes.len()
    }
    fn param_info(&self, index: usize) -> Option<ParamInfo> {
        match index.checked_sub(self.infos.len()) {
            None => Some(self.infos[index]),
            Some(slot) if slot < self.routes.len() => {
                Some(ParamInfo::new(DEPTH_IDS[slot], self.depth_names[slot], ParamUnit::Fraction, -1.0, 1.0, 0.0))
            }
            Some(_) => None,
        }
    }
    fn param_group(&self, index: usize) -> Option<&'static str> {
        if index < self.infos.len() {
            self.inner.param_group(index)
        } else {
            (index < self.param_count()).then_some("Modulation")
        }
    }
    /// The base value (what was set), or a route's depth.
    fn get_param(&self, index: usize) -> Option<f64> {
        match index.checked_sub(self.infos.len()) {
            None => Some(self.base[index]),
            Some(slot) => self.depths.get(slot).copied(),
        }
    }
    fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError> {
        let applied = self.param_info(index).ok_or(ParamError::UnknownIndex(index))?.validate(value)?;
        match index.checked_sub(self.infos.len()) {
            None => {
                self.base[index] = applied;
                // takes effect at once (with the current modulation), not only at the next step
                let offset = self.routes.iter().zip(&self.depths).fold(0.0, |sum, (route, &depth)| match route {
                    Some(r) if r.destination == index => sum + depth * self.sources[r.source] * r.via.map_or(1.0, |v| self.sources[v]),
                    _ => sum,
                });
                self.offsets[index] = offset;
                let value = self.combined(index);
                self.pushed[index] = value;
                let _ = self.inner.set_param(index, value);
            }
            Some(slot) => self.depths[slot] = applied,
        }
        Ok(applied)
    }
}

/// With fixed sources (nothing updates them during the block), processing applies the modulation
/// once and runs the whole block.
impl<T: Float, P: Processor<T> + Parameterized> Processor<T> for Modulated<P> {
    fn process(&mut self, block: &mut [T]) {
        self.apply();
        self.inner.process(block);
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
}

impl<T: Float, P: MultiProcessor<T> + Parameterized> MultiProcessor<T> for Modulated<P> {
    fn process(&mut self, buffer: &mut AudioBuffer<T>) {
        self.apply();
        self.inner.process(buffer);
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::Lfo;
    use crate::filter::{Biquad, BUTTERWORTH_Q};
    use crate::params::Smoothed;

    const FS: f64 = 48_000.0;

    fn filter() -> Modulated<Biquad<f64>> {
        Modulated::new(Biquad::lowpass(1_000.0, BUTTERWORTH_Q, FS), 2, 4)
    }

    fn cutoff(m: &Modulated<Biquad<f64>>) -> f64 {
        m.inner().design().unwrap().frequency
    }

    #[test]
    fn depth_moves_a_parameter_in_normalized_units() {
        let mut m = filter();
        let f = m.param_index("frequency_hz").unwrap();
        let info = m.param_info(f).unwrap();
        m.set_route(0, Some(Route { source: 0, destination: f, via: None })).unwrap();
        m.set_depth(0, 0.1).unwrap();
        m.set_source(0, 1.0);
        m.apply();
        let expected = info.from_normalized(info.to_normalized(1_000.0) + 0.1);
        assert!((cutoff(&m) / expected - 1.0).abs() < 1e-12);
        assert_eq!(m.get_param(f), Some(1_000.0), "the base is unchanged");
        assert_eq!(m.modulated_value(f), Some(cutoff(&m)));

        // a second route adds; a via source scales its route
        m.set_route(1, Some(Route { source: 1, destination: f, via: Some(0) })).unwrap();
        m.set_depth(1, -0.2).unwrap();
        m.set_source(1, 0.5);
        m.apply();
        let expected = info.from_normalized(info.to_normalized(1_000.0) + 0.1 - 0.2 * 0.5 * 1.0);
        assert!((cutoff(&m) / expected - 1.0).abs() < 1e-12, "the two cancel");
        assert!((cutoff(&m) - 1_000.0).abs() < 1e-6);

        // clamped at the range ends; disconnecting returns exactly to the base
        m.set_depth(0, 1.0).unwrap();
        m.apply();
        assert_eq!(cutoff(&m), info.max);
        m.set_route(0, None).unwrap();
        m.set_route(1, None).unwrap();
        m.apply();
        assert_eq!(cutoff(&m), 1_000.0);
    }

    #[test]
    fn base_changes_keep_their_modulation() {
        let mut m = filter();
        let f = m.param_index("frequency_hz").unwrap();
        let info = m.param_info(f).unwrap();
        m.set_route(0, Some(Route { source: 0, destination: f, via: None })).unwrap();
        m.set_depth(0, 0.05).unwrap();
        m.set_source(0, -1.0);
        m.set_param(f, 2_000.0).unwrap(); // applies at once, modulation included
        let expected = info.from_normalized(info.to_normalized(2_000.0) - 0.05);
        assert!((cutoff(&m) / expected - 1.0).abs() < 1e-12);
    }

    #[test]
    fn depths_are_parameters_and_routes_are_validated() {
        let mut m = filter();
        let inner = Biquad::<f64>::lowpass(1_000.0, BUTTERWORTH_Q, FS).param_count();
        assert_eq!(m.param_count(), inner + 4);
        let d = m.param_index("mod_2_depth").unwrap();
        assert_eq!(m.param_group(d), Some("Modulation"));
        m.set_param(d, 0.3).unwrap();
        assert_eq!(m.depth(1), Some(0.3));
        assert_eq!(m.param_info(d).unwrap().format(-0.25), "-25%");
        m.set_depth_name(1, "LFO > cutoff").unwrap();
        assert_eq!(m.param_info(d).unwrap().name, "LFO > cutoff");
        assert_eq!(m.param_info(d).unwrap().id, "mod_2_depth", "ids stay stable");
        assert_eq!(m.set_route(9, None), Err(RouteError::UnknownSlot(9)));
        let bad = Route { source: 5, destination: 0, via: None };
        assert_eq!(m.set_route(0, Some(bad)), Err(RouteError::UnknownSource(5)));
        let bad = Route { source: 0, destination: 99, via: None };
        assert_eq!(m.set_route(0, Some(bad)), Err(RouteError::UnknownDestination(99)));
    }

    #[test]
    fn an_lfo_sweeps_a_filter_at_control_rate() {
        let mut m = filter();
        let f = m.param_index("frequency_hz").unwrap();
        m.set_route(0, Some(Route { source: 0, destination: f, via: None })).unwrap();
        m.set_depth(0, 0.2).unwrap();
        let mut lfo = Lfo::new(FS);
        lfo.set_rate(5.0);
        let mut block = vec![0.0; 9_600]; // one LFO cycle
        let mut seen = Vec::new();
        m.run(
            block.len(),
            |sources, len| sources[0] = lfo.advance(len, None),
            |p, range| {
                seen.push(p.design().unwrap().frequency);
                p.process(&mut block[range]);
            },
        );
        assert_eq!(seen.len(), 9_600 / RAMP_STEP, "one update per control step");
        let (lo, hi) = seen.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert!(lo < 1_000.0 && hi > 1_000.0, "sweeps both ways around the base: {lo}..{hi}");
        let info = m.param_info(f).unwrap();
        let ceiling = info.from_normalized(info.to_normalized(1_000.0) + 0.2);
        assert!(hi <= ceiling * (1.0 + 1e-9));
    }

    #[test]
    fn smoothing_wraps_modulation() {
        // host automation ramps the base; the matrix adds modulation on top
        let mut sm = Smoothed::new(filter(), 0.01, FS);
        let f = sm.param_index("frequency_hz").unwrap();
        sm.set_param(f, 4_000.0).unwrap();
        sm.process(&mut [0.0; 480]);
        assert!((sm.inner().inner().design().unwrap().frequency - 4_000.0).abs() < 1e-6);
    }
}
