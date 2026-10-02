use super::*;

#[test]
fn conversions_round_trip() {
    let tf = TransferFunction::new(vec![1.0, 3.0, 2.0], vec![1.0, 6.0, 11.0, 6.0], Domain::Continuous);
    let zpk = tf.to_zpk().unwrap();
    let back = zpk.to_tf();
    for (a, b) in back.num.iter().zip(&tf.num) {
        assert!((a - b).abs() < 1e-12);
    }
    for (a, b) in back.den.iter().zip(&tf.den) {
        assert!((a - b).abs() < 1e-12);
    }
    let ss = tf.to_ss().unwrap();
    let again = ss.to_tf_siso().unwrap();
    // the state-space numerator is padded to the denominator's length
    assert!(again.num[1..].iter().zip(&tf.num).all(|(a, b)| (a - b).abs() < 1e-12));
}

#[test]
fn stability_continuous_and_discrete() {
    let stable = TransferFunction::new(vec![1.0], vec![1.0, 3.0, 2.0], Domain::Continuous); // poles -1, -2
    assert!(stable.is_stable().unwrap());
    assert!((stable.stability_margin().unwrap() - 1.0).abs() < 1e-12);
    let unstable = TransferFunction::new(vec![1.0], vec![1.0, -1.0, 2.0], Domain::Continuous);
    assert!(!unstable.is_stable().unwrap());
    let digital = Zpk::new(vec![], vec![C64::new(0.5, 0.5), C64::new(0.5, -0.5)], 1.0, Domain::sampled(100.0));
    assert!(digital.is_stable());
    assert!((digital.stability_margin() - (1.0 - 0.5f64.sqrt())).abs() < 1e-12);
}

#[test]
fn feedback_and_series() {
    // unity feedback around 1/s: 1/(s + 1)
    let integrator = TransferFunction::new(vec![1.0], vec![1.0, 0.0], Domain::Continuous);
    let one = TransferFunction::new(vec![1.0], vec![1.0], Domain::Continuous);
    let closed = integrator.feedback(&one).unwrap();
    assert_eq!(closed.num, vec![1.0]);
    assert_eq!(closed.den, vec![1.0, 1.0]);
    let twice = integrator.series(&integrator).unwrap();
    assert_eq!(twice.den, vec![1.0, 0.0, 0.0]);
    let sum = integrator.parallel(&one).unwrap(); // 1/s + 1 = (s + 1)/s
    assert_eq!(sum.num, vec![1.0, 1.0]);
}

#[test]
fn step_response_settles_to_the_dc_gain() {
    // 1/(s + 1): y(t) = 1 - e^(-t)
    let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], Domain::Continuous);
    let y = tf.step(51, Some(0.1)).unwrap();
    for (k, v) in y.iter().enumerate() {
        assert!((v - (1.0 - (-0.1 * k as f64).exp())).abs() < 1e-12, "t = {}", 0.1 * k as f64);
    }
    let h = tf.frequency_response(&[0.0, 1.0]);
    assert!((h[0].norm() - 1.0).abs() < 1e-12 && (h[1].norm() - 0.5f64.sqrt()).abs() < 1e-12);
}
