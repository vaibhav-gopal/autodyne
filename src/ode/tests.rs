use super::*;

fn opts(method: OdeMethod, rtol: f64, atol: f64) -> OdeOptions<'static> {
    OdeOptions { method, rtol, atol, ..Default::default() }
}

#[test]
fn dense_output_rows_sum_to_the_weights() {
    // at x = 1 the dense output must land on the step's solution
    for tb in [&RK45, &RK23] {
        for (j, row) in tb.p.iter().enumerate() {
            let b = tb.b.get(j).copied().unwrap_or(0.0);
            assert!((row.iter().sum::<f64>() - b).abs() < 1e-15, "stage {j}");
        }
    }
}

#[test]
fn exponential_decay_is_exact_to_tolerance() {
    for method in [OdeMethod::Rk45, OdeMethod::Rk23, OdeMethod::Rosenbrock23] {
        let sol = solve_ivp(|_t, y, dy| dy[0] = -0.5 * y[0], (0.0, 4.0), &[2.0], opts(method, 1e-8, 1e-10)).unwrap();
        let got = sol.y.last().unwrap()[0];
        let want = 2.0 * (-2.0f64).exp();
        // the second-order Rosenbrock method accumulates more global error per unit of local tolerance
        let tol = if method == OdeMethod::Rosenbrock23 { 1e-4 } else { 1e-6 };
        assert!((got - want).abs() < tol * want, "{method:?}: {got} vs {want}");
        assert_eq!(*sol.t.last().unwrap(), 4.0);
        assert_eq!(sol.status, OdeStatus::Finished);
    }
}

#[test]
fn harmonic_oscillator_with_t_eval_and_backwards() {
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = y[1];
        dy[1] = -y[0];
    };
    let t_eval: Vec<f64> = (0..=20).map(|i| i as f64 * 0.5).collect();
    let sol = solve_ivp(f, (0.0, 10.0), &[1.0, 0.0], OdeOptions { rtol: 1e-9, atol: 1e-12, t_eval: Some(t_eval.clone()), ..Default::default() }).unwrap();
    assert_eq!(sol.t, t_eval);
    for (t, y) in sol.t.iter().zip(&sol.y) {
        // interpolated points too: the dense output is fourth order
        assert!((y[0] - t.cos()).abs() < 1e-7 && (y[1] + t.sin()).abs() < 1e-7, "t {t}: {y:?}");
    }
    // integrating back returns to the start
    let back = solve_ivp(f, (10.0, 0.0), sol.y.last().unwrap(), opts(OdeMethod::Rk45, 1e-10, 1e-12)).unwrap();
    let y0 = back.y.last().unwrap();
    assert!((y0[0] - 1.0).abs() < 1e-7 && y0[1].abs() < 1e-7, "{y0:?}");
}

#[test]
fn stiff_problems_take_few_rosenbrock_steps() {
    // Van der Pol with mu = 1000 over [0, 3000]: explicit methods crawl, Rosenbrock strides
    let mu = 1000.0;
    let vdp = move |_t: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = y[1];
        dy[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
    };
    let stiff = solve_ivp(vdp, (0.0, 3000.0), &[2.0, 0.0], opts(OdeMethod::Rosenbrock23, 1e-4, 1e-6)).unwrap();
    assert_eq!(stiff.status, OdeStatus::Finished);
    assert!(stiff.t.len() < 2000, "{} steps", stiff.t.len());
    // the relaxation oscillation: y0 swings between about ±2
    assert!(stiff.y.iter().all(|y| y[0].abs() < 2.1));
    // a short stretch with RK45 needs far more steps for the same time
    let explicit = solve_ivp(vdp, (0.0, 30.0), &[2.0, 0.0], opts(OdeMethod::Rk45, 1e-4, 1e-6)).unwrap();
    assert!(explicit.t.len() > 10 * stiff.t.iter().filter(|&&t| t <= 30.0).count(), "{} vs {}", explicit.t.len(), stiff.t.len());
}

#[test]
fn events_are_located_and_can_stop_the_integration() {
    // a ball thrown up at 10 m/s from 0: it lands at t = 2 v / g
    let g = 9.81;
    let fall = |_t: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = y[1];
        dy[1] = -g;
    };
    let hit_ground = Event { g: Box::new(|_t, y: &[f64]| y[0]), terminal: true, direction: -1.0 };
    let apex = Event { g: Box::new(|_t, y: &[f64]| y[1]), terminal: false, direction: 0.0 };
    let sol = solve_ivp(fall, (0.0, 10.0), &[0.0, 10.0], OdeOptions { events: vec![hit_ground, apex], rtol: 1e-10, atol: 1e-12, ..Default::default() }).unwrap();
    assert_eq!(sol.status, OdeStatus::Event);
    let land = 2.0 * 10.0 / g;
    assert!((sol.t_events[0][0] - land).abs() < 1e-9, "{:?}", sol.t_events);
    assert!((sol.t_events[1][0] - land / 2.0).abs() < 1e-9);
    assert!((*sol.t.last().unwrap() - land).abs() < 1e-9 && sol.y.last().unwrap()[0].abs() < 1e-8);
    // the start (y = 0, rising) is not an event
    assert_eq!(sol.t_events[0].len(), 1);
}

#[test]
fn rejects_bad_problems() {
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| dy[0] = y[0];
    assert!(solve_ivp(f, (0.0, 1.0), &[], OdeOptions::default()).is_err());
    assert!(solve_ivp(f, (0.0, 1.0), &[1.0], opts(OdeMethod::Rk45, 0.0, 1e-6)).is_err());
    assert!(solve_ivp(f, (0.0, 1.0), &[1.0], OdeOptions { t_eval: Some(vec![0.5, 2.0]), ..Default::default() }).is_err());
}
