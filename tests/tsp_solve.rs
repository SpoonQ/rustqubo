extern crate rustqubo;
use rustqubo::expr::Expr;
use rustqubo::solve::SimpleSolver;

#[allow(unused)]
fn run_tsp(cities: usize) {
	#[derive(PartialEq, Eq, Copy, Clone, Debug, Hash, PartialOrd, Ord)]
	struct TspQubit(usize, usize);

	let H_city: Expr<(), _, _> = (0..cities).into_iter().fold(Expr::Number(0), |exp, c| {
		let inner = (0..cities)
			.into_iter()
			.fold(Expr::Number(-1), |e, o| e + Expr::Binary(TspQubit(c, o)));
		exp + Expr::Constraint {
			label: format!("city {:}", c),
			expr: Box::new(inner.clone() * inner),
		}
	});
	let H_order = (0..cities).into_iter().fold(Expr::Number(0), |exp, o| {
		let inner = (0..cities)
			.into_iter()
			.fold(Expr::Number(-1), |e, c| e + Expr::Binary(TspQubit(c, o)));
		exp + Expr::Constraint {
			label: format!("order {:}", o),
			expr: Box::new(inner.clone() * inner),
		}
	});
	let mut H_distance = Expr::Number(0);
	for i in (0..cities).into_iter() {
		for j in (0..cities).into_iter() {
			for k in (0..cities).into_iter() {
				let d_ij = Expr::Number(10);
				H_distance = H_distance
					+ d_ij
						* Expr::Binary(TspQubit(k, i))
						* Expr::Binary(TspQubit((k + 1) % cities, j))
			}
		}
	}
	let H = H_city + H_order + H_distance;
	let compiled = H.compile();
	let solver = SimpleSolver::new(&compiled);
	let (c, qubits, constraints) = solver.solve();
	println!("{:?}", constraints);
	assert!(constraints.len() == 0);
}

#[test]
fn tsp_test() {
	run_tsp(100);
}

#[test]
#[ignore]
fn test() {
	let exp: Expr<(), _, ()> = Expr::Binary(1) * Expr::Number(-1) + Expr::Number(12);
	let compiled = exp.compile();
	// let compiled = compiled.feed_dict([(a: 1.2), (b: 2.3)].into_iter().collect());
	let solver = SimpleSolver::new(&compiled);
	let (c, qubits, _constraints) = solver.solve();
	assert_eq!(*qubits.get(&1).unwrap(), true);
	assert!((c - 12.0).abs() < 1e-4);
}
