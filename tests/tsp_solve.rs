extern crate rustqubo;
use rustqubo::expr::Expr;
use rustqubo::solve::SimpleSolver;

#[allow(unused)]
fn run_tsp() {
	#[derive(PartialEq, Eq, Copy, Clone, Debug, Hash, PartialOrd, Ord)]
	struct TspQubit(usize, usize);

	let cities = 5usize;
	let H_city = (0..cities).into_iter().fold(Expr::Number(0), |exp, c| {
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
	let table = [
		[0.0, 5.0, 5.0, 3.0, 4.5],
		[5.0, 0.0, 3.5, 5.0, 7.0],
		[5.0, 3.5, 0.0, 3.0, 4.5],
		[3.0, 5.0, 3.0, 0.0, 2.5],
		[4.5, 7.0, 4.5, 2.5, 0.0],
	];
	let mut H_distance = Expr::Number(0);
	for i in (0..cities).into_iter() {
		for j in (0..cities).into_iter() {
			for k in (0..cities).into_iter() {
				let d_ij = Expr::Float(table[i][j]);
				H_distance = H_distance
					+ d_ij
						* Expr::Binary(TspQubit(i, k))
						* Expr::Binary(TspQubit(j, (k + 1) % cities))
			}
		}
	}
	let H: Expr<(), _, _> = Expr::Number(700) * (H_city + H_order) + H_distance;
	let compiled = H.compile();
	let mut solver = SimpleSolver::new(&compiled);
	solver.generations = 1;
	solver.beta_count = 1000;
	solver.sweeps_per_beta = 1;
	solver.samples = 1;
	let (c, qubits, constraints) = solver.solve();
	println!("{:?} {:?}", qubits, constraints);
	assert!(constraints.len() == 0);
}

#[test]
fn tsp_test() {
	run_tsp();
}

#[test]
#[ignore]
fn test() {
	let exp: Expr<(), _, ()> =
		Expr::Binary(1) * Expr::Number(-1) + Expr::Binary(2) + Expr::Number(12);
	let compiled = exp.compile();
	println!("{:?}", &compiled);
	// let compiled = compiled.feed_dict([(a: 1.2), (b: 2.3)].into_iter().collect());
	let solver = SimpleSolver::new(&compiled);
	let (c, qubits, _constraints) = solver.solve();
	assert_eq!(*qubits.get(&1).unwrap(), true);
	assert_eq!(*qubits.get(&2).unwrap(), false);
	assert!((c - 11.0).abs() < 1e-4);
}
