extern crate rustqubo;
use rustqubo::solve::SimpleSolver;
use rustqubo::Expr;

#[allow(unused)]
fn run_tsp() {
	#[derive(PartialEq, Eq, Copy, Clone, Debug, Hash, PartialOrd, Ord)]
	struct TspQubit(usize, usize);

	let cities = 5;
	let hmlt_city = (0..cities).into_iter().fold(Expr::zero(), |exp, c| {
		let inner = (0..cities)
			.into_iter()
			.fold(-Expr::one(), |e, o| e + Expr::Binary(TspQubit(c, o)));
		exp + Expr::Constraint {
			label: format!("city {:}", c),
			expr: Box::new(inner ^ 2),
		}
	});
	let hmlt_order = (0..cities).into_iter().fold(Expr::zero(), |exp, o| {
		let inner = (0..cities)
			.into_iter()
			.fold(-Expr::one(), |e, c| e + Expr::Binary(TspQubit(c, o)));
		exp + Expr::Constraint {
			label: format!("order {:}", o),
			expr: Box::new(inner ^ 2),
		}
	});
	let table = [
		[0.0, 5.0, 5.0, 3.0, 4.5],
		[5.0, 0.0, 3.5, 5.0, 7.0],
		[5.0, 3.5, 0.0, 3.0, 4.5],
		[3.0, 5.0, 3.0, 0.0, 2.5],
		[4.5, 7.0, 4.5, 2.5, 0.0],
	];
	let mut hmlt_distance = Expr::zero();
	for i in (0..cities).into_iter() {
		for j in (0..cities).into_iter() {
			for k in (0..cities).into_iter() {
				hmlt_distance = hmlt_distance
					+ table[i][j]
						* Expr::Binary(TspQubit(i, k))
						* Expr::Binary(TspQubit(j, (k + 1) % cities))
			}
		}
	}
	let hmlt = 10.0_f64 * (hmlt_city + hmlt_order) + hmlt_distance;
	let compiled = hmlt.compile();
	let mut solver = SimpleSolver::new(&compiled);
	solver.generations = 10;
	solver.iterations = 1;
	solver.samples = 1;
	let (c, qubits, constraints) = solver.solve_with_constraints().unwrap();
	// println!("{:?} {:?}", qubits, constraints);
	assert!(constraints.len() == 0);
}

#[test]
fn tsp_test() {
	run_tsp();
}

#[test]
fn test() {
	let exp = -10_i32 * Expr::Binary(1) + 5_i32 * Expr::Binary(2) + 12_i32;
	let compiled = exp.compile();
	let solver = SimpleSolver::new(&compiled);
	let (c, sol) = solver.solve().unwrap();
	assert_eq!(sol.get(&1).unwrap(), true);
	assert_eq!(sol.get(&2).unwrap(), false);
	assert_eq!(c, 2);
}
