extern crate rustqubo;
use rustqubo::expr::Expr;
use rustqubo::solve::SimpleSolver;

#[test]
fn test() {
	let exp: Expr<(), _, ()> = Expr::Binary(1) * Expr::Number(-1) + Expr::Number(12);
	let compiled = exp.compile();
	// let compiled = compiled.feed_dict([(a: 1.2), (b: 2.3)].into_iter().collect());
	let solver = SimpleSolver::new(&compiled);
	let (c, qubits, _constraints) = solver.solve();
	assert_eq!(*qubits.get(&1).unwrap(), false);
	assert!((c - 12.0).abs() < 1e-4);
}
