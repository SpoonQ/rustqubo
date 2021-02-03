extern crate classical_solver;

use crate::compiled::CompiledModel;
use crate::wrapper::{Placeholder, Qubit};
use crate::{TcType, TqType};
use annealers::model::{FixedSingleQuadricModel, SingleModelView};
use annealers::node::Binary;
use annealers::solution::SingleSolution;
use annealers::solver::{ClassicalSolver, Solver, SolverGenerator, UnstructuredSolverGenerator};
use annealers::variable::Real;
use classical_solver::sa::{SimulatedAnnealer, SimulatedAnnealerGenerator};

use rand::rngs::{OsRng, StdRng};
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::marker::PhantomData;
pub struct SimpleSolver<
	'a,
	Tq: TqType,
	Tc: TcType,
	T: UnstructuredSolverGenerator<'static, P>,
	P: SingleModelView,
	ST: Solver,
	R: Real,
> {
	model: &'a CompiledModel<(), Tq, Tc, R>,
	qubits: Vec<&'a Qubit<Tq>>,
	_phantom: PhantomData<(P, ST)>,
	pub iterations: usize,
	pub samples: usize,
	// pub processes: usize,
	pub generations: usize,
	pub coeff_strength: R,
	pub solver_generator: T,
}

impl<'a, Tq, Tc, R: Real>
	SimpleSolver<
		'a,
		Tq,
		Tc,
		SimulatedAnnealerGenerator<'static, FixedSingleQuadricModel<Binary<R>>>,
		FixedSingleQuadricModel<Binary<R>>,
		SimulatedAnnealer<'static, FixedSingleQuadricModel<Binary<R>>, R>,
		R,
	> where
	Tq: TqType,
	Tc: TcType,
{
	pub fn new(model: &'a CompiledModel<(), Tq, Tc, R>) -> Self {
		Self::with_solver(model, SimulatedAnnealerGenerator::new())
	}
}

impl<'a, Tq, Tc, T: UnstructuredSolverGenerator<'static, P>, P: SingleModelView, R: Real>
	SimpleSolver<'a, Tq, Tc, T, P, T::SolverType, R>
where
	Tq: TqType,
	Tc: TcType,
{
	pub fn with_solver(model: &'a CompiledModel<(), Tq, Tc, R>, solver_generator: T) -> Self {
		let qubits = model.get_qubits().into_iter().collect::<Vec<_>>();
		Self {
			model,
			qubits,
			samples: rayon::current_num_threads(),
			iterations: 10,
			generations: 30,
			coeff_strength: R::from_i32(50),
			solver_generator,
			_phantom: PhantomData,
		}
	}

	pub fn get_qubits(&self) -> Vec<&'a Tq> {
		self.qubits
			.iter()
			.filter_map(|q| {
				if let Qubit::Qubit(q) = q {
					Some(q)
				} else {
					None
				}
			})
			.collect()
	}
}

// TODO: implement where ST: AsyncSolver
impl<
		'a,
		Tq,
		T: UnstructuredSolverGenerator<'static, FixedSingleQuadricModel<Binary<R>>, SolverType = ST>,
		ST: ClassicalSolver<SolutionType = SingleSolution<Binary<R>>, ErrorType = T::ErrorType>,
		R: Real,
	> SimpleSolver<'a, Tq, (), T, FixedSingleQuadricModel<Binary<R>>, ST, R>
where
	Tq: TqType + Send + Sync,
{
	pub fn solve(
		&self,
	) -> Result<
		(R, HashMap<&Tq, bool>),
		<T as SolverGenerator<'static, FixedSingleQuadricModel<Binary<R>>>>::ErrorType,
	> {
		// Drop constraint missing information
		self.solve_with_constraints().map(|(a, b, _)| (a, b))
	}
}

impl<
		'a,
		Tq,
		Tc,
		T: UnstructuredSolverGenerator<'static, FixedSingleQuadricModel<Binary<R>>, SolverType = ST>,
		ST: ClassicalSolver<SolutionType = SingleSolution<Binary<R>>, ErrorType = T::ErrorType>,
		R: Real,
	> SimpleSolver<'a, Tq, Tc, T, FixedSingleQuadricModel<Binary<R>>, ST, R>
where
	Tq: TqType + Send + Sync,
	Tc: TcType + Send + Sync,
{
	/// Solve the model using internal annealer.
	pub fn solve_with_constraints(
		&self,
	) -> Result<
		(R, HashMap<&Tq, bool>, Vec<&Tc>),
		<T as SolverGenerator<'static, FixedSingleQuadricModel<Binary<R>>>>::ErrorType,
	> {
		let ph = self.model.get_placeholders();
		let mut ret = None;
		for _ in 0..self.iterations {
			let mut phdict: HashMap<&Placeholder<(), Tc>, usize> =
				ph.iter().map(|p| (*p, 10)).collect();
			let mut size = ph.len() * 10;
			let mut old_energy = R::MAX;
			for _ in 0..self.generations {
				let (c, model) = self.model.generate_qubo(&self.qubits, &mut |p| {
					if let Some(cnt) = phdict.get(&p) {
						R::from_i32(*cnt as i32) / R::from_i32(size as i32) * self.coeff_strength
					} else {
						panic!()
					}
				});
				let fut_ret = std::iter::repeat_with(|| {
					self.solver_generator.generate(unsafe {
						// SAFETY: model lives longer than solver
						std::mem::transmute(&model as *const FixedSingleQuadricModel<_>)
					})
				})
				.take(self.samples)
				.collect::<Result<Vec<_>, _>>()?
				.par_iter()
				.map(|solver| {
					let mut r = StdRng::from_rng(OsRng).unwrap();
					solver.solve_with_rng(&mut r).map(|v| v.into_iter())
				})
				.collect::<Result<Vec<_>, _>>()?
				.into_iter()
				.flat_map(std::convert::identity)
				.map(|sol| sol.with_energy(&model))
				.collect::<Vec<_>>();
				let min: f64 = fut_ret
					.iter()
					.fold(0.0 / 0.0, |m, v| v.energy.unwrap().as_f64().min(m));
				assert!(min.is_finite());
				let sol = fut_ret
					.into_iter()
					.filter(|r| r.energy.unwrap().as_f64() == min)
					.next()
					.unwrap();
				let energy = sol.energy.unwrap();
				// println!("{}, {}, {}", min, old_energy, energy);
				if old_energy <= energy {
					continue;
				}
				old_energy = energy;
				let ans: HashMap<&Qubit<Tq>, bool> = self
					.qubits
					.iter()
					.enumerate()
					.map(|(i, q)| (*q, sol[i]))
					.collect();
				let mut constraint_labels = Vec::new();
				for c in self.model.get_unsatisfied_constraints(&ans) {
					if let Some(ph) = &c.placeholder {
						if let Some(point) = phdict.get_mut(ph) {
							*point += 1;
							size += 1;
						}
					}
					if let Some(label) = &c.label {
						constraint_labels.push(label);
					}
				}
				let is_satisfied = constraint_labels.len() == 0;
				ret = Some((
					energy + c,
					ans.into_iter()
						.filter_map(|(q, b)| {
							if let Qubit::Qubit(q) = q {
								Some((q, b))
							} else {
								None
							}
						})
						.collect(),
					constraint_labels,
				));
				if is_satisfied {
					return Ok(ret.unwrap());
				}
			}
		}
		Ok(ret.unwrap())
	}
}
