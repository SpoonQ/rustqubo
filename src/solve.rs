extern crate classical_solver;

use crate::compiled::CompiledModel;
use crate::wrapper::{Placeholder, Qubit};
use crate::{TcType, TqType};
use annealers::model::{FixedSingleQuadricModel, SingleModel};
use annealers::node::Binary;
use annealers::solution::SingleSolution;
use annealers::solver::{ClassicalSolver, Solver, SolverGenerator, UnstructuredSolverGenerator};
use classical_solver::sa::{SimulatedAnnealer, SimulatedAnnealerGenerator};

use rand::rngs::{OsRng, StdRng};
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::marker::PhantomData;

pub struct SimpleSolver<'a, Tq, Tc, T: UnstructuredSolverGenerator<P>, P: SingleModel, ST: Solver>
where
	Tq: TqType,
	Tc: TcType,
{
	model: &'a CompiledModel<(), Tq, Tc>,
	qubits: Vec<&'a Qubit<Tq>>,
	_phantom: PhantomData<(P, ST)>,
	pub iterations: usize,
	pub samples: usize,
	// pub processes: usize,
	pub generations: usize,
	pub coeff_strength: f64,
	pub solver_generator: T,
}

impl<'a, Tq, Tc>
	SimpleSolver<
		'a,
		Tq,
		Tc,
		SimulatedAnnealerGenerator<FixedSingleQuadricModel<Binary<f64>>>,
		FixedSingleQuadricModel<Binary<f64>>,
		SimulatedAnnealer<FixedSingleQuadricModel<Binary<f64>>, f64>,
	> where
	Tq: TqType,
	Tc: TcType,
{
	pub fn new(model: &'a CompiledModel<(), Tq, Tc>) -> Self {
		Self::with_solver(model, SimulatedAnnealerGenerator::new())
	}
}

impl<'a, Tq, Tc, T: UnstructuredSolverGenerator<P>, P: SingleModel>
	SimpleSolver<'a, Tq, Tc, T, P, T::SolverType>
where
	Tq: TqType,
	Tc: TcType,
{
	pub fn with_solver(model: &'a CompiledModel<(), Tq, Tc>, solver_generator: T) -> Self {
		let qubits = model.get_qubits().into_iter().collect::<Vec<_>>();
		Self {
			model,
			qubits,
			samples: rayon::current_num_threads(),
			iterations: 10,
			generations: 30,
			coeff_strength: 50.0,
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
		T: UnstructuredSolverGenerator<FixedSingleQuadricModel<Binary<f64>>, SolverType = ST>,
		ST: ClassicalSolver<SolutionType = SingleSolution<Binary<f64>>, ErrorType = T::ErrorType>,
	> SimpleSolver<'a, Tq, (), T, FixedSingleQuadricModel<Binary<f64>>, ST>
where
	Tq: TqType + Send + Sync,
{
	pub fn solve(
		&self,
	) -> Result<
		(f64, HashMap<&Tq, bool>),
		<T as SolverGenerator<FixedSingleQuadricModel<Binary<f64>>>>::ErrorType,
	> {
		// Drop constraint missing information
		self.solve_with_constraints().map(|(a, b, _)| (a, b))
	}
}

impl<
		'a,
		Tq,
		Tc,
		T: UnstructuredSolverGenerator<FixedSingleQuadricModel<Binary<f64>>, SolverType = ST>,
		ST: ClassicalSolver<SolutionType = SingleSolution<Binary<f64>>, ErrorType = T::ErrorType>,
	> SimpleSolver<'a, Tq, Tc, T, FixedSingleQuadricModel<Binary<f64>>, ST>
where
	Tq: TqType + Send + Sync,
	Tc: TcType + Send + Sync,
{
	/// Solve the model using internal annealer.
	pub fn solve_with_constraints(
		&self,
	) -> Result<
		(f64, HashMap<&Tq, bool>, Vec<&Tc>),
		<T as SolverGenerator<FixedSingleQuadricModel<Binary<f64>>>>::ErrorType,
	> {
		let ph = self.model.get_placeholders();
		let mut ret = None;
		for _ in 0..self.iterations {
			let mut phdict: HashMap<&Placeholder<(), Tc>, usize> =
				ph.iter().map(|p| (*p, 10)).collect();
			let mut size = ph.len() * 10;
			let mut old_energy = f64::INFINITY;
			for _ in 0..self.generations {
				let (c, model) = self.model.generate_qubo(&self.qubits, &mut |p| {
					if let Some(cnt) = phdict.get(&p) {
						*cnt as f64 / size as f64 * self.coeff_strength
					} else {
						panic!()
					}
				});
				let fut_ret = std::iter::repeat_with(|| self.solver_generator.generate(&model))
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
					.fold(0.0 / 0.0, |m, v| v.energy.unwrap().min(m));
				assert!(min.is_finite());
				let sol = fut_ret
					.into_iter()
					.filter(|r| r.energy.unwrap() == min)
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
