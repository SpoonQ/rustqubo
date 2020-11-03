use crate::anneal::{QubitState, SimpleAnnealer};
use crate::compiled::CompiledModel;
use crate::wrapper::{Placeholder, Qubit};
use crate::{TcType, TpType, TqType};
use rand::rngs::{OsRng, SmallRng};
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::Deref;

pub struct SimpleSolver<'a, Tp, Tq, Tc>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
{
	model: &'a CompiledModel<Tp, Tq, Tc>,
	qubits: Vec<&'a Qubit<Tq>>,
	pub iterations: usize,
	pub samples: usize,
	// pub processes: usize,
	pub generations: usize,
	pub beta_count: usize,
	pub sweeps_per_beta: usize,
	pub coeff_strength: f64,
}

impl<'a, Tp, Tq, Tc> SimpleSolver<'a, Tp, Tq, Tc>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
{
	pub fn new(model: &'a CompiledModel<Tp, Tq, Tc>) -> Self {
		let qubits = model.get_qubits().into_iter().collect::<Vec<_>>();
		println!("{:}", rayon::current_num_threads());
		Self {
			model,
			qubits,
			samples: rayon::current_num_threads(),
			// processes: 1,
			iterations: 10,
			generations: 30,
			beta_count: 100,
			sweeps_per_beta: 30,
			coeff_strength: 50.0,
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

	fn calculate_energy(
		state: &mut QubitState,
		c: f64,
		h: &[f64],
		neighbors: &[&[(usize, f64)]],
	) -> f64 {
		let mut energy = c;
		for (i, (h, neigh)) in h.iter().zip(neighbors.iter()).enumerate() {
			if !state.get(i) {
				continue;
			}
			energy += *h;
			for (j, coeff) in neigh.iter() {
				if i < *j {
					break;
				}
				if state.get(*j) {
					energy += *coeff;
				}
			}
		}
		energy
	}
}

impl<'a, Tp, Tq, Tc> SimpleSolver<'a, Tp, Tq, Tc>
where
	Tp: TpType + Send + Sync,
	Tq: TqType + Send + Sync,
	Tc: TcType + Send + Sync,
{
	/// Solve the model using internal annealer.
	pub fn solve(&self) -> (f64, HashMap<&Tq, bool>, Vec<&Tc>) {
		let ph = self.model.get_placeholders();
		let mut ret = None;
		for _ in 0..self.iterations {
			let mut phdict: HashMap<&Placeholder<Tp, Tc>, usize> =
				ph.iter().map(|p| (*p, 10)).collect();
			let mut size = ph.len() * 10;
			let mut old_energy = f64::INFINITY;
			for _ in 0..self.generations {
				let (c, h, neighbors) = self.model.generate_qubo(&self.qubits, &mut |p| {
					if let Some(cnt) = phdict.get(&p) {
						*cnt as f64 / size as f64 * self.coeff_strength
					} else {
						panic!()
					}
				});
				let neighbors = neighbors
					.iter()
					.map(|v| v.deref())
					.collect::<Vec<&[(usize, f64)]>>();
				let (beta_min, beta_max) = Self::generate_beta_range(&h, &neighbors);
				let beta_schedule =
					Self::generate_beta_schedule(beta_min, beta_max, self.beta_count);
				let fut_ret = std::iter::repeat((&h, &neighbors))
					.take(self.samples)
					.collect::<Vec<_>>()
					.par_iter()
					.map(|(h, neighbors)| {
						let annealer =
							SimpleAnnealer::new(self.sweeps_per_beta, beta_schedule.clone());
						let mut r = SmallRng::from_rng(OsRng).unwrap();
						let mut state = QubitState::new_random(self.qubits.len(), &mut r);
						annealer.run(&mut state, &mut r, &h, &neighbors);
						(Self::calculate_energy(&mut state, c, &h, &neighbors), state)
					})
					.collect::<Vec<_>>();
				let max = fut_ret.iter().fold(0.0 / 0.0, |m, v| v.0.max(m));
				let (energy, state) = fut_ret
					.into_iter()
					.filter(|(e, _)| *e == max)
					.next()
					.unwrap();
				if old_energy < energy {
					continue;
				}
				old_energy = energy;
				let ans: HashMap<&Qubit<Tq>, bool> = self
					.qubits
					.iter()
					.enumerate()
					.map(|(i, q)| (*q, state.get(i)))
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
					energy,
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
					return ret.unwrap();
				}
			}
		}
		ret.unwrap()
	}

	fn generate_beta_schedule(beta_min: f64, beta_max: f64, count: usize) -> Vec<f64> {
		let r = f64::ln(beta_max / beta_min) / (count as f64 - 1.0);
		(0..count)
			.map(|index| beta_min * f64::exp(index as f64 * r))
			.collect()
	}

	fn generate_beta_range(h: &[f64], neighbors: &[&[(usize, f64)]]) -> (f64, f64) {
		let eg_min = h
			.iter()
			.chain(neighbors.iter().flat_map(|sl| sl.iter().map(|(_, f)| f)))
			.map(|f| f64::abs(*f))
			.fold(0.0 / 0.0 as f64, |p: f64, n: f64| n.max(p));
		let eg_max = h
			.iter()
			.enumerate()
			.map(|(index, h)| {
				*h + neighbors[index]
					.iter()
					.map(|(_, f)| f64::abs(*f) as f64)
					.sum::<f64>() as f64
			})
			.fold(0.0 / 0.0 as f64, |p: f64, n: f64| n.max(p));
		if eg_max.is_finite() && eg_min.is_finite() {
			(f64::ln(2.0) / eg_max, f64::ln(100.0) / eg_min)
		} else {
			(0.1, 1.0)
		}
	}
}
