use crate::anneal::{Annealer, AnnealerInfo, InternalAnnealerInfo, QubitState};
use crate::compiled::CompiledModel;
use crate::wrapper::{Placeholder, Qubit};
use crate::{TcType, TpType, TqType};
use rand::rngs::{OsRng, SmallRng};
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashMap;

pub struct SimpleSolver<'a, Tp, Tq, Tc, T: AnnealerInfo>
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
	pub coeff_strength: f64,
	pub annealer_info: T,
}

impl<'a, Tp, Tq, Tc> SimpleSolver<'a, Tp, Tq, Tc, InternalAnnealerInfo>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
{
	pub fn new(model: &'a CompiledModel<Tp, Tq, Tc>) -> Self {
		Self::with_annealer(model, InternalAnnealerInfo::new())
	}
}

impl<'a, Tp, Tq, Tc, T: AnnealerInfo> SimpleSolver<'a, Tp, Tq, Tc, T>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
{
	pub fn with_annealer(model: &'a CompiledModel<Tp, Tq, Tc>, annealer_info: T) -> Self {
		let qubits = model.get_qubits().into_iter().collect::<Vec<_>>();
		Self {
			model,
			qubits,
			samples: rayon::current_num_threads(),
			// processes: 1,
			iterations: 10,
			generations: 30,
			coeff_strength: 50.0,
			annealer_info,
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
		state: &QubitState,
		c: f64,
		h: &[f64],
		neighbors: &[Vec<(usize, f64)>],
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

impl<'a, Tp, Tq, T: AnnealerInfo> SimpleSolver<'a, Tp, Tq, (), T>
where
	Tp: TpType + Send + Sync,
	Tq: TqType + Send + Sync,
{
	pub fn solve(&self) -> Result<(f64, HashMap<&Tq, bool>), <T as AnnealerInfo>::ErrorType> {
		// Drop constraint missing information
		self.solve_with_constraints().map(|(a, b, _)| (a, b))
	}
}

impl<'a, Tp, Tq, Tc, T: AnnealerInfo> SimpleSolver<'a, Tp, Tq, Tc, T>
where
	Tp: TpType + Send + Sync,
	Tq: TqType + Send + Sync,
	Tc: TcType + Send + Sync,
{
	/// Solve the model using internal annealer.
	pub fn solve_with_constraints(
		&self,
	) -> Result<(f64, HashMap<&Tq, bool>, Vec<&Tc>), <T as AnnealerInfo>::ErrorType> {
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
				// let neighbors = neighbors
				// 	.iter()
				// 	.map(|v| v.deref())
				// 	.collect::<Vec<&[(usize, f64)]>>();
				let fut_ret = std::iter::repeat((h, neighbors))
					.take(self.samples)
					.collect::<Vec<_>>()
					.par_iter()
					.map(|(h, neighbors)| {
						let mut r = SmallRng::from_rng(OsRng).unwrap();
						match self.annealer_info.build(h.clone(), neighbors.clone()) {
							Ok(annealer) => annealer.anneal(&mut r).map(|state| {
								(Self::calculate_energy(&state, c, &h, neighbors), state)
							}),
							Err(e) => Err(e),
						}
					})
					.collect::<Vec<_>>();
				let max =
					fut_ret
						.iter()
						.fold(0.0 / 0.0, |m, v| if let Ok(v) = v { v.0.max(m) } else { m });
				if max.is_infinite() {
					return Err(fut_ret.into_iter().next().unwrap().unwrap_err());
				} else {
					let (energy, state) = fut_ret
						.into_iter()
						.filter(|r| if let Ok((e, _)) = r { *e == max } else { false })
						.next()
						.unwrap()
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
						return Ok(ret.unwrap());
					}
				}
			}
		}
		Ok(ret.unwrap())
	}
}
