use crate::expanded::Expanded;
use crate::expr::{NumberOrFloat, StaticExpr};
use crate::model::Constraint;
use crate::wrapper::{Builder, Placeholder, Qubit};

use crate::{TcType, TpType, TqType};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Debug)]
pub struct CompiledModel<Tp, Tq, Tc>
where
	Tp: TpType, // Placeholder
	Tq: TqType,
	Tc: TcType,
{
	expanded: Expanded<Tp, Tq, Tc>,
	constraints: Vec<Constraint<Tp, Tq, Tc>>,
	builder: Builder<Tp, Tq>,
}

impl<Tp, Tq, Tc> CompiledModel<Tp, Tq, Tc>
where
	Tp: TpType, // Placeholder
	Tq: TqType,
	Tc: TcType,
{
	pub(crate) fn new(
		expanded: Expanded<Tp, Tq, Tc>,
		constraints: Vec<Constraint<Tp, Tq, Tc>>,
	) -> Self {
		let builder = Builder::new();
		Self {
			expanded,
			constraints,
			builder,
		}
	}

	/// Feed real values to fill the placeholders.
	pub fn feed_dict(mut self, mut dict: HashMap<Tp, NumberOrFloat>) -> Self {
		let dict: HashMap<Placeholder<Tp, Tc>, NumberOrFloat> = dict
			.drain()
			.map(|(k, v)| (Placeholder::Placeholder(k), v))
			.collect();
		self.expanded = self.expanded.feed_dict(&dict);
		self.constraints = self
			.constraints
			.into_iter()
			.map(|cs| cs.feed_dict(&dict))
			.collect();
		self
	}

	fn generate_replace(
		set: &BTreeSet<Qubit<Tq>>,
		builder: &mut Builder<Tp, Tq>,
		p: Option<bool>,
	) -> (Expanded<Tp, Tq, Tc>, Option<Expanded<Tp, Tq, Tc>>) {
		let mut exp = Expanded::new();
		if let Some(p) = p {
			let d = set.len();
			let xs = set.iter().collect::<Vec<_>>();
			if p {
				let n = (d - 1) / 2;
				if d % 2 == 0 {
					for i in 0..n {
						let w = builder.ancilla();
						for j in 0..d {
							exp.insert(
								vec![w.clone(), xs[j].clone()].into_iter().collect(),
								StaticExpr::Number(-2),
							);
						}
						exp.insert(
							Some(w).into_iter().collect(),
							StaticExpr::Number((4 * i - 1) as i32),
						);
					}
				} else {
					{
						let wn = builder.ancilla();
						for j in 0..d {
							exp.insert(
								vec![wn.clone(), xs[j].clone()].into_iter().collect(),
								StaticExpr::Number(-1),
							);
						}
						exp.insert(
							Some(wn).into_iter().collect(),
							StaticExpr::Number((2 * n - 1) as i32),
						);
					}
					for i in 0..n - 1 {
						let w = builder.ancilla();
						for j in 0..d {
							exp.insert(
								vec![w.clone(), xs[j].clone()].into_iter().collect(),
								StaticExpr::Number(-2),
							);
						}
						exp.insert(
							Some(w).into_iter().collect(),
							StaticExpr::Number((4 * i - 1) as i32),
						);
					}
				}
				for i in 0..d {
					for j in i + 1..d {
						exp.insert(
							vec![xs[i].clone(), xs[j].clone()].into_iter().collect(),
							StaticExpr::Number(1),
						);
					}
				}
			} else {
				// a * x_1 * ... * x_d = min a * w  * { x_1 * ... * x_d - (d - 1) }  (a < 0)
				let w = builder.ancilla();
				for x in set.iter() {
					exp.insert(
						vec![w.clone(), x.clone()].into_iter().collect(),
						StaticExpr::Number(1),
					);
				}
				exp.insert(
					Some(w).into_iter().collect(),
					StaticExpr::Number(1 - d as i32),
				);
			}
			(exp, None)
		} else {
			// Cannot determine sign of a
			// x * y -> min{1 + w * (3 - 2x - 2y)}, xyz = a * w
			if let &[x, y] = &set.iter().take(2).collect::<Vec<&Qubit<Tq>>>() as &[&Qubit<Tq>] {
				let w = builder.ancilla();
				exp.insert(Some(w.clone()).into_iter().collect(), StaticExpr::Number(3));
				exp.insert(
					vec![x, &w].into_iter().cloned().collect(),
					StaticExpr::Number(-2),
				);
				exp.insert(
					(vec![y, &w]).into_iter().cloned().collect(),
					StaticExpr::Number(-2),
				);
				exp.insert(
					(vec![x, y]).into_iter().cloned().collect(),
					StaticExpr::Number(1),
				);
				(Expanded::from_qubit(w), Some(exp))
			} else {
				panic!();
			}
		}
	}

	pub(crate) fn get_unsatisfied_constraints(
		&self,
		map: &HashMap<&Qubit<Tq>, bool>,
	) -> Vec<&Constraint<Tp, Tq, Tc>> {
		self.constraints
			.iter()
			.filter(|cc| !cc.is_satisfied(map))
			.collect()
	}

	pub(crate) fn reduce_order(mut self, max_order: usize) -> Self {
		let mut builder = self.builder.clone();
		while self.expanded.get_order() > max_order {
			let mut m = self.expanded.count_qubit_subsets(max_order, 2, None);
			if let Some(max_count) = m.values().map(|nonzero| (*nonzero).get()).max() {
				let sets = m
					.drain()
					.filter_map(|(k, v)| if v.get() == max_count { Some(k) } else { None })
					.collect::<Vec<_>>();
				let max_set_size = sets.iter().map(|(set, _)| set.len()).max().unwrap();
				let (replaced_set, p) = sets
					.into_iter()
					.filter(|(set, _)| set.len() == max_set_size)
					.next()
					.unwrap();
				let replaced_set = replaced_set.into_iter().cloned().collect();
				let (replacing_exp, constraint) =
					Self::generate_replace(&replaced_set, &mut builder, p);
				let mut new_expanded = Expanded::new();
				for mut expanded in self
					.expanded
					.drain()
					.map(|(set, exp)| Expanded::from(set, exp))
				{
					if expanded.is_superset(&replaced_set) {
						expanded = expanded.remove_qubits(&replaced_set);
						expanded *= replacing_exp.clone();
					}
					new_expanded += expanded;
				}
				self.expanded = new_expanded;
				if let Some(constraint) = constraint {
					self.constraints
						.push(Constraint::from_raw(None, constraint.into(), None));
				}
			} else {
				break;
			}
		}
		self.builder = builder;
		self
	}

	pub(crate) fn get_qubits(&self) -> BTreeSet<&Qubit<Tq>> {
		self.expanded.get_qubits()
	}

	pub fn get_placeholders(&self) -> BTreeSet<&Placeholder<Tp, Tc>> {
		self.expanded.get_placeholders()
	}

	pub(crate) fn generate_qubo<F>(
		&self,
		qubits: &[&Qubit<Tq>],
		ph_feedback: &mut F,
	) -> (f64, Vec<f64>, Vec<Vec<(usize, f64)>>)
	where
		F: FnMut(&Placeholder<Tp, Tc>) -> f64,
	{
		self.expanded.generate_qubo(qubits, ph_feedback)
	}
}
