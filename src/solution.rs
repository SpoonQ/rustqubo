use crate::TqType;
use annealers::node::Binary;
use annealers::solution::SingleSolution;
use annealers::variable::Real;
use std::collections::HashMap;

pub struct SolutionView<Tq: TqType, R: Real>(SingleSolution<Binary<R>>, HashMap<Tq, usize>);

impl<Tq: TqType, R: Real> std::fmt::Debug for SolutionView<Tq, R> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_map()
			.entries(self.1.iter().map(|(k, v)| (k, self.0.state.get(*v))))
			.finish()
	}
}

impl<Tq: TqType, R: Real> SolutionView<Tq, R> {
	pub(crate) fn new(sol: SingleSolution<Binary<R>>, map: HashMap<Tq, usize>) -> Self {
		Self(sol, map)
	}

	pub fn occurrences(&self) -> usize {
		self.0.occurrences
	}

	pub fn energy(&self) -> Option<R> {
		self.0.energy
	}

	pub fn local_field(&self, q: &Tq) -> Option<R> {
		self.0.local_field.as_ref().map(|v| v[self.1[q]])
	}

	pub fn keys(&self) -> impl Iterator<Item = &Tq> {
		self.1.keys()
	}

	pub fn get(&self, q: &Tq) -> Option<bool> {
		if self.1.contains_key(q) {
			Some(self.0.state.get(self.1[q]))
		} else {
			None
		}
	}
}

impl<Tq: TqType, R: Real> std::ops::Index<&Tq> for SolutionView<Tq, R> {
	type Output = bool;

	fn index(&self, key: &Tq) -> &Self::Output {
		if self.get(key).unwrap() {
			&true
		} else {
			&false
		}
	}
}
