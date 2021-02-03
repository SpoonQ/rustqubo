use crate::TqType;
use annealers::node::SingleNode;
use annealers::solution::SingleSolution;
use std::collections::HashMap;

pub struct SoutionView<Tq: TqType, M: SingleNode>(SingleSolution<M>, HashMap<Tq, usize>);

impl<Tq: TqType, M: SingleNode> std::fmt::Debug for SoutionView<Tq, M> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_map()
			.entries(self.1.iter().map(|(k, v)| (k, self.0.state.get(*v))))
			.finish()
	}
}

impl<Tq: TqType, M: SingleNode> SoutionView<Tq, M> {
	pub(crate) fn new(sol: SingleSolution<M>, map: HashMap<Tq, usize>) -> Self {
		Self(sol, map)
	}

	pub fn occurrences(&self) -> usize {
		self.0.occurrences
	}

	pub fn energy(&self) -> Option<M::RealType> {
		self.0.energy
	}

	pub fn local_field(&self, q: &Tq) -> Option<M::RealType> {
		self.0.local_field.as_ref().map(|v| v[self.1[q]])
	}

	pub fn keys(&self) -> impl Iterator<Item = &Tq> {
		self.1.keys()
	}

	pub fn get(&self, q: &Tq) -> bool {
		self.0.state.get(self.1[q])
	}
}

impl<Tq: TqType, M: SingleNode> std::ops::Index<&Tq> for SoutionView<Tq, M> {
	type Output = bool;

	fn index(&self, key: &Tq) -> &Self::Output {
		if self.get(key) {
			&true
		} else {
			&false
		}
	}
}
