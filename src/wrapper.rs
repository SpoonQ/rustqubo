use crate::{TpType, TqType};

#[derive(Clone, Debug)]
pub struct Builder<Tp, Tq>
where
	Tp: TpType,
	Tq: TqType,
{
	ancillas: usize,
	placeholders: usize,
	_phantom: std::marker::PhantomData<(Tp, Tq)>,
}

impl<Tp, Tq> Builder<Tp, Tq>
where
	Tp: TpType,
	Tq: TqType,
{
	pub fn new() -> Self {
		Self {
			ancillas: 0,
			placeholders: 0,
			_phantom: std::marker::PhantomData,
		}
	}

	// pub fn placeholder(&mut self) -> Placeholder<Tp> {
	// 	self.placeholders += 1;
	// 	Placeholder::Internal
	// }

	pub fn ancilla(&mut self) -> Qubit<Tq>
	where
		Tq: TqType,
	{
		self.ancillas += 1;
		Qubit::Ancilla(self.ancillas - 1)
	}
}

#[derive(Debug, PartialEq, Eq, Clone, Hash, Ord, PartialOrd)]
pub enum Qubit<Tq>
where
	Tq: TqType,
{
	Qubit(Tq),
	Ancilla(usize),
}

// impl<Tq> TqType for Qubit<Tq> where Tq: TqType {}
// impl<Tq> crate::LabelType for Qubit<Tq> where Tq: TqType {}

impl<Tq> Qubit<Tq>
where
	Tq: TqType,
{
	pub(crate) fn new(ltq: Tq) -> Self {
		Self::Qubit(ltq)
	}
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub enum Placeholder<Tp>
where
	Tp: TpType,
{
	Placeholder(Tp),
	Internal,
}

// impl<Tp> TpType for Placeholder<Tp> where Tp: TpType {}
// impl<Tp> crate::LabelType for Placeholder<Tp> where Tp: TpType {}

impl<Tp> std::fmt::Debug for Placeholder<Tp>
where
	Tp: TpType,
{
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			Self::Placeholder(p) => p.fmt(f),
			Self::Internal => f.write_str("(Internal)"),
		}
	}
}
