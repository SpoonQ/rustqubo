use crate::{TcType, TpType, TqType};

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

impl<Tq> Qubit<Tq>
where
	Tq: TqType,
{
	pub(crate) fn new(ltq: Tq) -> Self {
		Self::Qubit(ltq)
	}
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Hash, Debug)]
pub enum Placeholder<Tp, Tc>
where
	Tp: TpType,
	Tc: TcType,
{
	Placeholder(Tp),
	Constraint(Tc),
}
