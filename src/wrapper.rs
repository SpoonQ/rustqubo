use crate::{TcType, TpType, TqType};

#[derive(Clone, Debug)]
pub struct Builder<Tq>
where
	Tq: TqType,
{
	ancillas: usize,
	_phantom: std::marker::PhantomData<Tq>,
}

impl<Tq> Builder<Tq>
where
	Tq: TqType,
{
	pub fn new() -> Self {
		Self {
			ancillas: 0,
			_phantom: std::marker::PhantomData,
		}
	}

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

impl<Tp, Tc> Placeholder<Tp, Tc>
where
	Tp: TpType,
	Tc: TcType,
{
	pub(crate) fn drop_placeholder(self) -> Placeholder<(), Tc> {
		match self {
			Self::Placeholder(p) => {
				panic!("Placeholder {:?} must be fulfilled.", &p)
			}
			Self::Constraint(c) => Placeholder::Constraint(c),
		}
	}
}
