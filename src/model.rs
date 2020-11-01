use crate::compiled::CompiledModel;
use crate::expanded::Expanded;
use crate::expr::{Expr, NumberOrFloat};
use crate::wrapper::{Placeholder, Qubit};
use crate::{TcType, TpType, TqType};
use std::collections::HashMap;
use std::convert::From;
use std::ops::{Add, Mul};

#[derive(Clone, Debug)]
pub struct Model<Tp, Tq, Tc>
where
	Tp: TpType, // Placeholder
	Tq: TqType,
	Tc: TcType,
{
	expanded: Expanded<Tp, Tq, Tc>,
	penalties: Expanded<Tp, Tq, Tc>,
	constraints: Vec<Constraint<Tp, Tq, Tc>>,
}

impl<Tp, Tq, Tc> Model<Tp, Tq, Tc>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
{
	pub fn new() -> Self {
		Self {
			expanded: Expanded::new(),
			penalties: Expanded::new(),
			constraints: Vec::new(),
		}
	}

	#[inline]
	pub fn add_penalty(mut self, other: Self) -> Self {
		self.penalties += other.penalties + other.expanded;
		self.constraints.extend_from_slice(&other.constraints);
		self
	}

	#[inline]
	pub fn add_constraint(
		mut self,
		lb: Tc,
		e: Expr<Tp, Tq, Tc>,
		ph: Option<Placeholder<Tp, Tc>>,
	) -> Self {
		self.constraints.push(Constraint::new(lb, e, ph));
		self
	}

	pub fn to_compiled(self) -> CompiledModel<Tp, Tq, Tc> {
		CompiledModel::new(self.expanded + self.penalties, self.constraints)
	}
}

impl<Tp, Tq, Tc, Q> From<Q> for Model<Tp, Tq, Tc>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
	Q: Into<Expanded<Tp, Tq, Tc>>,
{
	fn from(q: Q) -> Self {
		let mut ret = Model::new();
		ret.expanded = q.into();
		ret
	}
}

impl<Tp, Tq, Tc, RHS> Add<RHS> for Model<Tp, Tq, Tc>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
	RHS: Into<Self>,
{
	type Output = Self;
	#[inline]
	fn add(mut self, other: RHS) -> Self::Output {
		let other = other.into();
		self.expanded += other.expanded;
		self.penalties += other.penalties;
		self.constraints.extend_from_slice(&other.constraints);
		self
	}
}

impl<Tp, Tq, Tc, RHS> Mul<RHS> for Model<Tp, Tq, Tc>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
	RHS: Into<Self>,
{
	type Output = Self;
	#[inline]
	fn mul(mut self, other: RHS) -> Self::Output {
		let other = other.into();
		self.expanded *= other.expanded;
		self.penalties += other.penalties;
		self.constraints.extend_from_slice(&other.constraints);
		self
	}
}

#[derive(Clone, Debug)]
pub struct Constraint<Tp, Tq, Tc>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
{
	pub label: Option<Tc>,
	expr: Expr<Placeholder<Tp, Tc>, Qubit<Tq>, Tc>,
	pub placeholder: Option<Placeholder<Tp, Tc>>,
}

impl<Tp, Tq, Tc> Constraint<Tp, Tq, Tc>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
{
	pub fn new(
		label: Tc,
		expr: Expr<Tp, Tq, Tc>,
		placeholder: Option<Placeholder<Tp, Tc>>,
	) -> Self {
		let expr = expr.map_label(&mut |ltp| Placeholder::Placeholder(ltp), &mut |ltq| {
			Qubit::new(ltq)
		});
		Self {
			label: Some(label),
			expr: expr,
			placeholder,
		}
	}

	pub fn is_satisfied(&self, map: &HashMap<&Qubit<Tq>, bool>) -> bool {
		if let Some(i) = self.expr.calculate_i(map) {
			i == 0
		} else if let Some(f) = self.expr.calculate_f(map) {
			f.abs() < 1.0e-4
		} else {
			true
		}
	}

	pub fn feed_dict(mut self, dict: &HashMap<Placeholder<Tp, Tc>, NumberOrFloat>) -> Self {
		self.expr = self.expr.feed_dict(dict);
		if let Some(p) = &self.placeholder {
			if let Some(_) = dict.get(p) {
				self.placeholder = None;
			}
		}
		self
	}

	pub fn from_raw(
		label: Option<Tc>,
		expr: Expr<Placeholder<Tp, Tc>, Qubit<Tq>, Tc>,
		placeholder: Option<Placeholder<Tp, Tc>>,
	) -> Self {
		Self {
			label: label,
			expr: expr,
			placeholder,
		}
	}
}
