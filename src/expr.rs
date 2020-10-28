use crate::compiled::CompiledModel;
use crate::model::Model;
use crate::wrapper::Placeholder;
use crate::{TcType, TpType, TqType};
use std::collections::{BTreeSet, HashMap};
use std::ops::{Add, Mul, Neg, Sub};

// pub trait Expression<Tp, Tq, Tc>
// where
// 	Tp: TpType,
// 	Tq: TqType,
// 	Tc: TcType,
// {
// 	fn to_expr(self) -> Expr<Tp, Tq, Tc>;
//
// 	fn compile(self) -> Model<Tp, Tq, Tc> {
// 		self.to_expr().compile()
// 	}
// }
//
// impl<Tp, Tq, Tc> Expression<Tp, Tq, Tc> for Expr<Tp, Tq, Tc>
// where
// 	Tp: TpType,
// 	Tq: TqType,
// 	Tc: TcType,
// {
// 	fn to_expr(self) -> Self {
// 		self
// 	}
// }

#[derive(PartialEq, Clone, Debug)]
pub enum Expr<Tp, Tq, Tc>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
{
	Placeholder(Tp), // The real value of placeholder must be positive
	Add(Box<Self>, Box<Self>),
	Mul(Box<Self>, Box<Self>),
	Number(i32),
	Float(f64),
	Binary(Tq), // Qubit represented with +1, 0
	Spin(Tq),   // Qubit represented with +1, -1
	Constraint {
		label: Tc,
		expr: Box<Expr<Tp, Tq, Tc>>,
	},
	WithPenalty {
		expr: Box<Self>,
		penalty: Box<Self>,
	},
}

impl<Tp, Tq, Tc> Expr<Tp, Tq, Tc>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
{
	pub fn feed_dict(self, dict: &HashMap<Tp, NumberOrFloat>) -> Self {
		match self {
			Self::Placeholder(p) => {
				if let Some(val) = dict.get(&p) {
					if let Some(n) = val.get_number() {
						Self::Number(n)
					} else {
						Self::Float(val.get_float())
					}
				} else {
					Self::Placeholder(p)
				}
			}
			Self::Add(a, b) => Self::Add(
				Box::new((*a).feed_dict(dict)),
				Box::new((*b).feed_dict(dict)),
			),
			Self::Mul(a, b) => Self::Mul(
				Box::new((*a).feed_dict(dict)),
				Box::new((*b).feed_dict(dict)),
			),
			o => o,
		}
	}

	pub(crate) fn to_model(self) -> Model<Tp, Tq, Tc> {
		match self {
			Self::Placeholder(lb) => StaticExpr::Placeholder(Placeholder::Placeholder(lb)).into(),
			Self::Add(lhs, rhs) => lhs.to_model() + rhs.to_model(),
			Self::Mul(lhs, rhs) => lhs.to_model() * rhs.to_model(),
			Self::Number(n) => (StaticExpr::Number(n)).into(),
			Self::Float(f) => (StaticExpr::Float(f)).into(),
			Self::Binary(lb) => (lb).into(),
			Self::Spin(lb) => (Expr::Binary(lb) * Expr::Number(2) - Expr::Number(1)).to_model(),
			Self::Constraint { label: lb, expr: e } => e.clone().to_model().add_constraint(lb, *e),
			Self::WithPenalty {
				expr: e,
				penalty: p,
			} => e.to_model().add_penalty(p.to_model()),
		}
	}

	pub(crate) fn calculate_i(&self, map: &HashMap<&Tq, bool>) -> Option<i32> {
		match self {
			Self::Placeholder(_) => None,
			Self::Add(lhs, rhs) => {
				if let (Some(lhs), Some(rhs)) = (lhs.calculate_i(map), rhs.calculate_i(map)) {
					Some(lhs + rhs)
				} else {
					None
				}
			}
			Self::Mul(lhs, rhs) => match (lhs.calculate_i(map), rhs.calculate_i(map)) {
				(Some(lhs), Some(rhs)) => Some(lhs * rhs),
				(Some(e), None) | (None, Some(e)) => {
					if e == 0 {
						Some(0)
					} else {
						None
					}
				}
				_ => None,
			},
			Self::Number(n) => Some(*n),
			Self::Float(_) => None,
			Self::Binary(lb) | Self::Spin(lb) => {
				if let Some(b) = map.get(lb) {
					if *b {
						Some(1)
					} else {
						if let Self::Spin(_) = self {
							Some(-1)
						} else {
							Some(0)
						}
					}
				} else {
					None
				}
			}
			Self::Constraint { label: _, expr: e } => e.calculate_i(map),
			Self::WithPenalty {
				expr: e,
				penalty: _,
			} => e.calculate_i(map),
		}
	}

	pub(crate) fn calculate_f(&self, map: &HashMap<&Tq, bool>) -> Option<f64> {
		match self {
			Self::Add(lhs, rhs) => {
				if let (Some(lhs), Some(rhs)) = (lhs.calculate_f(map), rhs.calculate_f(map)) {
					Some(lhs + rhs)
				} else {
					None
				}
			}
			Self::Mul(lhs, rhs) => match (lhs.calculate_f(map), rhs.calculate_f(map)) {
				(Some(lhs), Some(rhs)) => Some(lhs * rhs),
				(Some(e), None) | (None, Some(e)) => {
					if e.abs() < 1.0e-6 {
						Some(0.0)
					} else {
						None
					}
				}
				_ => None,
			},
			Self::Float(f) => Some(*f),
			Self::Constraint { label: _, expr: e } => e.calculate_f(map),
			Self::WithPenalty {
				expr: e,
				penalty: _,
			} => e.calculate_f(map),
			o => o.calculate_i(map).map(|o| o as f64),
		}
	}

	pub fn compile(self) -> CompiledModel<Tp, Tq, Tc> {
		self.to_model().to_compiled().reduce_order(2)
	}

	pub(crate) fn map<Tpn, Fp, Tqn, Fq>(self, fp: &mut Fp, fq: &mut Fq) -> Expr<Tpn, Tqn, Tc>
	where
		Fp: FnMut(Tp) -> Tpn,
		Fq: FnMut(Tq) -> Tqn,
		Tpn: TpType,
		Tqn: TqType,
	{
		match self {
			Self::Placeholder(lb) => Expr::Placeholder(fp(lb)),
			Self::Add(lhs, rhs) => Expr::Add(Box::new(lhs.map(fp, fq)), Box::new(rhs.map(fp, fq))),
			Self::Mul(lhs, rhs) => Expr::Mul(Box::new(lhs.map(fp, fq)), Box::new(rhs.map(fp, fq))),
			Self::Number(n) => Expr::Number(n),
			Self::Float(f) => Expr::Float(f),
			Self::Binary(lb) => Expr::Binary(fq(lb)),
			Self::Spin(lb) => Expr::Spin(fq(lb)),
			Self::Constraint { label: _, expr: _ }
			| Self::WithPenalty {
				expr: _,
				penalty: _,
			} => panic!("cannot map on Constraint | WithPenalty"),
		}
	}
}

impl<Tp, Tq, Tc> From<f64> for Expr<Tp, Tq, Tc>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
{
	#[inline]
	fn from(f: f64) -> Self {
		Expr::Float(f)
	}
}

impl<Tp, Tq, Tc> From<i32> for Expr<Tp, Tq, Tc>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
{
	#[inline]
	fn from(f: i32) -> Self {
		Expr::Number(f)
	}
}

impl<Tp, Tq, Tc> From<StaticExpr<Tp>> for Expr<Tp, Tq, Tc>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
{
	fn from(from: StaticExpr<Tp>) -> Self {
		match from {
			StaticExpr::Placeholder(lb) => Self::Placeholder(lb),
			StaticExpr::Add(mut v) => {
				if let Some(item) = v.pop() {
					if v.len() > 0 {
						Self::Add(Box::new(StaticExpr::Add(v).into()), Box::new(item.into()))
					} else {
						item.into()
					}
				} else {
					Self::Number(0)
				}
			}
			StaticExpr::Mul(mut v) => {
				if let Some(item) = v.pop() {
					if v.len() > 0 {
						Self::Mul(Box::new(StaticExpr::Mul(v).into()), Box::new(item.into()))
					} else {
						item.into()
					}
				} else {
					Self::Number(1)
				}
			}
			StaticExpr::Number(n) => Self::Number(n),
			StaticExpr::Float(f) => Self::Float(f),
		}
	}
}

impl<Tp, Tq, Tc> Neg for Expr<Tp, Tq, Tc>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
{
	type Output = Self;
	#[inline]
	fn neg(self) -> Self::Output {
		Self::Mul(Box::new(Expr::Number(-1)), Box::new(self))
	}
}

impl<Tp, Tq, Tc> Add for Expr<Tp, Tq, Tc>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
{
	type Output = Expr<Tp, Tq, Tc>;
	#[inline]
	fn add(self, other: Self) -> Self::Output {
		Self::Add(Box::new(self), Box::new(other))
	}
}

impl<Tp, Tq, Tc> Sub for Expr<Tp, Tq, Tc>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
{
	type Output = Self;
	#[inline]
	fn sub(self, other: Self) -> Self::Output {
		Self::Add(Box::new(self), Box::new(-other))
	}
}
impl<Tp, Tq, Tc> Mul for Expr<Tp, Tq, Tc>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
{
	type Output = Self;
	#[inline]
	fn mul(self, other: Self) -> Self::Output {
		Self::Mul(Box::new(self), Box::new(other))
	}
}

#[derive(PartialEq, Clone, Debug)]
pub enum StaticExpr<Tp>
where
	Tp: TpType,
{
	Placeholder(Tp),
	Add(Vec<Self>),
	Mul(Vec<Self>),
	Number(i32),
	Float(f64),
}

#[test]
fn expand_simplify_test() {
	#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
	struct S(i32);
	// impl TpType for S {}
	// impl crate::LabelType for S {}

	fn get_ph(n: i32) -> StaticExpr<S> {
		StaticExpr::Placeholder(S(n))
	}

	assert_eq!(
		StaticExpr::Mul(vec![
			StaticExpr::Add(vec![get_ph(1), get_ph(2)]),
			StaticExpr::Add(vec![get_ph(3), get_ph(4)]),
			get_ph(5)
		])
		.expand_add(),
		vec![
			StaticExpr::Mul(vec![get_ph(1), get_ph(3), get_ph(5)]),
			StaticExpr::Mul(vec![get_ph(1), get_ph(4), get_ph(5)]),
			StaticExpr::Mul(vec![get_ph(2), get_ph(3), get_ph(5)]),
			StaticExpr::Mul(vec![get_ph(2), get_ph(4), get_ph(5)])
		]
	)
}

impl<Tp> StaticExpr<Tp>
where
	Tp: TpType,
{
	pub(crate) fn get_placeholders(&self) -> BTreeSet<&Tp> {
		match self {
			Self::Placeholder(p) => Some(p).into_iter().collect(),
			Self::Add(v) | Self::Mul(v) => v
				.iter()
				.flat_map(|item| item.get_placeholders().into_iter())
				.collect(),
			_ => BTreeSet::new(),
		}
	}
	fn get_cross(v: Vec<Vec<Self>>) -> Vec<Vec<Self>> {
		v.into_iter().fold(vec![Vec::new()], |outer, inner| {
			outer
				.iter()
				.flat_map(move |v| {
					inner
						.iter()
						.map(|item| {
							let mut v = v.clone();
							v.push(item.clone());
							v
						})
						.collect::<Vec<_>>()
				})
				.collect()
		})
	}

	pub(crate) fn expand_add(self) -> Vec<Self> {
		match self {
			Self::Add(v) => v.into_iter().flat_map(Self::expand_add).collect(),
			Self::Mul(v) => Self::get_cross(v.into_iter().map(Self::expand_add).collect())
				.into_iter()
				.map(|v| Self::Mul(v))
				.collect(),
			o => vec![o],
		}
	}

	pub(crate) fn expand_mul(self) -> Vec<Self> {
		match self {
			Self::Mul(v) => v.into_iter().flat_map(Self::expand_mul).collect(),
			o => vec![o],
		}
	}

	pub(crate) fn simplify(self) -> Self {
		let is_add = if let Self::Add(_) = &self {
			true
		} else {
			false
		};
		match self {
			Self::Add(v) | Self::Mul(v) => {
				let mut fval = None;
				let mut nval = if is_add { 0i32 } else { 1i32 };
				let v = if is_add {
					Self::expand_add(Self::Add(v))
				} else {
					Self::expand_mul(Self::Mul(v))
				};
				let mut v = v
					.into_iter()
					.filter_map(|exp| match exp.simplify() {
						Self::Number(n) => {
							if is_add {
								nval += n;
							} else {
								nval *= n;
							}
							None
						}
						Self::Float(mut f) => {
							if let Some(ff) = fval {
								if is_add {
									f += ff;
								} else {
									f *= ff;
								}
							}
							fval = Some(f);
							None
						}
						o => Some(o),
					})
					.collect::<Vec<_>>();
				if let Some(mut ff) = fval {
					ff += nval as f64;
					v.push(Self::Float(ff));
				} else if nval != if is_add { 0 } else { 1 } {
					v.push(Self::Number(nval));
				}
				if v.len() == 1 {
					v.pop().unwrap()
				} else if is_add {
					Self::Add(v)
				} else {
					Self::Mul(v)
				}
			}
			o => o,
		}
	}

	pub(crate) fn is_positive(&self) -> Option<bool> {
		match self {
			Self::Add(v) => {
				let mut ret = None;
				for exp in v.iter() {
					if let Some(b) = exp.is_positive() {
						if let Some(bb) = ret {
							if b != bb {
								return None;
							}
						} else {
							ret = Some(b);
						}
					} else {
						return None;
					}
				}
				ret
			}
			Self::Mul(v) => {
				let mut ret = None;
				for exp in v.iter() {
					if let Some(mut b) = exp.is_positive() {
						if let Some(bb) = ret {
							b = b == bb;
						}
						ret = Some(b);
					} else {
						return None;
					}
				}
				ret
			}
			Self::Number(n) => Some(*n > 0),
			Self::Float(f) => Some(*f > 0.0),
			Self::Placeholder(_) => Some(true),
		}
	}

	pub(crate) fn feed_dict(self, dict: &HashMap<Tp, NumberOrFloat>) -> Self {
		match self {
			Self::Placeholder(p) => {
				if let Some(val) = dict.get(&p) {
					if let Some(n) = val.get_number() {
						Self::Number(n)
					} else {
						Self::Float(val.get_float())
					}
				} else {
					Self::Placeholder(p)
				}
			}
			Self::Add(v) => Self::Add(v.into_iter().map(|item| item.feed_dict(dict)).collect()),
			Self::Mul(v) => Self::Mul(v.into_iter().map(|item| item.feed_dict(dict)).collect()),
			o => o,
		}
	}

	pub(crate) fn calculate<F>(&self, ph_feedback: &mut F) -> f64
	where
		F: FnMut(&Tp) -> f64,
	{
		match self {
			Self::Placeholder(p) => {
				let f = ph_feedback(p);
				assert!(f >= 0.0);
				f
			}
			Self::Add(v) => v.iter().map(|item| item.calculate(ph_feedback)).sum(),
			Self::Mul(v) => v
				.iter()
				.fold(1.0, |f, item| f * item.calculate(ph_feedback)),
			Self::Number(n) => *n as f64,
			Self::Float(f) => *f,
		}
	}
}

#[derive(Debug, Copy, Clone)]
enum NumberOrFloatInner {
	Number(i32),
	Float(f64),
}

#[derive(Debug, Copy, Clone)]
pub struct NumberOrFloat(NumberOrFloatInner);

impl NumberOrFloat {
	fn get_number(&self) -> Option<i32> {
		match self.0 {
			NumberOrFloatInner::Number(n) => Some(n),
			_ => None,
		}
	}

	fn get_float(&self) -> f64 {
		match self.0 {
			NumberOrFloatInner::Number(n) => n as f64,
			NumberOrFloatInner::Float(f) => f,
		}
	}
}

impl Default for NumberOrFloat {
	fn default() -> Self {
		Self(NumberOrFloatInner::Number(0))
	}
}

impl From<i32> for NumberOrFloat {
	fn from(i: i32) -> Self {
		if i < 0 {
			panic!("Placeholder value must be positive.");
		}
		Self(NumberOrFloatInner::Number(i))
	}
}

impl From<f64> for NumberOrFloat {
	fn from(f: f64) -> Self {
		if f < 0.0 {
			panic!("Placeholder value must be positive.");
		}
		Self(NumberOrFloatInner::Float(f))
	}
}

// impl<Tp> StaticExpr<Tp>
// where
// 	Tp: TpType,
// {
// 	fn map<Tpn, F>(self, f: &mut F) -> StaticExpr<Tpn>
// 	where
// 		Tpn: TpType,
// 		F: FnMut(Tp) -> Tpn,
// 	{
// 		match self {
// 			Self::Placeholder(p) => Self::Placeholder(f(p)),
// 			_ => unimplemented!(),
// 		}
// 	}
// }