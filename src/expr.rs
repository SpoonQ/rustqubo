use crate::compiled::CompiledModel;
use crate::model::Model;
use crate::wrapper::Placeholder;
use crate::{TcType, TpType, TqType};
use annealers::variable::{ConvertFrom, ConvertTo, ConvertWith, Real};
use std::collections::{BTreeSet, HashMap};
use std::mem::MaybeUninit;
use std::ops::{Add, AddAssign, BitXor, Mul, MulAssign, Neg, Sub, SubAssign};

// TODO: hide the implementation from public
#[derive(PartialEq, Clone, Debug)]
pub enum Expr<Tp, Tq, Tc, R>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
	R: Real,
{
	Placeholder(Tp), // The real value of placeholder must be positive
	Add(Box<Self>, Box<Self>),
	Mul(Box<Self>, Box<Self>),
	Number(R),
	// TODO: use annealers_rust::node::{Spin,Binary}
	Binary(Tq), // Qubit represented with +1, 0
	Spin(Tq),   // Qubit represented with +1, -1
	Constraint { label: Tc, expr: Box<Self> },
	WithPenalty { expr: Box<Self>, penalty: Box<Self> },
}

impl<Tp, Tq, Tc, R> Expr<Tp, Tq, Tc, R>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
	R: Real,
	Self: Mul<Self, Output = Self>,
{
	pub fn map<F>(self, f: &mut F) -> Self
	where
		F: FnMut(Self) -> Self,
	{
		match f(self) {
			Self::Add(a, b) => Self::Add(Box::new(a.map(f)), Box::new(b.map(f))),
			Self::Mul(a, b) => Self::Mul(Box::new(a.map(f)), Box::new(b.map(f))),
			Self::Constraint { label, expr } => Self::Constraint {
				label,
				expr: Box::new(expr.map(f)),
			},
			Self::WithPenalty { expr, penalty } => Self::WithPenalty {
				expr: Box::new(expr.map(f)),
				penalty: Box::new(penalty.map(f)),
			},
			o => o,
		}
	}
	pub fn feed_dict(self, dict: &HashMap<Tp, R>) -> Self {
		match self {
			Self::Placeholder(p) => {
				if let Some(val) = dict.get(&p) {
					Self::Number(*val)
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

	pub(crate) fn calculate(&self, map: &HashMap<&Tq, bool>) -> Option<R> {
		match self {
			Self::Placeholder(_) => None,
			Self::Add(lhs, rhs) => {
				if let (Some(lhs), Some(rhs)) = (lhs.calculate(map), rhs.calculate(map)) {
					Some(lhs + rhs)
				} else {
					None
				}
			}
			Self::Mul(lhs, rhs) => match (lhs.calculate(map), rhs.calculate(map)) {
				(Some(lhs), Some(rhs)) => Some(lhs * rhs),
				(Some(e), None) | (None, Some(e)) => {
					if e.as_f64() == 0.0 {
						Some(R::from_i32(0))
					} else {
						None
					}
				}
				_ => None,
			},
			Self::Number(n) => Some(*n),
			Self::Binary(lb) | Self::Spin(lb) => {
				if let Some(b) = map.get(lb) {
					if *b {
						Some(R::from_i32(1))
					} else {
						if let Self::Spin(_) = self {
							Some(R::from_i32(-1))
						} else {
							Some(R::from_i32(0))
						}
					}
				} else {
					None
				}
			}
			Self::Constraint { label: _, expr: e } => e.calculate(map),
			Self::WithPenalty {
				expr: e,
				penalty: _,
			} => e.calculate(map),
		}
	}

	pub fn compile(self) -> CompiledModel<Tp, Tq, Tc, R> {
		self.to_model().to_compiled().reduce_order(2)
	}

	fn map_number<R2: ConvertFrom<R>>(self) -> Expr<Tp, Tq, Tc, R2> {
		match self {
			Self::Number(n) => Expr::Number(<R2 as ConvertFrom<R>>::convert_from(n)),
			Self::Add(a, b) => Expr::Add(Box::new(a.map_number()), Box::new(b.map_number())),
			Self::Mul(a, b) => Expr::Mul(Box::new(a.map_number()), Box::new(b.map_number())),
			Self::Constraint { label, expr } => Expr::Constraint {
				label,
				expr: Box::new(expr.map_number()),
			},
			Self::WithPenalty { expr, penalty } => Expr::WithPenalty {
				expr: Box::new(expr.map_number()),
				penalty: Box::new(penalty.map_number()),
			},
			Self::Placeholder(a) => Expr::Placeholder(a),
			Self::Binary(a) => Expr::Binary(a),
			Self::Spin(a) => Expr::Spin(a),
		}
	}

	pub(crate) fn map_label<Tpn, Fp, Tqn, Fq>(
		self,
		fp: &mut Fp,
		fq: &mut Fq,
	) -> Expr<Tpn, Tqn, Tc, R>
	where
		Fp: FnMut(Tp) -> Tpn,
		Fq: FnMut(Tq) -> Tqn,
		Tpn: TpType,
		Tqn: TqType,
	{
		match self {
			Self::Placeholder(lb) => Expr::Placeholder(fp(lb)),
			Self::Add(lhs, rhs) => Expr::Add(
				Box::new(lhs.map_label(fp, fq)),
				Box::new(rhs.map_label(fp, fq)),
			),
			Self::Mul(lhs, rhs) => Expr::Mul(
				Box::new(lhs.map_label(fp, fq)),
				Box::new(rhs.map_label(fp, fq)),
			),
			Self::Number(n) => Expr::Number(n),
			Self::Binary(lb) => Expr::Binary(fq(lb)),
			Self::Spin(lb) => Expr::Spin(fq(lb)),
			Self::Constraint { label: _, expr: _ }
			| Self::WithPenalty {
				expr: _,
				penalty: _,
			} => panic!("cannot map on Constraint | WithPenalty"),
		}
	}
	pub(crate) fn to_model(self) -> Model<Tp, Tq, Tc, R> {
		match self {
			Self::Placeholder(lb) => {
				Model::from(StaticExpr::Placeholder(Placeholder::Placeholder(lb)))
			}
			Self::Add(lhs, rhs) => lhs.to_model() + rhs.to_model(),
			Self::Mul(lhs, rhs) => lhs.to_model() * rhs.to_model(),
			Self::Number(n) => Model::from(StaticExpr::Number(n)),
			Self::Binary(lb) => Model::from(lb),
			Self::Spin(lb) => (Expr::Number(R::from_i32(2)) * (Expr::Binary(lb))
				- (Expr::Number(R::from_i32(1))))
			.to_model(),
			Self::Constraint { label: lb, expr: e } => {
				let ph: Model<Tp, Tq, Tc, R> =
					Model::from(StaticExpr::Placeholder(Placeholder::Constraint(lb.clone())));
				(e.clone().to_model() * ph.clone()).add_constraint(
					lb.clone(),
					*e,
					Some(Placeholder::Constraint(lb)),
				)
			}
			Self::WithPenalty {
				expr: e,
				penalty: p,
			} => e.to_model().add_penalty(p.to_model()),
		}
	}
}

impl<Tp, Tq, Tc, R> From<R> for Expr<Tp, Tq, Tc, R>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
	R: Real,
{
	#[inline]
	fn from(f: R) -> Self {
		Expr::Number(f)
	}
}

impl<Tp, Tq, Tc, R> From<StaticExpr<Tp, R>> for Expr<Tp, Tq, Tc, R>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
	R: Real,
{
	fn from(from: StaticExpr<Tp, R>) -> Self {
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
					Self::Number(R::from_i32(0))
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
					Self::Number(R::from_i32(1))
				}
			}
			StaticExpr::Number(n) => Self::Number(n),
		}
	}
}

impl<Tp, Tq, Tc, R> Neg for Expr<Tp, Tq, Tc, R>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
	R: Real,
{
	type Output = Self;
	#[inline]
	fn neg(self) -> Self::Output {
		Self::Mul(Box::new(Expr::Number(R::from_i32(-1))), Box::new(self))
	}
}

impl<Tp, Tq, Tc, R> Add<Expr<Tp, Tq, Tc, R>> for Expr<Tp, Tq, Tc, R>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
	R: Real,
{
	type Output = Expr<Tp, Tq, Tc, R>;
	#[inline]
	fn add(self, other: Expr<Tp, Tq, Tc, R>) -> Self::Output {
		Expr::Add(Box::new(self.into()), Box::new(other.into()))
	}
}

impl<Tp, Tq, Tc, R> Sub<Expr<Tp, Tq, Tc, R>> for Expr<Tp, Tq, Tc, R>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
	R: Real,
{
	type Output = Expr<Tp, Tq, Tc, R>;
	#[inline]
	fn sub(self, other: Expr<Tp, Tq, Tc, R>) -> Self::Output {
		Expr::Add(
			Box::new(self.into()),
			Box::new(Expr::Mul(
				Box::new(Expr::Number(R::from_i32(-1))),
				Box::new(other.into()),
			)),
		)
	}
}

impl<Tp, Tq, Tc, R> Mul<Expr<Tp, Tq, Tc, R>> for Expr<Tp, Tq, Tc, R>
where
	Tp: TpType,
	Tq: TqType,
	Tc: TcType,
	R: Real,
{
	type Output = Expr<Tp, Tq, Tc, R>;
	#[inline]
	fn mul(self, other: Expr<Tp, Tq, Tc, R>) -> Self::Output {
		Expr::Mul(Box::new(self.into()), Box::new(other.into()))
	}
}

macro_rules! impl_binary_op {
	($real: ty) => {
		impl_binary_op!(Expr<Tp, Tq, Tc, $real>, $real, $real);
		impl_binary_op!($real, Expr<Tp, Tq, Tc, $real>, $real);
	};
	($lhs:ty, $rhs:ty, $real: ty) => {
		impl<Tp, Tq, Tc> Add<$rhs> for $lhs
		where
			Tp: TpType,
			Tq: TqType,
			Tc: TcType,
		{
			type Output = Expr<Tp, Tq, Tc, $real>;
			#[inline]
			fn add(self, other: $rhs) -> Self::Output {
				Expr::Add(Box::new(self.into()), Box::new(other.into()))
			}
		}

		impl<Tp, Tq, Tc> Sub<$rhs> for $lhs
		where
			Tp: TpType,
			Tq: TqType,
			Tc: TcType,
		{
			type Output = Expr<Tp, Tq, Tc, $real>;
			#[inline]
			fn sub(self, other: $rhs) -> Self::Output {
				Expr::Add(
					Box::new(self.into()),
					Box::new(Expr::Mul(
						Box::new(Expr::Number(<$real>::from_i32(-1))),
						Box::new(other.into()),
					)),
				)
			}
		}

		impl<Tp, Tq, Tc> Mul<$rhs> for $lhs
		where
			Tp: TpType,
			Tq: TqType,
			Tc: TcType,
		{
			type Output = Expr<Tp, Tq, Tc, $real>;
			#[inline]
			fn mul(self, other: $rhs) -> Self::Output {
				Expr::Mul(Box::new(self.into()), Box::new(other.into()))
			}
		}
	};
}

impl_binary_op!(i8);
impl_binary_op!(i16);
impl_binary_op!(i32);
impl_binary_op!(i64);
impl_binary_op!(i128);
impl_binary_op!(f32);
impl_binary_op!(f64);
// macro_rules! impl_binary_op {
// 	($lhs:ty, $rhs:ty,$out:ty) => {
// 		impl<Tp, Tq, Tc> Add<Into<Expr<Tp, Tq, Tc, $rhs>>> for Expr<Tp, Tq, Tc, $lhs>
// 		where
// 			Tp: TpType,
// 			Tq: TqType,
// 			Tc: TcType,
// 		{
// 			type Output = Expr<Tp, Tq, Tc, $out>;
// 			#[inline]
// 			fn add(self, other: $rhs) -> Self::Output {
// 				Expr::Add(
// 					Box::new(<$lhs as Into<Self::Output>>::into(self)),
// 					Box::new(<$rhs as Into<Self::Output>>::into(other)),
// 				)
// 			}
// 		}
// 		impl<Tp, Tq, Tc> Sub<$rhs> for $lhs
// 		where
// 			Tp: TpType,
// 			Tq: TqType,
// 			Tc: TcType,
// 		{
// 			type Output = Expr<Tp, Tq, Tc, $out>;
// 			#[inline]
// 			fn sub(self, other: $rhs) -> Self::Output {
// 				Expr::Add(
// 					Box::new(<$lhs as Into<Self::Output>>::into(self)),
// 					Box::new(-<$rhs as Into<Self::Output>>::into(other)),
// 				)
// 			}
// 		}
// 		impl<Tp, Tq, Tc> Mul<$rhs> for $lhs
// 		where
// 			Tp: TpType,
// 			Tq: TqType,
// 			Tc: TcType,
// 		{
// 			type Output = Expr<Tp, Tq, Tc, $out>;
// 			#[inline]
// 			fn mul(self, other: $rhs) -> Self::Output {
// 				Expr::Mul(
// 					Box::new(<$lhs as Into<Self::Output>>::into(self)),
// 					Box::new(<$rhs as Into<Self::Output>>::into(other)),
// 				)
// 			}
// 		}
// 	};
// }
//
// macro_rules! impl_binary_op_twice {
// 	($lhs:ty, $rhs:ty,$out:ty) => {
// 		impl_binary_op!($lhs, $rhs, $out);
// 		impl_binary_op!($rhs, $lhs, $out);
// 	};
// }
//
// impl_binary_op!(Expr<Tp, Tq, Tc, i8  >, Self,i8    );
// impl_binary_op!(Expr<Tp, Tq, Tc, i16 >, Self,i16  );
// impl_binary_op!(Expr<Tp, Tq, Tc, i32 >, Self,i32  );
// impl_binary_op!(Expr<Tp, Tq, Tc, i64 >, Self,i64  );
// impl_binary_op!(Expr<Tp, Tq, Tc, i128>, Self,i128);
// impl_binary_op!(Expr<Tp, Tq, Tc, f32 >, Self,f32 );
// impl_binary_op!(Expr<Tp, Tq, Tc, f64 >, Self,f64 );
//
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i8>, i8, i8);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i8>, i16, i16);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i8>, i32, i32);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i8>, i64, i64);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i8>, i128, i128);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i8>, f32, f32);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i8>, f64, f64);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i16>, i8, i16);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i16>, i16, i16);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i16>, i32, i32);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i16>, i64, i64);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i16>, i128, i128);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i16>, f32, f32);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i16>, f64, f64);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i32>, i8, i32);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i32>, i16, i32);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i32>, i32, i32);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i32>, i64, i64);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i32>, i128, i128);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i32>, f32, f32);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i32>, f64, f64);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i64>, i8, i64);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i64>, i16, i64);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i64>, i32, i64);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i64>, i64, i64);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i64>, i128, i128);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i64>, f32, f32);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i64>, f64, f64);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i128>, i8, i128);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i128>, i16, i128);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i128>, i32, i128);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i128>, i64, i128);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i128>, i128, i128);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i128>, f32, f32);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, i128>, f64, f64);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, f32>, i8, f32);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, f32>, i16, f32);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, f32>, i32, f32);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, f32>, i64, f32);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, f32>, i128, f32);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, f32>, f32, f32);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, f32>, f64, f64);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, f64>, i8, f64);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, f64>, i16, f64);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, f64>, i32, f64);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, f64>, i64, f64);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, f64>, i128, f64);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, f64>, f32, f64);
// impl_binary_op_twice!(Expr<Tp, Tq, Tc, f64>, f64, f64);

macro_rules! impl_bitxor {
	($typ:ty) => {
		impl<Tp, Tq, Tc> BitXor<usize> for Expr<Tp, Tq, Tc, $typ>
		where
			Tp: TpType,
			Tq: TqType,
			Tc: TcType,
		{
			type Output = Self;
			#[inline]
			fn bitxor(self, other: usize) -> Self {
				let mut hmlt = Expr::Number(<$typ as Real>::from_i32(1));
				for _ in 0..other {
					hmlt = hmlt * self.clone();
				}
				hmlt
			}
		}
	};
}

impl_bitxor!(i8);
impl_bitxor!(i16);
impl_bitxor!(i32);
impl_bitxor!(i64);
impl_bitxor!(i128);
impl_bitxor!(f32);
impl_bitxor!(f64);

// macro_rules! impl_binary_op {
// 	($typ:ty) => {
// 		impl<Tp, Tq, Tc, R> Add<$typ> for Expr<Tp, Tq, Tc, R>
// 		where
// 			Tp: TpType,
// 			Tq: TqType,
// 			Tc: TcType,
// 			R: ConvertWith<$typ>,
// 		{
// 			type Output = Expr<Tp, Tq, Tc, <R as ConvertWith<$typ>>::Output>;
// 			#[inline]
// 			fn add(self, other: $typ) -> Self::Output {
// 				Expr::Add(
// 					Box::new(self.map_number()),
// 					Box::new((<R as ConvertWith<$typ>>::convert_rhs(other)).into()),
// 				)
// 			}
// 		}
// 		impl<Tp, Tq, Tc, R> Add<Expr<Tp, Tq, Tc, R>> for $typ
// 		where
// 			Tp: TpType,
// 			Tq: TqType,
// 			Tc: TcType,
// 			R: ConvertWith<$typ>,
// 		{
// 			type Output = Expr<Tp, Tq, Tc, <R as ConvertWith<$typ>>::Output>;
// 			#[inline]
// 			fn add(self, other: Expr<Tp, Tq, Tc, R>) -> Self::Output {
// 				Expr::Add(
// 					Box::new((<R as ConvertWith<$typ>>::convert_rhs(self)).into()),
// 					Box::new(other.map_number()),
// 				)
// 			}
// 		}
// 		impl<Tp, Tq, Tc, R> Sub<$typ> for Expr<Tp, Tq, Tc, R>
// 		where
// 			Tp: TpType,
// 			Tq: TqType,
// 			Tc: TcType,
// 			R: ConvertWith<$typ>,
// 		{
// 			type Output = Expr<Tp, Tq, Tc, <R as ConvertWith<$typ>>::Output>;
// 			#[inline]
// 			fn sub(self, other: $typ) -> Self::Output {
// 				Expr::Add(
// 					Box::new(self.map_number()),
// 					Box::new(Expr::Mul(
// 						Box::new(<R as ConvertWith<$typ>>::from_i32(-1).into()),
// 						(<R as ConvertWith<$typ>>::convert_rhs(other)).into(),
// 					)),
// 				)
// 			}
// 		}
// 		impl<Tp, Tq, Tc, R> Sub<Expr<Tp, Tq, Tc, R>> for $typ
// 		where
// 			Tp: TpType,
// 			Tq: TqType,
// 			Tc: TcType,
// 			R: ConvertWith<$typ>,
// 		{
// 			type Output = Expr<Tp, Tq, Tc, <R as ConvertWith<$typ>>::Output>;
// 			#[inline]
// 			fn sub(self, other: Expr<Tp, Tq, Tc, R>) -> Self::Output {
// 				Expr::Add(
// 					Box::new((<R as ConvertWith<$typ>>::convert_rhs(self)).into()),
// 					Box::new(Expr::Mul(
// 						Box::new(<R as ConvertWith<$typ>>::from_i32(-1).into()),
// 						Box::new(other.map_number()),
// 					)),
// 				)
// 			}
// 		}
// 		impl<Tp, Tq, Tc, R> Mul<$typ> for Expr<Tp, Tq, Tc, R>
// 		where
// 			Tp: TpType,
// 			Tq: TqType,
// 			Tc: TcType,
// 			R: ConvertWith<$typ>,
// 		{
// 			type Output = Expr<Tp, Tq, Tc, <R as ConvertWith<$typ>>::Output>;
// 			#[inline]
// 			fn mul(self, other: $typ) -> Self::Output {
// 				Expr::Mul(
// 					Box::new(self.map_number()),
// 					Box::new((<R as ConvertWith<$typ>>::convert_rhs(other)).into()),
// 				)
// 			}
// 		}
// 		impl<Tp, Tq, Tc, R> Mul<Expr<Tp, Tq, Tc, R>> for $typ
// 		where
// 			Tp: TpType,
// 			Tq: TqType,
// 			Tc: TcType,
// 			R: ConvertWith<$typ>,
// 		{
// 			type Output = Expr<Tp, Tq, Tc, <R as ConvertWith<$typ>>::Output>;
// 			#[inline]
// 			fn mul(self, other: Expr<Tp, Tq, Tc, R>) -> Self::Output {
// 				Expr::Mul(
// 					Box::new((<R as ConvertWith<$typ>>::convert_rhs(self)).into()),
// 					Box::new(other.map_number()),
// 				)
// 			}
// 		}
// 	};
// }
//
// impl_binary_op!(i8);
// impl_binary_op!(i16);
// impl_binary_op!(i32);
// impl_binary_op!(i64);
// impl_binary_op!(i128);
// impl_binary_op!(f32);
// impl_binary_op!(f64);

macro_rules! impl_assign_op {
	($trait:ident, $trait_inner:ident, $fun:ident, $fun_inner:ident) => {
		// impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, Self);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i8, i8);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i8, i16);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i8, i32);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i8, i64);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i8, i128);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i8, f32);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i8, f64);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i16, i16);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i16, i32);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i16, i64);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i16, i128);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i16, f32);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i16, f64);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i32, i32);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i32, i64);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i32, i128);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i32, f32);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i32, f64);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i64, i64);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i64, i128);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i64, f32);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i64, f64);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i128, i128);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i128, f32);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, i128, f64);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, f32, f32);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, f32, f64);
		impl_assign_op_inner!($trait, $trait_inner, $fun, $fun_inner, f64, f64);
	};
}

// impl_assign_op!(AddAssign, Add, add_assign, add);
// impl_assign_op!(SubAssign, Sub, sub_assign, sub);
// impl_assign_op!(MulAssign, Mul, mul_assign, mul);

#[derive(PartialEq, Clone, Debug)]
pub(crate) enum StaticExpr<Tp, R>
where
	Tp: TpType,
{
	Placeholder(Tp),
	Add(Vec<Self>),
	Mul(Vec<Self>),
	Number(R),
}

#[test]
fn expand_simplify_test() {
	#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
	struct S(i32);
	// impl TpType for S {}
	// impl crate::LabelType for S {}

	fn get_ph(n: i32) -> StaticExpr<S, i32> {
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

impl<Tp, Tc, R> StaticExpr<Placeholder<Tp, Tc>, R>
where
	Tp: TpType,
	Tc: TcType,
	R: Real,
{
	pub(crate) fn drop_placeholder(self) -> StaticExpr<Placeholder<(), Tc>, R> {
		match self {
			Self::Placeholder(p) => StaticExpr::Placeholder(p.drop_placeholder()),
			Self::Add(v) => StaticExpr::Add(v.into_iter().map(|a| a.drop_placeholder()).collect()),
			Self::Mul(v) => StaticExpr::Mul(v.into_iter().map(|a| a.drop_placeholder()).collect()),
			Self::Number(a) => StaticExpr::Number(a),
		}
	}
}

impl<Tp, R> StaticExpr<Tp, R>
where
	Tp: TpType,
	R: Real,
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
		match &self {
			Self::Add(v) | Self::Mul(v) => {
				let v = if is_add {
					Self::expand_add(self)
				} else {
					Self::expand_mul(self)
				};
				let mut val = None;
				let mut v = v
					.into_iter()
					.filter_map(|exp| match exp.simplify() {
						Self::Number(n) => {
							if let Some(v) = val {
								val = Some(if is_add { v + n } else { v * n });
							} else {
								val = Some(n);
							}
							None
						}
						o => Some(o),
					})
					.collect::<Vec<_>>();
				if let Some(val) = val {
					v.push(Self::Number(val));
				}
				println!("v  {:?}", &v);
				if v.len() == 1 {
					v.pop().unwrap()
				} else if is_add {
					Self::Add(v)
				} else {
					Self::Mul(v)
				}
			}
			_ => self,
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
			Self::Number(n) => Some(n.as_f64() > 0.0),
			Self::Placeholder(_) => Some(true),
		}
	}

	pub(crate) fn feed_dict(self, dict: &HashMap<Tp, R>) -> Self {
		match self {
			Self::Placeholder(p) => {
				if let Some(val) = dict.get(&p) {
					Self::Number(*val)
				} else {
					Self::Placeholder(p)
				}
			}
			Self::Add(v) => Self::Add(v.into_iter().map(|item| item.feed_dict(dict)).collect()),
			Self::Mul(v) => Self::Mul(v.into_iter().map(|item| item.feed_dict(dict)).collect()),
			o => o,
		}
	}

	pub(crate) fn calculate<F>(&self, ph_feedback: &mut F) -> R
	where
		F: FnMut(&Tp) -> R,
	{
		match self {
			Self::Placeholder(p) => {
				let f = ph_feedback(p);
				assert!(f.as_f64() >= 0.0);
				f
			}
			Self::Add(v) => v.iter().map(|item| item.calculate(ph_feedback)).sum(),
			Self::Mul(v) => v.iter().map(|item| item.calculate(ph_feedback)).product(),
			Self::Number(n) => *n,
		}
	}
}
//
// #[derive(Debug, Copy, Clone)]
// enum NumberOrFloatInner {
// 	Number(i32),
// 	Float(f64),
// }
//
// #[derive(Debug, Copy, Clone)]
// pub struct NumberOrFloat(NumberOrFloatInner);
//
// impl NumberOrFloat {
// 	fn get_number(&self) -> Option<i32> {
// 		match self.0 {
// 			NumberOrFloatInner::Number(n) => Some(n),
// 			_ => None,
// 		}
// 	}
//
// 	fn get_float(&self) -> f64 {
// 		match self.0 {
// 			NumberOrFloatInner::Number(n) => n as f64,
// 			NumberOrFloatInner::Float(f) => f,
// 		}
// 	}
// }
//
// impl Default for NumberOrFloat {
// 	fn default() -> Self {
// 		Self(NumberOrFloatInner::Number(0))
// 	}
// }
//
// impl From<i32> for NumberOrFloat {
// 	fn from(i: i32) -> Self {
// 		if i < 0 {
// 			panic!("Placeholder value must be positive.");
// 		}
// 		Self(NumberOrFloatInner::Number(i))
// 	}
// }
//
// impl From<f64> for NumberOrFloat {
// 	fn from(f: f64) -> Self {
// 		if f < 0.0 {
// 			panic!("Placeholder value must be positive.");
// 		}
// 		Self(NumberOrFloatInner::Float(f))
// 	}
// }
