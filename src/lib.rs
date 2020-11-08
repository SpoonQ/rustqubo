use std::cmp::Ord;
use std::fmt::Debug;
use std::hash::Hash;

extern crate rand;
extern crate rayon;

pub trait LabelType: PartialEq + Eq + Clone + std::fmt::Debug {}
pub trait TpType: LabelType + Hash + Ord {}
pub trait TqType: LabelType + Hash + Ord {}
pub trait TcType: LabelType + Hash + Ord {}

impl<T> LabelType for T where T: PartialEq + Eq + Clone + Debug {}
impl<T> TpType for T where T: LabelType + Hash + Ord {}
impl<T> TqType for T where T: LabelType + Hash + Ord {}
impl<T> TcType for T where T: LabelType + Hash + Ord {}

#[cfg(feature = "external-apis")]
mod adapter;

mod anneal;
pub mod compiled;
mod expanded;
pub mod expr;
mod model;
pub mod solve;
mod util;
mod wrapper;
