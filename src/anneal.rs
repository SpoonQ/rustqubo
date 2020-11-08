use rand::Rng;

pub struct QubitState {
	state: Vec<u8>,
	len: usize,
}

static BITVALUES: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

impl QubitState {
	#[inline]
	pub fn new_random<T: Rng>(len: usize, r: &mut T) -> Self {
		let bytesize = std::mem::size_of::<u8>();
		let size = (len + bytesize - 1) / bytesize;
		let mut v = Vec::with_capacity(size);
		unsafe {
			v.set_len(size);
		}
		r.fill_bytes(&mut v);
		Self { state: v, len }
	}

	#[inline]
	pub fn len(&self) -> usize {
		self.len
	}

	#[allow(unused)]
	#[inline]
	pub fn get(&self, loc: usize) -> bool {
		let bytesize = std::mem::size_of::<u8>();
		assert!(loc < self.len);
		(self.state[loc / bytesize] & BITVALUES[loc % bytesize]) > 0
	}

	#[inline]
	pub unsafe fn get_unchecked(&self, loc: usize) -> bool {
		let bytesize = std::mem::size_of::<u8>();
		(self.state.get_unchecked(loc / bytesize) & BITVALUES.get_unchecked(loc % bytesize)) > 0
	}

	#[allow(unused)]
	#[inline]
	pub fn flip(&mut self, loc: usize) {
		let bytesize = std::mem::size_of::<u8>();
		self.state[loc / bytesize] ^= BITVALUES[loc % bytesize];
	}

	#[inline]
	pub unsafe fn flip_unchecked(&mut self, loc: usize) {
		let bytesize = std::mem::size_of::<u8>();
		*self.state.get_unchecked_mut(loc / bytesize) ^= BITVALUES.get_unchecked(loc % bytesize);
	}
}

impl std::fmt::Debug for QubitState {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		for i in 0..self.len {
			if self.get(i) {
				f.write_str("1")?;
			} else {
				f.write_str("0")?;
			}
		}
		Ok(())
	}
}

pub trait AnnealerInfo: std::marker::Send + std::marker::Sync {
	type AnnealerType: Annealer;
	fn build(&self, h: Vec<f64>, neighbors: Vec<Vec<(usize, f64)>>) -> Self::AnnealerType;
}

pub trait Annealer: std::marker::Send + std::marker::Sync {
	fn anneal<T: Rng>(&self, r: &mut T) -> QubitState;
}

#[derive(Clone)]
pub enum BetaType {
	Count(usize),
	CountRange(usize, f64, f64),
	Schedule(Vec<f64>),
}

impl BetaType {
	fn generate_beta_range(h: &[f64], neighbors: &[Vec<(usize, f64)>]) -> (f64, f64) {
		let eg_min = h
			.iter()
			.chain(neighbors.iter().flat_map(|sl| sl.iter().map(|(_, f)| f)))
			.map(|f| f64::abs(*f))
			.fold(0.0 / 0.0 as f64, |p: f64, n: f64| n.max(p));
		let eg_max = h
			.iter()
			.enumerate()
			.map(|(index, h)| {
				*h + neighbors[index]
					.iter()
					.map(|(_, f)| f64::abs(*f) as f64)
					.sum::<f64>() as f64
			})
			.fold(0.0 / 0.0 as f64, |p: f64, n: f64| n.max(p));
		if eg_max.is_finite() && eg_min.is_finite() {
			(f64::ln(2.0) / eg_max, f64::ln(100.0) / eg_min)
		} else {
			(0.1, 1.0)
		}
	}

	fn generate_beta_schedule(beta_min: f64, beta_max: f64, count: usize) -> Vec<f64> {
		let r = f64::ln(beta_max / beta_min) / (count as f64 - 1.0);
		(0..count)
			.map(|index| beta_min * f64::exp(index as f64 * r))
			.collect()
	}

	pub fn generate_schedule(&self, h: &[f64], neighbors: &[Vec<(usize, f64)>]) -> Vec<f64> {
		match self {
			BetaType::Schedule(v) => v.clone(),
			BetaType::Count(count) | BetaType::CountRange(count, _, _) => {
				let (min, max) = if let BetaType::CountRange(_, min, max) = self {
					(*min, *max)
				} else {
					Self::generate_beta_range(h, neighbors)
				};
				Self::generate_beta_schedule(min, max, *count)
			}
		}
	}
}

#[derive(Clone)]
pub struct InternalAnnealerInfo {
	pub sweeps_per_round: usize,
	pub beta: BetaType,
}

#[derive(Clone)]
pub struct InternalAnnealer {
	pub sweeps_per_round: usize,
	pub beta_schedule: Vec<f64>,
	h: Vec<f64>,
	neighbors: Vec<Vec<(usize, f64)>>,
}

impl InternalAnnealerInfo {
	pub fn new() -> Self {
		Self {
			sweeps_per_round: 30,
			beta: BetaType::Count(100),
		}
	}
}

impl AnnealerInfo for InternalAnnealerInfo {
	type AnnealerType = InternalAnnealer;
	fn build(&self, h: Vec<f64>, neighbors: Vec<Vec<(usize, f64)>>) -> Self::AnnealerType {
		let beta_schedule = self.beta.generate_schedule(&h, &neighbors);
		InternalAnnealer {
			sweeps_per_round: self.sweeps_per_round,
			h,
			neighbors,
			beta_schedule,
		}
	}
}

impl InternalAnnealer {
	fn run<T: Rng>(
		&self,
		state: &mut QubitState,
		random: &mut T,
		h: &[f64],
		neighbors: &[Vec<(usize, f64)>],
	) {
		assert_eq!(state.len(), neighbors.len());
		assert_eq!(state.len(), h.len());
		let mut energy_diffs = Vec::with_capacity(state.len());
		for (i, ngs) in neighbors.iter().enumerate() {
			let mut energy_diff = unsafe { *h.get_unchecked(i) };
			for (j, weight) in ngs.iter() {
				if unsafe { state.get_unchecked(*j) } {
					energy_diff += weight;
				}
			}
			if unsafe { state.get_unchecked(i) } {
				energy_diff = -energy_diff;
			}
			energy_diffs.push(energy_diff);
		}
		for beta in self.beta_schedule.iter() {
			for _ in 0..self.sweeps_per_round {
				let threshold = 44.36142 / beta;
				for i in 0..state.len() {
					let ed = energy_diffs[i];
					if ed > threshold {
						continue;
					}
					if ed <= 0.0 || f64::exp(-ed * beta) > random.gen_range(0.0, 1.0) {
						// accept
						unsafe {
							state.flip_unchecked(i);
						}
						let stat = unsafe { state.get_unchecked(i) };
						for (j, weight) in unsafe { neighbors.get_unchecked(i) }.iter() {
							if stat != unsafe { state.get_unchecked(*j) } {
								energy_diffs[*j] += weight;
							} else {
								energy_diffs[*j] -= weight;
							}
						}
						energy_diffs[i] *= -1.0;
					}
				}
			}
		}
	}
}

impl Annealer for InternalAnnealer {
	fn anneal<T: Rng>(&self, r: &mut T) -> QubitState {
		let mut state = QubitState::new_random(self.h.len(), r);
		self.run(&mut state, r, &self.h, &self.neighbors);
		state
	}
}

#[cfg(features = "external-apis")]
mod external_apis {
	extern crate cpython;
	use cpython::{PyDict, PyList, PyResult, Python};

	#[cfg(features = "d-wave")]
	mod d_wave {
		pub struct DWaveAnnealer {
			adapter: PythonAdapter,
			pub endpoint: String,
			pub token: Option<String>,
			pub machine: String,
			pub num_reads: usize,
		}

		impl DWaveAnnealer {
			pub fn new() -> Self {
				Self {
					adapter: PythonAdapter::new(),
					endpoint: "https://cloud.dwavesys.com/sapi".to_owned(),
					token: None,
					machine: "DW_2000Q_5".to_owned(),
					num_reads: 100,
				}
			}

			pub fn run(&self, h: &[f64], neighbors: &[&[(usize, f64)]]) {}
		}
	}

	#[cfg(features = "d-wave")]
	pub use self::d_wave::*;
}

#[cfg(features = "external-apis")]
pub use self::external_apis::*;
