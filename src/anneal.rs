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

	#[inline]
	pub fn get(&self, loc: usize) -> bool {
		let bytesize = std::mem::size_of::<u8>();
		assert!(loc < self.len);
		// debug_assert!(
		// 	loc < self.len,
		// 	format_args!("Location {} out of len {}", loc, self.len)
		// );
		(self.state[loc / bytesize] & BITVALUES[loc % bytesize]) > 0
	}

	#[inline]
	pub fn flip(&mut self, loc: usize) {
		let bytesize = std::mem::size_of::<u8>();
		self.state[loc / bytesize] ^= BITVALUES[loc % bytesize];
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

#[derive(Clone)]
pub struct SimpleAnnealer {
	pub sweeps_per_round: usize,
	pub beta_schedule: Vec<f64>,
}

impl SimpleAnnealer {
	pub fn new(sweeps_per_round: usize, beta_schedule: Vec<f64>) -> Self {
		Self {
			sweeps_per_round,
			beta_schedule,
		}
	}

	pub fn run<T: Rng>(
		&self,
		state: &mut QubitState,
		random: &mut T,
		h: &[f64],
		neighbors: &[&[(usize, f64)]],
	) {
		assert_eq!(state.len(), neighbors.len());
		assert_eq!(state.len(), h.len());
		let mut energy_diffs = Vec::with_capacity(state.len());
		for (i, ngs) in neighbors.iter().enumerate() {
			let mut energy_diff = h[i];
			for (j, weight) in ngs.iter() {
				if state.get(*j) {
					energy_diff += weight;
				}
			}
			if state.get(i) {
				energy_diff = -energy_diff;
			}
			energy_diffs.push(energy_diff);
		}
		for beta in self.beta_schedule.iter() {
			for _ in 0..self.sweeps_per_round {
				let threshold = 44.36142 / beta;
				for i in 0..state.len() {
					// println!("{:} {:}", i, beta);
					// println!("{:?}", &state);
					// println!("{:?}", &energy_diffs);
					let ed = energy_diffs[i];
					if ed > threshold {
						continue;
					}
					if ed <= 0.0 || f64::exp(-ed * beta) > random.gen_range(0.0, 1.0) {
						// accept
						state.flip(i);
						let stat = state.get(i);
						for (j, weight) in neighbors[i].iter() {
							if stat != state.get(*j) {
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
