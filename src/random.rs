use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

pub struct RandomState(StdRng);
impl RandomState {
    pub fn new() -> Self { Self(StdRng::from_os_rng()) }
    pub fn normal(&mut self) -> f64 { StandardNormal.sample(&mut self.0) }
}
