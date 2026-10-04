//! The host-facing parameters ([`Parameterized`](crate::params::Parameterized)) of oversampled processors.

use super::*;
use crate::params::{ParamError, ParamInfo, Parameterized};

/// An oversampled processor has exactly the parameters of the processor inside.
impl<P: Parameterized, T: Float> Parameterized for Oversampled<P, T> {
    fn param_count(&self) -> usize {
        self.inner().param_count()
    }
    fn param_info(&self, index: usize) -> Option<ParamInfo> {
        self.inner().param_info(index)
    }
    fn param_group(&self, index: usize) -> Option<&'static str> {
        self.inner().param_group(index)
    }
    fn get_param(&self, index: usize) -> Option<f64> {
        self.inner().get_param(index)
    }
    fn set_param(&mut self, index: usize, value: f64) -> Result<f64, ParamError> {
        self.inner_mut().set_param(index, value)
    }
}
