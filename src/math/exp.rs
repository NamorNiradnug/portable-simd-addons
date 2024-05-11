pub trait Exponent {
    /// Returns exponent, i.e. `e^(self)`, of each lane.
    fn exp(self) -> Self;
    /// Returns `2^(self)` of each lane.
    fn exp2(self) -> Self;
    /// Returns `e^(self) - 1` of each lane.
    fn exp_m1(self) -> Self;
}
