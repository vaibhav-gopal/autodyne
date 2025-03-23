use super::*;

/// Complex number in Cartesian form
/// Usually you'll want to use Complex32 and Complex64 instead
#[derive(Eq, PartialEq, Copy, Clone, Debug)]
pub struct Complex<T: RealUnit + PrimitiveUnit>{
    /// Real portion
    pub re: T,
    /// Complex portion
    pub im: T,
}

impl<T: RealUnit + PrimitiveUnit> Complex<T> {
    pub fn new(re: T, im: T) -> Self {
        Complex {re, im}
    }
}

impl <T: RealUnit + PrimitiveUnit> Default for Complex<T> {
    fn default() -> Self {
        Complex::new(T::zero(), T::zero())
    }
}

impl<T: RealUnit + PrimitiveUnit> From<T> for Complex<T> {
    fn from(re: T) -> Complex<T> {
        Complex {re, im: T::zero()}
    }
}

impl<'a, T: RealUnit + PrimitiveUnit> From<&'a T> for Complex<T> {
    fn from (re: &T) -> Complex<T> {
        Complex::from(re.clone())
    }
}

#[macro_export]
macro_rules! complex {
    () => {
        Complex::default()
    };
    ($re:expr, $im:expr) => {
        Complex::new($re, $im)
    };
}

#[macro_export]
macro_rules! re {
    () => {
        Complex::re()
    };
    ($re:expr) => {
        Complex::re().scale($re)
    };
}
#[macro_export]
macro_rules! im {
    () => {
        Complex::im()
    };
    ($im:expr) => {
        Complex::im().scale($im)
    };
}

// UNIT IMPLEMENTATIONS ============================================================================

impl<T: RealUnit + PrimitiveUnit> Unit for Complex<T> {}

impl<T: RealUnit + PrimitiveUnit> ComplexUnit for Complex<T> {
    type Item = T;
    
    fn re() -> Self {
        Complex::new(T::one(), T::zero())
    }

    fn im() -> Self {
        Complex::new(T::zero(), T::one())
    }

    fn norm_sqrt(&self) -> Self::Item {
        todo!()
    }

    fn scale(&self, k: Self::Item) -> Self {
        todo!()
    }

    fn unscale(&self, k: Self::Item) -> Self {
        todo!()
    }

    fn conj(&self) -> Self {
        todo!()
    }

    fn inv(&self) -> Self {
        todo!()
    }

    fn norm(&self) -> Self::Item {
        todo!()
    }

    fn arg(&self) -> Self::Item {
        todo!()
    }

    fn to_polar(&self) -> (Self::Item, Self::Item) {
        todo!()
    }

    fn from_polar(r: Self::Item, theta: Self::Item) -> Self {
        todo!()
    }

    fn exp(&self) -> Self {
        todo!()
    }
}

// OPERATOR OVERLOADS ==============================================================================

impl<T: RealUnit + PrimitiveUnit> Add for Complex<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Complex::new(self.re + rhs.re, self.im + rhs.im)
    }
}
impl<T: RealUnit + PrimitiveUnit> Sub for Complex<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Complex::new(self.re - rhs.re, self.im - rhs.im)
    }
}
impl<T: RealUnit + PrimitiveUnit> Mul for Complex<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Complex::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re
        )
    }
}
impl<T: RealUnit + PrimitiveUnit> Div for Complex<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        let norm_sqrt = rhs.norm_sqrt();
        Complex::new(
            (self.re * rhs.re + self.im * rhs.im) / norm_sqrt,
            (self.im * rhs.re - self.re * rhs.im) / norm_sqrt
        )
    }
}
impl<T: RealUnit + PrimitiveUnit> Neg for Complex<T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Complex::new(-self.re, -self.im)
    }
}
