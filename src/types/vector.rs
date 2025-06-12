use super::Rational;
/// Вектор фиксированной размерности с рациональными координатами
///
/// # Примеры
/// ```
/// use linearust::types::{Rational, Vector};
///
/// let v1 = Vector::new([Rational::new(1, 2).unwrap(), Rational::new(3, 4).unwrap()]);
/// let v2 = Vector::new([Rational::new(2, 3).unwrap(), Rational::new(1, 2).unwrap()]);
///
/// let sum = v1 + v2;
/// assert_eq!(sum, Vector::new([Rational::new(7, 6).unwrap(), Rational::new(5, 4).unwrap()]));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector<const N: usize> {
    coordinates: [Rational; N],
}

impl<const N: usize> Vector<N> {
    /// Создает новый вектор с заданными координатами
    pub fn new(coordinates: [Rational; N]) -> Self {
        Self { coordinates }
    }
    /// Поэлементное сложение векторов
    pub fn add(&self, other: &Self) -> Self {
        let mut result = [Rational::default(); N];
        for (i, (a, b)) in self
            .coordinates
            .iter()
            .zip(other.coordinates.iter())
            .enumerate()
        {
            result[i] = a + b;
        }
        Self {
            coordinates: result,
        }
    }
    /// Скалярное произведение векторов
    pub fn dot(&self, other: &Self) -> Rational {
        self.coordinates
            .iter()
            .zip(other.coordinates.iter())
            .map(|(a, b)| a * b)
            .sum::<Rational>()
    }
    /// Евклидова норма (длина) вектора
    pub fn norm(&self) -> f64 {
        let dot = self.dot(self);
        if dot.numerator() == 0 {
            return 0.0;
        }
        (dot.numerator() as f64 / dot.denominator() as f64).sqrt()
    }
    /// Умножение вектора на скаляр
    pub fn scale(&self, scalar: Rational) -> Self {
        Self {
            coordinates: self.coordinates.map(|x| x * scalar),
        }
    }
}
use std::ops::Add;

impl<const N: usize> Add for Vector<N> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut coordinates = [Rational::default(); N];
        for (i, (a, b)) in self
            .coordinates
            .iter()
            .zip(other.coordinates.iter())
            .enumerate()
        {
            coordinates[i] = a + b;
        }
        Self { coordinates }
    }
}
use std::ops::Sub;

impl<const N: usize> Sub for Vector<N> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut coordinates = [Rational::default(); N];
        for (i, (a, b)) in self
            .coordinates
            .iter()
            .zip(other.coordinates.iter())
            .enumerate()
        {
            coordinates[i] = a - b;
        }
        Self { coordinates }
    }
}
use std::ops::Mul;

impl<const N: usize> Mul<Rational> for Vector<N> {
    type Output = Self;

    fn mul(self, scalar: Rational) -> Self {
        self.scale(scalar)
    }
}
impl<const N: usize> Mul<Vector<N>> for Vector<N> {
    type Output = Rational;

    fn mul(self, other: Vector<N>) -> Rational {
        self.dot(&other)
    }
}
use std::ops::Div;

impl<const N: usize> Div<Rational> for Vector<N> {
    type Output = Self;

    fn div(self, scalar: Rational) -> Self {
        assert_ne!(scalar.numerator(), 0, "Division by zero");
        Self {
            coordinates: self.coordinates.map(|x| x / scalar),
        }
    }
}
use std::ops::Neg;

impl<const N: usize> Neg for Vector<N> {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            coordinates: self.coordinates.map(|x| -x),
        }
    }
}
use std::fmt;

impl<const N: usize> fmt::Display for Vector<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, coord) in self.coordinates.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", coord)?;
        }
        write!(f, "]")
    }
}
impl<const N: usize> From<[i64; N]> for Vector<N> {
    fn from(arr: [i64; N]) -> Self {
        Self {
            coordinates: arr.map(Rational::from),
        }
    }
}
impl<const N: usize> From<[f64; N]> for Vector<N> {
    fn from(arr: [f64; N]) -> Self {
        Self {
            coordinates: arr.map(Rational::from),
        }
    }
}
impl<const N: usize> From<[Rational; N]> for Vector<N> {
    fn from(arr: [Rational; N]) -> Self {
        Self { coordinates: arr }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_new() {
        let v = Vector::new([Rational::new(1, 2).unwrap(), Rational::new(3, 4).unwrap()]);
        assert_eq!(v.coordinates[0], Rational::new(1, 2).unwrap());
        assert_eq!(v.coordinates[1], Rational::new(3, 4).unwrap());
    }

    #[test]
    fn test_vector_add() {
        let v1 = Vector::new([Rational::new(1, 2).unwrap(), Rational::new(1, 3).unwrap()]);
        let v2 = Vector::new([Rational::new(1, 4).unwrap(), Rational::new(2, 3).unwrap()]);
        let sum = v1 + v2;
        assert_eq!(sum.coordinates[0], Rational::new(3, 4).unwrap());
        assert_eq!(sum.coordinates[1], Rational::new(1, 1).unwrap());
    }

    #[test]
    fn test_vector_dot() {
        let v1 = Vector::new([Rational::new(1, 2).unwrap(), Rational::new(1, 3).unwrap()]);
        let v2 = Vector::new([Rational::new(2, 1).unwrap(), Rational::new(3, 1).unwrap()]);
        let dot = v1 * v2;
        assert_eq!(dot, Rational::new(2, 1).unwrap());
    }

    #[test]
    fn test_vector_norm() {
        let v = Vector::new([Rational::new(3, 1).unwrap(), Rational::new(4, 1).unwrap()]);
        assert!((v.norm() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vector_scale() {
        let v = Vector::new([Rational::new(1, 2).unwrap(), Rational::new(1, 3).unwrap()]);
        let scaled = v.scale(Rational::new(2, 1).unwrap());
        assert_eq!(scaled.coordinates[0], Rational::new(1, 1).unwrap());
        assert_eq!(scaled.coordinates[1], Rational::new(2, 3).unwrap());
    }

    #[test]
    fn test_add_trait() {
        let v1 = Vector::new([Rational::new(1, 2).unwrap(), Rational::new(1, 3).unwrap()]);
        let v2 = Vector::new([Rational::new(1, 4).unwrap(), Rational::new(2, 3).unwrap()]);
        let sum = v1 + v2;
        assert_eq!(sum.coordinates[0], Rational::new(3, 4).unwrap());
        assert_eq!(sum.coordinates[1], Rational::new(1, 1).unwrap());
    }

    #[test]
    fn test_sub_trait() {
        let v1 = Vector::new([Rational::new(1, 2).unwrap(), Rational::new(1, 1).unwrap()]);
        let v2 = Vector::new([Rational::new(1, 4).unwrap(), Rational::new(1, 2).unwrap()]);
        let diff = v1 - v2;
        assert_eq!(diff.coordinates[0], Rational::new(1, 4).unwrap());
        assert_eq!(diff.coordinates[1], Rational::new(1, 2).unwrap());
    }

    #[test]
    fn test_mul_trait() {
        let v = Vector::new([Rational::new(1, 2).unwrap(), Rational::new(1, 3).unwrap()]);
        let scaled = v * Rational::new(2, 1).unwrap();
        assert_eq!(scaled.coordinates[0], Rational::new(1, 1).unwrap());
        assert_eq!(scaled.coordinates[1], Rational::new(2, 3).unwrap());
    }

    #[test]
    fn test_div_trait() {
        let v = Vector::new([Rational::new(1, 2).unwrap(), Rational::new(1, 3).unwrap()]);
        let divided = v / Rational::new(2, 1).unwrap();
        assert_eq!(divided.coordinates[0], Rational::new(1, 4).unwrap());
        assert_eq!(divided.coordinates[1], Rational::new(1, 6).unwrap());
    }

    #[test]
    #[should_panic(expected = "Division by zero")]
    fn test_div_by_zero() {
        let v = Vector::new([Rational::new(1, 2).unwrap(), Rational::new(1, 3).unwrap()]);
        let _ = v / Rational::new(0, 1).unwrap();
    }

    #[test]
    fn test_neg_trait() {
        let v = Vector::new([Rational::new(1, 2).unwrap(), Rational::new(-1, 3).unwrap()]);
        let neg = -v;
        assert_eq!(neg.coordinates[0], Rational::new(-1, 2).unwrap());
        assert_eq!(neg.coordinates[1], Rational::new(1, 3).unwrap());
    }
    #[test]
    fn test_display() {
        let v = Vector::new([Rational::new(1, 2).unwrap(), Rational::new(3, 4).unwrap()]);
        assert_eq!(format!("{}", v), "[1/2, 3/4]");
    }

    #[test]
    fn test_from_array() {
        let v: Vector<2> = [1, 2].into();
        assert_eq!(v.coordinates[0], Rational::new(1, 1).unwrap());
        assert_eq!(v.coordinates[1], Rational::new(2, 1).unwrap());
    }
}
