use crate::Result;
use std::ops::{Add, Div, Mul, Sub};

use super::Rational;

#[derive(Debug, Clone, PartialEq)]
pub struct Vector(Vec<Rational>);

impl Vector {
    /// Создает новый вектор заданного размера, заполненный нулями
    pub fn new(size: u32) -> Self {
        Self(vec![Rational::default(); size as usize])
    }

    /// Возвращает размер вектора
    pub fn size(&self) -> u32 {
        self.0.len() as u32
    }

    /// Получает элемент по индексу
    pub fn get(&self, index: usize) -> Option<&Rational> {
        self.0.get(index)
    }

    /// Устанавливает элемент по индексу
    pub fn set(&mut self, index: usize, value: Rational) -> Result<()> {
        if index >= self.0.len() {
            return Err(crate::Error::WrongInputParameter);
        }
        self.0[index] = value;
        Ok(())
    }
    /// Вычисляет векторное произведение для 3-мерных векторов
    /// Возвращает ошибку если векторы не 3-мерные
    pub fn cross(&self, rhs: &Self) -> Result<Self> {
        if self.size() != 3 || rhs.size() != 3 {
            return Err(crate::Error::WrongInputParameter);
        }

        let a = &self.0;
        let b = &rhs.0;

        let x = (a[1].clone() * b[2].clone())? - (a[2].clone() * b[1].clone())?;
        let y = (a[2].clone() * b[0].clone())? - (a[0].clone() * b[2].clone())?;
        let z = (a[0].clone() * b[1].clone())? - (a[1].clone() * b[0].clone())?;

        Ok(Self(vec![x?, y?, z?]))
    }
}
impl From<&[Rational]> for Vector {
    fn from(value: &[Rational]) -> Self {
        Self(value.to_vec())
    }
}
impl From<&[i64]> for Vector {
    fn from(value: &[i64]) -> Self {
        Self(value.iter().map(|v| Rational::from(*v)).collect())
    }
}

// Реализация операций

impl Add for Vector {
    type Output = Result<Self>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.size() != rhs.size() {
            return Err(crate::Error::WrongInputParameter);
        }
        Ok(Self(
            self.0
                .into_iter()
                .zip(rhs.0.into_iter())
                .map(|(a, b)| a + b)
                .collect::<Result<Vec<_>>>()?,
        ))
    }
}

impl Sub for Vector {
    type Output = Result<Self>;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.size() != rhs.size() {
            return Err(crate::Error::WrongInputParameter);
        }
        Ok(Self(
            self.0
                .into_iter()
                .zip(rhs.0.into_iter())
                .map(|(a, b)| a - b)
                .collect::<Result<Vec<_>>>()?,
        ))
    }
}

impl Mul<Rational> for Vector {
    type Output = Result<Self>;

    fn mul(self, scalar: Rational) -> Result<Self> {
        Ok(Self(
            self.0
                .into_iter()
                .map(|x| x * scalar.clone())
                .collect::<Result<Vec<_>>>()?,
        ))
    }
}
impl Mul<i64> for Vector {
    type Output = Result<Self>;

    fn mul(self, scalar: i64) -> Result<Self> {
        Ok(Self(
            self.0
                .into_iter()
                .map(|x| x * Rational::from(scalar))
                .collect::<Result<Vec<_>>>()?,
        ))
    }
}

impl Div<Rational> for Vector {
    type Output = Result<Self>;

    fn div(self, scalar: Rational) -> Result<Self> {
        if scalar == Rational::default() {
            return Err(crate::Error::DivisionByZero);
        }
        Ok(Self(
            self.0
                .into_iter()
                .map(|x| x / scalar.clone())
                .collect::<Result<Vec<_>>>()?,
        ))
    }
}
impl Div<i64> for Vector {
    type Output = Result<Self>;

    fn div(self, scalar: i64) -> Result<Self> {
        if scalar == 0 {
            return Err(crate::Error::DivisionByZero);
        }
        Ok(Self(
            self.0
                .into_iter()
                .map(|x| x / Rational::from(scalar))
                .collect::<Result<Vec<_>>>()?,
        ))
    }
}

// Тесты

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_creation() {
        let v = Vector::new(3);
        assert_eq!(v.size(), 3);
        assert_eq!(v.get(0), Some(&Rational::default()));
    }

    #[test]
    fn test_vector_addition() {
        // let v1 = Vector::from([Rational::from(1), Rational::from(2), Rational::from(3)].as_slice());
        let v1 = Vector::from([1, 2, 3].as_slice());
        let v2 = Vector::from([Rational::from(4), Rational::from(5), Rational::from(6)].as_slice());
        let sum = (v1 + v2).unwrap();
        assert_eq!(
            sum,
            Vector::from([Rational::from(5), Rational::from(7), Rational::from(9)].as_slice())
        );
    }

    #[test]
    fn test_vector_subtraction() {
        let v1 = Vector::from([Rational::from(5), Rational::from(7), Rational::from(9)].as_slice());
        let v2 = Vector::from([Rational::from(1), Rational::from(2), Rational::from(3)].as_slice());
        let diff = (v1 - v2).unwrap();
        assert_eq!(
            diff,
            Vector::from([Rational::from(4), Rational::from(5), Rational::from(6)].as_slice())
        );
    }

    #[test]
    fn test_vector_scalar_multiplication() {
        let v = Vector::from([Rational::from(1), Rational::from(2), Rational::from(3)].as_slice());
        let scalar = Rational::from(2);
        let product = v * scalar;
        assert_eq!(
            product.unwrap(),
            Vector::from([Rational::from(2), Rational::from(4), Rational::from(6)].as_slice())
        );
        let v = Vector::from([Rational::from(1), Rational::from(2), Rational::from(3)].as_slice());
        let scalar = 2;
        let product = v * scalar;
        assert_eq!(
            product.unwrap(),
            Vector::from([Rational::from(2), Rational::from(4), Rational::from(6)].as_slice())
        );
    }

    #[test]
    fn test_vector_scalar_division() {
        let v = Vector::from([Rational::from(2), Rational::from(4), Rational::from(6)].as_slice());
        let scalar = Rational::from(2);
        let quotient = (v / scalar).unwrap();
        assert_eq!(
            quotient,
            Vector::from([Rational::from(1), Rational::from(2), Rational::from(3)].as_slice())
        );
        let v = Vector::from([Rational::from(2), Rational::from(4), Rational::from(6)].as_slice());
        let scalar = 2;
        let quotient = (v / scalar).unwrap();
        assert_eq!(
            quotient,
            Vector::from([Rational::from(1), Rational::from(2), Rational::from(3)].as_slice())
        );
    }

    #[test]
    fn test_vector_division_by_zero() {
        let v = Vector::from([Rational::from(1)].as_slice());
        let scalar = Rational::from(0);
        assert_eq!(v / scalar, Err(crate::Error::DivisionByZero));
    }

    #[test]
    fn test_vector_size_mismatch() {
        let v1 = Vector::new(2);
        let v2 = Vector::new(3);
        assert_eq!(
            v1.clone() + v2.clone(),
            Err(crate::Error::WrongInputParameter)
        );
        assert_eq!(v1 - v2, Err(crate::Error::WrongInputParameter));
    }
    #[test]
    fn test_cross_product() {
        let v1 = Vector::from([Rational::from(1), Rational::from(0), Rational::from(0)].as_slice());
        let v2 = Vector::from([Rational::from(0), Rational::from(1), Rational::from(0)].as_slice());
        let cross = v1.cross(&v2).unwrap();
        assert_eq!(
            cross,
            Vector::from([Rational::from(0), Rational::from(0), Rational::from(1)].as_slice())
        );
    }

    #[test]
    fn test_cross_product_non_orthogonal() {
        let v1 = Vector::from([Rational::from(1), Rational::from(2), Rational::from(3)].as_slice());
        let v2 = Vector::from([Rational::from(4), Rational::from(5), Rational::from(6)].as_slice());
        let cross = v1.cross(&v2).unwrap();
        assert_eq!(
            cross,
            Vector::from([Rational::from(-3), Rational::from(6), Rational::from(-3)].as_slice())
        );
    }

    #[test]
    fn test_cross_product_wrong_dimensions() {
        let v1 = Vector::new(2);
        let v2 = Vector::new(3);
        assert_eq!(v1.cross(&v2), Err(crate::Error::WrongInputParameter));
    }
}
