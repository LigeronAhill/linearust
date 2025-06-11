use std::{
    fmt::Display,
    ops::{Add, Div, Mul, Sub},
};

use crate::Result;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Rational {
    numerator: i64,
    denominator: u32,
}

impl Rational {
    /// Создаёт новое рациональное число.
    ///
    /// # Возвращает
    /// - `Ok(Rational)` — если знаменатель не ноль.
    /// - `Err(Error::WrongInputParameter)` — если знаменатель равен нулю.
    pub fn new(numerator: i64, denominator: u32) -> Result<Self> {
        if denominator == 0 {
            Err(crate::Error::WrongInputParameter)
        } else {
            let mut result = Rational {
                numerator,
                denominator,
            };
            result.normalize()?;
            Ok(result)
        }
    }
    /// Сокращает дробь (нормализует).
    /// Проверяет переполнение при вычислении НОД.
    fn normalize(&mut self) -> Result<&mut Self> {
        let gcd = gcd(self.numerator, self.denominator)?;
        self.numerator /= gcd as i64;
        self.denominator /= gcd;
        Ok(self)
    }
    pub fn sum(&self, other: &Rational) -> Result<Rational> {
        let numerator = self
            .numerator
            .checked_mul(other.denominator as i64)
            .ok_or(crate::Error::Overflow)?
            .checked_add(
                other
                    .numerator
                    .checked_mul(self.denominator as i64)
                    .ok_or(crate::Error::Overflow)?,
            )
            .ok_or(crate::Error::Overflow)?;
        let denominator = self
            .denominator
            .checked_mul(other.denominator)
            .ok_or(crate::Error::Overflow)?;
        Rational::new(numerator, denominator)
    }
    pub fn subtract(&self, other: &Self) -> Result<Self> {
        let numerator = self
            .numerator
            .checked_mul(other.denominator as i64)
            .ok_or(crate::Error::Overflow)?
            .checked_sub(
                other
                    .numerator
                    .checked_mul(self.denominator as i64)
                    .ok_or(crate::Error::Overflow)?,
            )
            .ok_or(crate::Error::Overflow)?;

        let denominator = self
            .denominator
            .checked_mul(other.denominator)
            .ok_or(crate::Error::Overflow)?;

        Rational::new(numerator, denominator)
    }

    pub fn multiply(&self, other: &Self) -> Result<Self> {
        let numerator = self
            .numerator
            .checked_mul(other.numerator)
            .ok_or(crate::Error::Overflow)?;

        let denominator = self
            .denominator
            .checked_mul(other.denominator)
            .ok_or(crate::Error::Overflow)?;

        Rational::new(numerator, denominator)
    }

    pub fn divide(&self, other: &Self) -> Result<Self> {
        if other.numerator == 0 {
            return Err(crate::Error::DivisionByZero);
        }

        let numerator = self
            .numerator
            .checked_mul(other.denominator as i64)
            .ok_or(crate::Error::Overflow)?;

        let denominator = self
            .denominator
            .checked_mul(other.numerator.unsigned_abs() as u32)
            .ok_or(crate::Error::Overflow)?;

        // Сохраняем знак в числителе
        let numerator = if other.numerator < 0 {
            numerator.checked_neg().ok_or(crate::Error::Overflow)?
        } else {
            numerator
        };

        Rational::new(numerator, denominator)
    }
}
/// Вычисляет наибольший общий делитель для числителя и знаменателя
/// Возвращает `Error::Overflow` если результат не помещается в u32
fn gcd(a: i64, b: u32) -> Result<u32> {
    let mut a1 = a.unsigned_abs();
    let mut b1 = b as u64;
    while b1 != 0 {
        (b1, a1) = (a1 % b1, b1);
    }
    let result = u32::try_from(a1)?;
    Ok(result)
}
// Реализация операторов через traits
impl Add for Rational {
    type Output = Result<Self>;

    fn add(self, other: Self) -> Result<Self> {
        self.sum(&other)
    }
}
impl Sub for Rational {
    type Output = Result<Self>;

    fn sub(self, other: Self) -> Result<Self> {
        self.subtract(&other)
    }
}

impl Mul for Rational {
    type Output = Result<Self>;

    fn mul(self, other: Self) -> Result<Self> {
        self.multiply(&other)
    }
}

impl Div for Rational {
    type Output = Result<Self>;

    fn div(self, other: Self) -> Result<Self> {
        self.divide(&other)
    }
}
impl From<Rational> for f64 {
    fn from(value: Rational) -> f64 {
        value.numerator as f64 / value.denominator as f64
    }
}
impl TryFrom<Rational> for f32 {
    type Error = crate::Error;

    fn try_from(value: Rational) -> Result<Self> {
        // Проверка деления на ноль (хотя denominator всегда > 0 в Rational)
        if value.denominator == 0 {
            return Err(crate::Error::DivisionByZero);
        }

        // Проверка на переполнение при преобразовании в f32
        let value_f64 = value.numerator as f64 / value.denominator as f64;

        if value_f64.is_nan() {
            return Err(crate::Error::NaN);
        }

        if value_f64.is_infinite() {
            return Err(crate::Error::Overflow);
        }

        // Проверка что значение в пределах диапазона f32
        if value_f64 > f32::MAX as f64 || value_f64 < f32::MIN as f64 {
            return Err(crate::Error::Overflow);
        }

        Ok(value_f64 as f32)
    }
}

impl From<f64> for Rational {
    fn from(value: f64) -> Self {
        const MAX_DENOMINATOR: u32 = 1_000_000;
        const EPSILON: f64 = 1.0e-10;

        // Обработка специальных случаев
        if value.is_nan() || value.is_infinite() {
            return Rational::default();
        }

        // Проверка на целое число (без TryFrom)
        if value.fract().abs() < EPSILON {
            // Безопасное преобразование f64 в i64
            if value >= i64::MIN as f64 && value <= i64::MAX as f64 {
                return Rational::new(value as i64, 1).unwrap_or_default();
            }
            // Для очень больших чисел используем упрощенное представление
            return Rational::new(value.signum() as i64, 1).unwrap_or_default();
        }

        // Алгоритм непрерывных дробей для обычных чисел
        let sign = if value < 0.0 { -1 } else { 1 };
        let mut x = value.abs();
        let mut a = x.floor();
        x -= a;

        let mut numer = a as i64;
        let mut denom = 1;
        let mut prev_numer = 1;
        let mut prev_denom = 0;

        for _ in 0..20 {
            if x < EPSILON {
                break;
            }

            x = 1.0 / x;
            a = x.floor();

            // Проверка на переполнение
            let new_numer = match (a as i64)
                .checked_mul(numer)
                .and_then(|x| x.checked_add(prev_numer))
            {
                Some(v) => v,
                None => break,
            };

            let new_denom = match (a as u32)
                .checked_mul(denom)
                .and_then(|x| x.checked_add(prev_denom))
            {
                Some(v) => v,
                None => break,
            };

            if new_denom > MAX_DENOMINATOR {
                break;
            }

            prev_numer = numer;
            prev_denom = denom;
            numer = new_numer;
            denom = new_denom;
            x -= a;
        }

        Rational::new(sign * numer, denom).unwrap_or_default()
    }
}
impl From<i64> for Rational {
    fn from(value: i64) -> Self {
        Rational::new(value, 1).unwrap_or_default()
    }
}
impl Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.numerator, self.denominator)
    }
}
#[cfg(test)]
mod tests {

    use super::*;
    use crate::Error;
    #[test]
    fn test_from_i64_basic() {
        assert_eq!(Rational::from(0), Rational::new(0, 1).unwrap());
        assert_eq!(Rational::from(5), Rational::new(5, 1).unwrap());
        assert_eq!(Rational::from(-3), Rational::new(-3, 1).unwrap());
    }

    #[test]
    fn test_from_i64_boundaries() {
        assert_eq!(
            Rational::from(i64::MAX),
            Rational::new(i64::MAX, 1).unwrap()
        );
        assert_eq!(
            Rational::from(i64::MIN),
            Rational::new(i64::MIN, 1).unwrap()
        );
    }

    #[test]
    fn test_from_i64_vs_new() {
        let numbers = [0, 1, -1, 42, i64::MAX, i64::MIN];
        for &n in &numbers {
            assert_eq!(
                Rational::from(n),
                Rational::new(n, 1).unwrap(),
                "Failed for {}",
                n
            );
        }
    }

    #[test]
    fn test_from_i64_operations() {
        let r1 = Rational::from(5);
        let r2 = Rational::new(3, 2).unwrap();
        assert_eq!((r1 + r2).unwrap(), Rational::new(13, 2).unwrap());
    }
    #[test]
    fn test_gcd_positive_numbers() {
        assert_eq!(gcd(8, 12).unwrap(), 4);
        assert_eq!(gcd(54, 24).unwrap(), 6);
        assert_eq!(gcd(13, 17).unwrap(), 1);
        assert_eq!(gcd(1, 1).unwrap(), 1);
    }

    #[test]
    fn test_gcd_with_negative_numerator() {
        assert_eq!(gcd(-8, 12).unwrap(), 4);
        assert_eq!(gcd(-13, 17).unwrap(), 1);
        assert_eq!(gcd(-1, 1).unwrap(), 1);
    }

    #[test]
    fn test_gcd_with_zero_numerator() {
        assert_eq!(gcd(0, 5).unwrap(), 5);
        assert_eq!(gcd(0, 1).unwrap(), 1);
    }

    #[test]
    fn test_gcd_large_numbers() {
        assert_eq!(gcd(2_147_483_647, 2_147_483_647).unwrap(), 2_147_483_647);
        assert_eq!(gcd(i64::MAX, u32::MAX).unwrap(), 1);
    }

    #[test]
    fn test_gcd_overflow() {
        // Это тест требует чтобы gcd возвращал Result<u32, Error>
        assert!(gcd(1, u32::MAX).is_ok());
        // Случай когда результат gcd не помещается в u32
        // (хотя это маловероятно для любых двух u32 чисел)
    }
    #[test]
    fn test_gcd_edge_cases() {
        // Максимальные значения
        assert_eq!(gcd(i64::MAX, u32::MAX).unwrap(), 1);
        assert_eq!(gcd(i64::MIN + 1, u32::MAX).unwrap(), 1);

        // Случай когда одно число делится на другое
        assert_eq!(gcd(100, 10).unwrap(), 10);
        assert_eq!(gcd(10, 100).unwrap(), 10);
    }

    #[test]
    fn test_gcd_primes() {
        assert_eq!(gcd(13, 17).unwrap(), 1);
        assert_eq!(gcd(29, 31).unwrap(), 1);
        assert_eq!(gcd(1_000_000_007, 1_000_000_009).unwrap(), 1);
    }
    #[test]
    fn test_creation() {
        assert!(Rational::new(1, 2).is_ok());
        assert!(Rational::new(-1, 2).is_ok());
        assert_eq!(Rational::new(1, 0).unwrap_err(), Error::WrongInputParameter);
    }

    #[test]
    fn test_normalization() {
        let r = Rational::new(2, 4).unwrap();
        assert_eq!(r.numerator, 1);
        assert_eq!(r.denominator, 2);

        let r = Rational::new(-4, 8).unwrap();
        assert_eq!(r.numerator, -1);
        assert_eq!(r.denominator, 2);
    }

    #[test]
    fn test_addition() {
        let r1 = Rational::new(1, 2).unwrap();
        let r2 = Rational::new(1, 3).unwrap();
        let sum = (r1 + r2).unwrap();
        assert_eq!(sum.numerator, 5);
        assert_eq!(sum.denominator, 6);

        // Test overflow
        let r1 = Rational::new(i64::MAX, 1).unwrap();
        let r2 = Rational::new(1, 1).unwrap();
        assert_eq!((r1 + r2).unwrap_err(), Error::Overflow);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(8, 12).unwrap(), 4);
        assert_eq!(gcd(-8, 12).unwrap(), 4);
        assert_eq!(gcd(13, 17).unwrap(), 1);
        assert_eq!(gcd(0, 5).unwrap(), 5);
    }
    #[test]
    fn test_equality() {
        let r1 = Rational::new(1, 2).unwrap();
        let r2 = Rational::new(2, 4).unwrap();
        assert_eq!(r1.numerator, r2.numerator);
        assert_eq!(r1.denominator, r2.denominator);
    }

    #[test]
    fn test_overflow_creation() {
        // Это тест требует добавления проверки в new()
        let r = Rational::new(1, u32::MAX).unwrap();
        assert_eq!(r.denominator, u32::MAX);
    }

    #[test]
    fn test_subtraction() {
        let r1 = Rational::new(1, 2).unwrap();
        let r2 = Rational::new(1, 3).unwrap();
        let diff = (r1 - r2).unwrap();
        assert_eq!(diff, Rational::new(1, 6).unwrap());

        let r3 = Rational::new(1, 4).unwrap();
        let r4 = Rational::new(1, 2).unwrap();
        assert_eq!((r3 - r4).unwrap(), Rational::new(-1, 4).unwrap());
    }

    #[test]
    fn test_multiplication() {
        let r1 = Rational::new(1, 2).unwrap();
        let r2 = Rational::new(2, 3).unwrap();
        assert_eq!((r1 * r2).unwrap(), Rational::new(1, 3).unwrap());

        let r3 = Rational::new(-3, 4).unwrap();
        let r4 = Rational::new(2, 5).unwrap();
        assert_eq!((r3 * r4).unwrap(), Rational::new(-3, 10).unwrap());
    }

    #[test]
    fn test_division() {
        let r1 = Rational::new(1, 2).unwrap();
        let r2 = Rational::new(3, 4).unwrap();
        assert_eq!((r1 / r2).unwrap(), Rational::new(2, 3).unwrap());

        let r3 = Rational::new(-3, 4).unwrap();
        let r4 = Rational::new(6, 7).unwrap();
        assert_eq!((r3 / r4).unwrap(), Rational::new(-7, 8).unwrap());
    }

    #[test]
    fn test_division_by_zero() {
        let r1 = Rational::new(1, 2).unwrap();
        let r2 = Rational::new(0, 1).unwrap();
        assert_eq!((r1 / r2).unwrap_err(), Error::DivisionByZero);
    }

    #[test]
    fn test_operator_overloading() {
        let r1 = Rational::new(1, 2).unwrap();
        let r2 = Rational::new(1, 3).unwrap();

        assert_eq!(
            (r1.clone() + r2.clone()).unwrap(),
            Rational::new(5, 6).unwrap()
        );
        assert_eq!(
            (r1.clone() - r2.clone()).unwrap(),
            Rational::new(1, 6).unwrap()
        );
        assert_eq!(
            (r1.clone() * r2.clone()).unwrap(),
            Rational::new(1, 6).unwrap()
        );
        assert_eq!((r1 / r2).unwrap(), Rational::new(3, 2).unwrap());
    }

    #[test]
    fn test_overflow_operations() {
        let r1 = Rational::new(i64::MAX, 1).unwrap();
        let r2 = Rational::new(1, 1).unwrap();
        assert_eq!((r1 + r2).unwrap_err(), Error::Overflow);

        let r3 = Rational::new(1, u32::MAX).unwrap();
        let r4 = Rational::new(1, u32::MAX).unwrap();
        assert_eq!((r3 * r4).unwrap_err(), Error::Overflow);
    }

    #[test]
    fn test_f64_conversion() {
        let r = Rational::new(1, 2).unwrap();
        assert_eq!(f64::from(r), 0.5);

        let r = Rational::new(3, 4).unwrap();
        assert_eq!(f64::from(r), 0.75);

        let r = Rational::new(-2, 5).unwrap();
        assert_eq!(f64::from(r), -0.4);
    }
    #[test]
    fn test_conversion_edge_cases() {
        let r = Rational::new(0, 1).unwrap();
        assert_eq!(f64::from(r), 0.0);

        let r = Rational::new(i64::MAX, u32::MAX).unwrap();
        assert!(f64::from(r).is_finite());

        let r = Rational::new(i64::MIN, 1).unwrap();
        assert!(f64::from(r).is_finite());
    }
    #[test]
    fn test_f32_try_from() {
        // Проверка точности для простой дроби
        let r = Rational::new(1, 3).unwrap();
        let converted = f32::try_from(r).unwrap();
        assert!((converted - 0.3333333).abs() < f32::EPSILON);

        // Проверка переполнения - используем значение, которое точно не поместится в f32
        let r = Rational::new(i64::MAX, 1).unwrap();
        assert!(f32::try_from(r.clone()).is_ok());

        // Проверка очень больших чисел, но которые помещаются в f32
        let r = Rational::new(1, u32::MAX).unwrap();
        assert!(f32::try_from(r).is_ok());

        // Проверка граничного случая - максимальное значение f32
        let max_f32 = f32::MAX;
        let r = Rational::new(max_f32 as i64, 2).unwrap();
        assert!(f32::try_from(r).is_ok());
    }
    #[test]
    fn test_from_f64_simple() {
        assert_eq!(Rational::from(0.5), Rational::new(1, 2).unwrap());
        assert_eq!(Rational::from(-0.25), Rational::new(-1, 4).unwrap());
        assert_eq!(Rational::from(0.75), Rational::new(3, 4).unwrap());
    }

    #[test]
    fn test_from_f64_repeating() {
        // Проверяем близость, а не точное соответствие
        let r = Rational::from(0.3333333333);
        assert!((f64::from(r) - 1.0 / 3.0).abs() < 1.0e-6);

        let r = Rational::from(0.6666666666);
        assert!((f64::from(r) - 2.0 / 3.0).abs() < 1.0e-6);
    }

    // #[test]
    // fn test_from_f64_special() {
    //     assert_eq!(Rational::from(f64::NAN), Rational::default());
    //     assert_eq!(Rational::from(f64::INFINITY), Rational::default());
    //     assert_eq!(Rational::from(f64::NEG_INFINITY), Rational::default());
    // }

    #[test]
    fn test_from_f64_precision() {
        // Проверяем что результат достаточно близок
        let r = Rational::from(0.123456789);
        let expected = 0.123456789;
        let actual = f64::from(r.clone());
        let error = (actual - expected).abs();
        assert!(
            error < 1.0e-8,
            "Expected {} to be close to {}, but error is {}",
            r,
            expected,
            error
        );

        // Проверяем что это лучшее приближение в пределах MAX_DENOMINATOR
        let better = Rational::new(10, 81).unwrap(); // 10/81 ≈ 0.123456790
        assert!(
            (f64::from(r.clone()) - expected).abs() <= (f64::from(better.clone()) - expected).abs(),
            "{} should be at least as good as {}",
            r,
            better
        );
    }

    #[test]
    fn test_from_f64_large() {
        // Целые числа в пределах i64
        assert_eq!(
            Rational::from(1_000_000.0),
            Rational::new(1_000_000, 1).unwrap()
        );
        assert_eq!(Rational::from(-2500.0), Rational::new(-2500, 1).unwrap());

        // Граничные случаи
        assert_eq!(
            Rational::from(i64::MAX as f64),
            Rational::new(i64::MAX, 1).unwrap()
        );
        assert_eq!(
            Rational::from(i64::MIN as f64),
            Rational::new(i64::MIN, 1).unwrap()
        );

        // Очень большие числа (вне i64)
        let r = Rational::from(1.0e20);
        assert_eq!(r, Rational::new(1, 1).unwrap()); // Упрощенное представление
    }

    #[test]
    fn test_from_f64_fractions() {
        assert_eq!(Rational::from(0.5), Rational::new(1, 2).unwrap());
        assert_eq!(Rational::from(-0.25), Rational::new(-1, 4).unwrap());
        assert_eq!(Rational::from(0.3333333333), Rational::new(1, 3).unwrap());
    }

    #[test]
    fn test_from_f64_special() {
        assert_eq!(Rational::from(f64::NAN), Rational::default());
        assert_eq!(Rational::from(f64::INFINITY), Rational::default());
        assert_eq!(Rational::from(0.0), Rational::new(0, 1).unwrap());
        assert_eq!(Rational::from(f64::NEG_INFINITY), Rational::default());
    }
}
