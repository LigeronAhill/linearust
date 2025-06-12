use crate::Result;
use std::{
    cmp::Ordering,
    fmt::Display,
    iter::Sum,
    ops::{Add, Div, Mul, Neg, Sub},
};
/// Рациональное число, представленное дробью с числителем (i64) и знаменателем (u32).
///
/// Поддерживает основные арифметические операции, сравнение и преобразования.
/// Автоматически нормализуется при создании (сокращает дробь).
///
/// # Примеры
/// ```
/// use linearust::types::Rational;
///
/// let a = Rational::new(2, 4).unwrap(); // Автоматически сокращается до 1/2
/// let b = Rational::new(1, 3).unwrap();
/// assert_eq!(a + b, Rational::new(5, 6).unwrap());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub struct Rational {
    numerator: i64,
    denominator: u32,
}
impl Default for Rational {
    fn default() -> Self {
        Rational {
            numerator: 0,
            denominator: 1,
        }
    }
}

impl Rational {
    /// Создает новое рациональное число.
    ///
    /// # Аргументы
    /// * `numerator` - Числитель (знаковый)
    /// * `denominator` - Знаменатель (беззнаковый, должен быть ≠ 0)
    ///
    /// # Возвращает
    /// * `Ok(Rational)` - если создание успешно
    /// * `Err(Error::DivisionByZero)` - если знаменатель равен 0
    /// * `Err(Error::Overflow)` - при переполнении в процессе нормализации
    ///
    /// # Примеры
    /// ```
    /// use linearust::types::Rational;
    ///
    /// let r = Rational::new(3, 4).unwrap();
    /// assert_eq!(r.numerator(), 3);
    /// assert_eq!(r.denominator(), 4);
    /// ```
    pub fn new(numerator: i64, denominator: u32) -> Result<Rational> {
        if denominator == 0 {
            return Err(crate::Error::DivisionByZero);
        }
        let mut res = Rational {
            numerator,
            denominator,
        };
        res.normalize()?;
        Ok(res)
    }
    fn normalize(&mut self) -> Result<&mut Self> {
        let gcd = gcd(self.numerator, self.denominator)?;
        if gcd == 0 {
            return Err(crate::Error::DivisionByZero);
        }
        self.numerator /= gcd as i64;
        self.denominator /= gcd;
        Ok(self)
    }
    /// Возвращает числитель дроби
    pub fn numerator(&self) -> i64 {
        self.numerator
    }
    /// Возвращает знаменатель дроби
    pub fn denominator(&self) -> u32 {
        self.denominator
    }
    pub fn checked_add(&self, other: &Rational) -> Result<Rational> {
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
    pub fn checked_sub(&self, other: &Self) -> Result<Self> {
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

    pub fn checked_mul(&self, other: &Self) -> Result<Self> {
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

    pub fn checked_div(&self, other: &Self) -> Result<Self> {
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
    pub fn checked_neg(self) -> Result<Self> {
        Ok(Self {
            numerator: self.numerator.checked_neg().ok_or(crate::Error::Overflow)?,
            denominator: self.denominator,
        })
    }
    /// Возвращает абсолютное значение числа
    pub fn abs(self) -> Self {
        Self {
            numerator: self.numerator.abs(),
            denominator: self.denominator,
        }
    }
    /// Возвращает обратное число (1/self)
    ///
    /// # Ошибки
    /// Возвращает `Error::DivisionByZero` если числитель равен 0
    pub fn reciprocal(self) -> Result<Self> {
        if self.numerator == 0 {
            Err(crate::Error::DivisionByZero)
        } else {
            Ok(Self {
                numerator: self.denominator as i64 * self.numerator.signum(),
                denominator: self.numerator.unsigned_abs() as u32,
            })
        }
    }
    /// Проверяет, является ли число отрицательным
    pub fn is_negative(&self) -> bool {
        self.numerator < 0
    }
    /// Проверяет, является ли число целым
    pub fn is_integer(&self) -> bool {
        self.denominator == 1
    }
    /// Возводит рациональное число в целую степень
    ///
    /// # Аргументы
    /// * `exp` - показатель степени (может быть отрицательным)
    ///
    /// # Возвращает
    /// * результат возведения в степень
    /// * паникует при переполнении или делении на ноль
    ///
    /// # Примеры
    /// ```
    /// use linearust::types::Rational;
    ///
    /// let r = Rational::new(2, 3).unwrap();
    /// assert_eq!(r.pow(2), Rational::new(4, 9).unwrap());
    /// assert_eq!(r.pow(-1), Rational::new(3, 2).unwrap());
    /// ```
    pub fn pow(&self, exp: i32) -> Self {
        self.checked_pow(exp).unwrap()
    }

    /// Оптимизированное возведение в степень с проверкой переполнения
    ///
    /// Использует алгоритм быстрого возведения в степень
    pub fn checked_pow(&self, exp: i32) -> Result<Self> {
        if exp == 0 {
            return Rational::new(1, 1);
        }

        let mut result = Rational::new(1, 1)?;
        let mut base = if exp > 0 {
            self.to_owned()
        } else {
            self.reciprocal()?
        };
        let mut exponent = exp.unsigned_abs();

        while exponent > 0 {
            if exponent % 2 == 1 {
                result = result.checked_mul(&base)?;
            }
            base = base.checked_mul(&base)?;
            exponent /= 2;
        }

        Ok(result)
    }
}
/// Computes the greatest common divisor (GCD) of two numbers.
///
/// # Arguments
/// * `a` - A signed 64-bit integer
/// * `b` - An unsigned 32-bit integer
///
/// # Returns
/// * `Ok(u32)` - The GCD of the absolute value of `a` and `b`
/// * `Err(crate::Error::Underflow)` - If `a` is `i64::MIN` (can't get absolute value)
/// * `Err(crate::Error::Overflow)` - If the result exceeds `u32::MAX`
///
fn gcd(a: i64, b: u32) -> Result<u32> {
    // Special case: i64::MIN.abs() would overflow
    if a == i64::MIN {
        return Err(crate::Error::Underflow);
    }

    let mut x = a.unsigned_abs(); // Safe after i64::MIN check
    let mut y = u64::from(b); // Safe u32 to u64 conversion

    // Euclidean algorithm
    while y != 0 {
        (x, y) = (y, x % y);
    }

    // Convert result back to u32
    u32::try_from(x).map_err(|_| crate::Error::Overflow)
}
impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Rational {
    fn cmp(&self, other: &Self) -> Ordering {
        let left = (self.numerator as i128) * (other.denominator as i128);
        let right = (other.numerator as i128) * (self.denominator as i128);
        left.cmp(&right)
    }
}
impl std::str::FromStr for Rational {
    type Err = crate::Error;

    fn from_str(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split('/').collect();
        match parts.as_slice() {
            [num, den] => {
                let numerator = num.parse().map_err(|_| crate::Error::ParseError)?;
                let denominator = den.parse().map_err(|_| crate::Error::ParseError)?;
                Self::new(numerator, denominator)
            }
            [num] => {
                let numerator = num.parse().map_err(|_| crate::Error::ParseError)?;
                Self::new(numerator, 1)
            }
            _ => Err(crate::Error::ParseError),
        }
    }
}
impl Neg for Rational {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Rational {
            numerator: -self.numerator,
            denominator: self.denominator,
        }
    }
}
impl<'a> Neg for &'a Rational {
    type Output = Rational;
    fn neg(self) -> Rational {
        Rational {
            numerator: -self.numerator,
            denominator: self.denominator,
        }
    }
}
impl Add for Rational {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let numerator = self.numerator * (other.denominator as i64)
            + other.numerator * (self.denominator as i64);
        let denominator = self.denominator * other.denominator;
        Rational::new(numerator, denominator).unwrap()
    }
}
impl<'a> Add for &'a Rational {
    type Output = Rational;

    fn add(self, other: Self) -> Rational {
        let numerator = self.numerator * (other.denominator as i64)
            + other.numerator * (self.denominator as i64);
        let denominator = self.denominator * other.denominator;
        Rational::new(numerator, denominator).unwrap()
    }
}
impl Sub for Rational {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let numerator = self.numerator * (other.denominator as i64)
            - other.numerator * (self.denominator as i64);

        let denominator = self.denominator * other.denominator;
        Rational::new(numerator, denominator).unwrap()
    }
}
impl<'a> Sub for &'a Rational {
    type Output = Rational;

    fn sub(self, other: Self) -> Rational {
        let numerator = self.numerator * (other.denominator as i64)
            - other.numerator * (self.denominator as i64);

        let denominator = self.denominator * other.denominator;
        Rational::new(numerator, denominator).unwrap()
    }
}

impl Mul<Rational> for Rational {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let numerator = self.numerator * other.numerator;

        let denominator = self.denominator * other.denominator;

        Rational::new(numerator, denominator).unwrap()
    }
}
impl<'a> Mul<&'a Rational> for &'a Rational {
    type Output = Rational;

    fn mul(self, other: Self) -> Rational {
        let numerator = self.numerator * other.numerator;

        let denominator = self.denominator * other.denominator;

        Rational::new(numerator, denominator).unwrap()
    }
}
impl Mul<i64> for Rational {
    type Output = Self;

    fn mul(self, other: i64) -> Self {
        let numerator = self.numerator * other;

        let denominator = self.denominator;

        Rational::new(numerator, denominator).unwrap()
    }
}
impl Mul<f64> for Rational {
    type Output = Self;

    fn mul(self, other: f64) -> Self {
        let o = Rational::from(other);
        self * o
    }
}

impl Div<Rational> for Rational {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let numerator = self.numerator * (other.denominator as i64);

        let denominator = self.denominator * (other.numerator.unsigned_abs() as u32);

        // Сохраняем знак в числителе
        let numerator = if other.numerator < 0 {
            -numerator
        } else {
            numerator
        };

        Rational::new(numerator, denominator).unwrap()
    }
}
impl<'a> Div<&'a Rational> for &'a Rational {
    type Output = Rational;

    fn div(self, other: Self) -> Rational {
        let numerator = self.numerator * (other.denominator as i64);

        let denominator = self.denominator * (other.numerator.unsigned_abs() as u32);

        // Сохраняем знак в числителе
        let numerator = if other.numerator < 0 {
            -numerator
        } else {
            numerator
        };

        Rational::new(numerator, denominator).unwrap()
    }
}
impl Div<i64> for Rational {
    type Output = Self;

    fn div(self, other: i64) -> Self {
        let numerator = self.numerator;

        let denominator = self.denominator * other.unsigned_abs() as u32;

        // Сохраняем знак в числителе
        let numerator = if other < 0 { -numerator } else { numerator };

        Rational::new(numerator, denominator).unwrap()
    }
}
impl Div<f64> for Rational {
    type Output = Self;

    fn div(self, other: f64) -> Self {
        self / Rational::from(other)
    }
}
impl From<Rational> for f64 {
    fn from(value: Rational) -> f64 {
        value.numerator as f64 / value.denominator as f64
    }
}
impl TryFrom<Rational> for f32 {
    type Error = crate::Error;

    fn try_from(value: Rational) -> crate::Result<Self> {
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
        // Handle special cases
        if value.is_nan() || value.is_infinite() {
            return Rational::default();
        }

        // Fast path for integers
        if value.fract() == 0.0 {
            if let Ok(int_val) = i64::try_from(value as i128) {
                return Rational::new(int_val, 1).unwrap_or_default();
            }
            return Rational::new(value.signum() as i64, 1).unwrap();
        }

        const MAX_DENOMINATOR: u32 = 1_000_000;
        const EPSILON: f64 = 1.0e-10;

        let sign = value.signum();
        let x = value.abs();

        // Stern-Brocot tree search
        let (mut a, mut b) = (0u32, 1u32); // a/b (left bound)
        let (mut c, mut d) = (1u32, 0u32); // c/d (right bound)
        let (mut p, mut q) = (1u32, 1u32); // p/q (current approximation)

        while q <= MAX_DENOMINATOR {
            let current = p as f64 / q as f64;
            if (current - x).abs() < EPSILON {
                break;
            }

            if x > current {
                // Move right in the tree
                a = p;
                b = q;
                p = p.checked_add(c).unwrap_or(MAX_DENOMINATOR);
                q = q.checked_add(d).unwrap_or(MAX_DENOMINATOR);
            } else {
                // Move left in the tree
                c = p;
                d = q;
                p = p.checked_add(a).unwrap_or(MAX_DENOMINATOR);
                q = q.checked_add(b).unwrap_or(MAX_DENOMINATOR);
            }
        }

        // Ensure we don't exceed max denominator
        if q > MAX_DENOMINATOR {
            p = (x * MAX_DENOMINATOR as f64).round() as u32;
            q = MAX_DENOMINATOR;
        }

        // Apply sign and create Rational
        let numerator = (sign * p as f64).round() as i64;
        Rational::new(numerator, q).unwrap_or_else(|_| {
            // Fallback to simpler approximation if needed
            let simple = (value * 1000.0).round() / 1000.0;
            Rational::new((sign * simple).round() as i64, 1000).unwrap_or_default()
        })
    }
}
impl From<i64> for Rational {
    fn from(value: i64) -> Self {
        Rational::new(value, 1).unwrap()
    }
}
impl Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.numerator, self.denominator)
    }
}
// Реализация Sum для поддержки .sum() в итераторах
impl Sum for Rational {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |acc, x| acc + x)
    }
}

// Для суммирования ссылок (&Rational)
impl<'a> Sum<&'a Rational> for Rational {
    fn sum<I: Iterator<Item = &'a Rational>>(iter: I) -> Self {
        iter.fold(Self::default(), |acc, x| acc + *x)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use std::str::FromStr;

    // Тесты для gcd()
    #[test]
    fn test_gcd_basic() {
        assert_eq!(gcd(48, 18).unwrap(), 6);
        assert_eq!(gcd(18, 48).unwrap(), 6);
        assert_eq!(gcd(13, 13).unwrap(), 13);
        assert_eq!(gcd(0, 5).unwrap(), 5);
        assert_eq!(gcd(5, 0).unwrap(), 5);
        assert_eq!(gcd(0, 0).unwrap(), 0);
    }

    #[test]
    fn test_gcd_negative() {
        assert_eq!(gcd(-48, 18).unwrap(), 6);
    }

    #[test]
    fn test_gcd_edge_cases() {
        assert_eq!(gcd(i64::MAX, 1).unwrap(), 1);
        assert_eq!(gcd(1, u32::MAX).unwrap(), 1);
        assert!(matches!(gcd(i64::MIN, 1), Err(Error::Underflow)));
    }

    // Тесты для Rational::new()
    #[test]
    fn test_rational_creation() {
        assert!(Rational::new(1, 0).is_err()); // division by zero
        assert_eq!(Rational::new(2, 4).unwrap(), Rational::new(1, 2).unwrap());
        assert_eq!(Rational::new(-2, 4).unwrap(), Rational::new(-1, 2).unwrap());
    }

    // Тесты для арифметических операций
    #[test]
    fn test_rational_add() {
        let r1 = Rational::new(1, 2).unwrap();
        let r2 = Rational::new(1, 3).unwrap();
        assert_eq!(r1 + r2, Rational::new(5, 6).unwrap());
    }

    #[test]
    fn test_rational_sub() {
        let r1 = Rational::new(1, 2).unwrap();
        let r2 = Rational::new(1, 3).unwrap();
        assert_eq!(r1 - r2, Rational::new(1, 6).unwrap());
    }

    #[test]
    fn test_rational_mul() {
        let r1 = Rational::new(1, 2).unwrap();
        let r2 = Rational::new(2, 3).unwrap();
        assert_eq!(r1 * r2, Rational::new(1, 3).unwrap());
    }

    #[test]
    fn test_rational_div() {
        let r1 = Rational::new(1, 2).unwrap();
        let r2 = Rational::new(2, 3).unwrap();
        assert_eq!(r1 / r2, Rational::new(3, 4).unwrap());
    }

    // Тесты для преобразований
    #[test]
    fn test_from_f64() {
        assert_eq!(Rational::from(0.5), Rational::new(1, 2).unwrap());
        assert_eq!(Rational::from(-0.25), Rational::new(-1, 4).unwrap());
        assert_eq!(Rational::from(2.0), Rational::new(2, 1).unwrap());
    }

    #[test]
    fn test_into_f64() {
        let r = Rational::new(1, 2).unwrap();
        assert_eq!(f64::from(r), 0.5);
    }

    // Тесты для сравнений
    #[test]
    fn test_rational_ordering() {
        let r1 = Rational::new(1, 2).unwrap();
        let r2 = Rational::new(1, 3).unwrap();
        assert!(r1 > r2);
        assert!(r2 < r1);
        assert_eq!(r1, r1);
    }

    // Тесты для FromStr
    #[test]
    fn test_from_str() {
        assert_eq!(
            Rational::from_str("1/2").unwrap(),
            Rational::new(1, 2).unwrap()
        );
        assert_eq!(
            Rational::from_str("3").unwrap(),
            Rational::new(3, 1).unwrap()
        );
        assert!(Rational::from_str("a/b").is_err());
        assert!(Rational::from_str("1/2/3").is_err());
    }

    // Тесты для специальных методов
    #[test]
    fn test_reciprocal() {
        assert_eq!(
            Rational::new(2, 3).unwrap().reciprocal().unwrap(),
            Rational::new(3, 2).unwrap()
        );
        assert!(Rational::new(0, 1).unwrap().reciprocal().is_err());
    }

    #[test]
    fn test_abs() {
        assert_eq!(
            Rational::new(-1, 2).unwrap().abs(),
            Rational::new(1, 2).unwrap()
        );
    }

    // Тесты для переполнений
    #[test]
    fn test_overflow_cases() {
        let max = Rational::new(i64::MAX, 1).unwrap();
        assert!(max.checked_add(&max).is_err());
        assert!(max.checked_mul(&max).is_err());
    }

    // Тесты для операций с разными типами
    #[test]
    fn test_mixed_operations() {
        let r = Rational::new(1, 2).unwrap();
        assert_eq!(r * 3, Rational::new(3, 2).unwrap());
        assert_eq!(r / 2, Rational::new(1, 4).unwrap());
    }
    #[test]
    fn test_pow_positive() {
        let r = Rational::new(2, 3).unwrap();
        assert_eq!(r.pow(3), Rational::new(8, 27).unwrap());
        assert_eq!(r.pow(1), r);
        assert_eq!(r.pow(0), Rational::new(1, 1).unwrap());
    }

    #[test]
    fn test_pow_negative() {
        let r = Rational::new(2, 3).unwrap();
        assert_eq!(r.pow(-1), Rational::new(3, 2).unwrap());
        assert_eq!(r.pow(-2), Rational::new(9, 4).unwrap());
    }

    #[test]
    fn test_pow_zero() {
        let r = Rational::new(2, 3).unwrap();
        assert_eq!(r.pow(0), Rational::new(1, 1).unwrap());
    }

    #[test]
    fn test_checked_pow_overflow() {
        let large = Rational::new(i64::MAX, 1).unwrap();
        assert!(large.checked_pow(2).is_err());
    }

    #[test]
    fn test_checked_pow_division_by_zero() {
        let zero = Rational::new(0, 1).unwrap();
        assert!(zero.checked_pow(-1).is_err());
    }

    #[test]
    fn test_pow_edge_cases() {
        let r = Rational::new(1, 1).unwrap();
        assert_eq!(r.pow(i32::MAX), r);
        assert_eq!(r.pow(i32::MIN), r);
    }

    #[test]
    fn test_pow_normalization() {
        let r = Rational::new(2, 4).unwrap(); // 1/2 после нормализации
        assert_eq!(r.pow(2), Rational::new(1, 4).unwrap());
    }
}
