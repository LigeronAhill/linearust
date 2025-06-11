use std::fmt::Display;

#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    WrongInputParameter,
    Overflow,
    Underflow,
    DivisionByZero,
    Infinity,
    NaN,
    TryFromIntError(String),
}

pub type Result<T> = core::result::Result<T, Error>;
impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for Error {}

impl From<std::num::TryFromIntError> for Error {
    fn from(value: std::num::TryFromIntError) -> Self {
        Self::TryFromIntError(value.to_string())
    }
}
