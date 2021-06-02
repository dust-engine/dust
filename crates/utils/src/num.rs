pub trait Count: private::Sealed {
    fn value(self) -> u32;
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub struct Z();
impl private::Sealed for Z {}
impl Count for Z {
    #[inline]
    fn value(self) -> u32 {
        0
    }
}
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub struct S<N: Count>(N);
impl<N: Count> private::Sealed for S<N> {}
impl<N: Count> Count for S<N> {
    #[inline]
    fn value(self) -> u32 {
        1 + self.0.value()
    }
}

mod private {
    pub trait Sealed {}
}
