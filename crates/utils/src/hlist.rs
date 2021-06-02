use crate::hkt::Mapping;

pub trait HList: private::Sealed {
    type Mapped<M: Mapping>: HList;
}

pub struct HNil();

pub struct HCons<Head, Tail: HList> {
    pub head: Head,
    pub tail: Tail,
}
impl private::Sealed for HNil {}
impl HList for HNil {
    type Mapped<M: Mapping> = HNil;
}

impl<Head, Tail: HList> private::Sealed for HCons<Head, Tail> {}
impl<Head, Tail: HList> HList for HCons<Head, Tail> {
    type Mapped<M: Mapping> = HCons<M::Lift<Head>, Tail::Mapped<M>>;
}

mod private {
    pub trait Sealed {}
}
