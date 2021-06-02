use crate::hkt::Mapping;
use crate::num;

pub trait HList: private::Sealed {
    type Mapped<M: Mapping>: HList;
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub struct HNil();
impl private::Sealed for HNil {}
impl HList for HNil {
    type Mapped<M: Mapping> = HNil;
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub struct HCons<H, T: HList> {
    pub head: H,
    pub tail: T,
}
impl<H, T: HList> private::Sealed for HCons<H, T> {}
impl<H, T: HList> HList for HCons<H, T> {
    type Mapped<M: Mapping> = HCons<M::Lift<H>, T::Mapped<M>>;
}

// Macros:
// bound_hlist![HListCopyCloneBound: Copy + Clone]
// bound_mapping! {
//     type ThisMapping = T: Copy + Clone =>> Wrapper<T>;
// }
// bound_mapping! {
//     type ThisMapping = <T: Copy + Clone>(t: T) -> Wrapper<T> {
//         Wrapper::new(t)
//     };
// }

trait HContains<T, Tag> {
    fn h_get(&self) -> &T;
    fn h_get_mut(&mut self) -> &mut T;
}

impl<H, T: HList> HContains<H, num::Z> for HCons<H, T> {
    fn h_get(&self) -> &H {
        &self.head
    }

    fn h_get_mut(&mut self) -> &mut H {
        &mut self.head
    }
}

impl<H, S, T: HList, N: num::Count> HContains<S, num::S<N>> for HCons<H, T> where T: HContains<S, N> {
    fn h_get(&self) -> &S {
        self.tail.h_get()
    }
    fn h_get_mut(&mut self) -> &mut S {
        self.tail.h_get_mut()
    }
}

#[macro_export]
macro_rules! hlist_bound {
    ($name:ident : $first:tt $(+ $next:tt)*) => {
        trait $name: dust_utils::hlist::HList {}
        impl $name for dust_utils::hlist::HNil {}
        impl<H: $first $(+ $next)*, T: $name> $name for dust_utils::hlist::HCons<H, T> {}
    }
}

mod private {
    pub trait Sealed {}
}
