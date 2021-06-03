use crate::num;
use derive_new::*;

pub trait HList: private::Sealed {}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub struct HNil;
impl private::Sealed for HNil {}
impl HList for HNil {}

#[derive(new, Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub struct HCons<H, T: HList> {
    pub head: H,
    pub tail: T,
}
impl<H, T: HList> private::Sealed for HCons<H, T> {}
impl<H, T: HList> HList for HCons<H, T> {}

pub trait HContains<T, Tag> {
    fn h_get(&self) -> &T;
    fn h_get_mut(&mut self) -> &mut T;
    fn h_extract(self) -> T;
}

impl<H, T: HList> HContains<H, num::Z> for HCons<H, T> {
    #[inline]
    fn h_get(&self) -> &H {
        &self.head
    }
    #[inline]
    fn h_get_mut(&mut self) -> &mut H {
        &mut self.head
    }
    #[inline]
    fn h_extract(self) -> H {
        self.head
    }
}

impl<H, S, T: HList, N: num::Count> HContains<S, num::S<N>> for HCons<H, T>
where
    T: HContains<S, N>,
{
    #[inline]
    fn h_get(&self) -> &S {
        self.tail.h_get()
    }
    #[inline]
    fn h_get_mut(&mut self) -> &mut S {
        self.tail.h_get_mut()
    }
    #[inline]
    fn h_extract(self) -> S {
        self.tail.h_extract()
    }
}

pub trait SubsetOf<X, Tag> {
    fn subset(x: X) -> Self;
}

impl<X> SubsetOf<X, HNil> for HNil {
    #[inline]
    fn subset(_: X) -> Self {
        HNil
    }
}

impl<X, H, HT, T, TT> SubsetOf<X, HCons<HT, TT>> for HCons<H, T>
where
    T: HList + SubsetOf<X, TT>,
    TT: HList,
    X: HContains<H, HT> + Clone,
{
    #[inline]
    fn subset(x: X) -> Self {
        HCons {
            head: x.clone().h_extract(),
            tail: SubsetOf::subset(x),
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test() {
        let mut list = HCons::new(1, HCons::new("foo", HCons::new(true, HNil)));
        let first: &mut u32 = list.h_get_mut();
        *first = 2;
        let second: &str = *list.h_get();
        assert_eq!(second, "foo");
        let subset = <HCons<u32, HCons<bool, HNil>> as SubsetOf<_, _>>::subset(list);
        assert_eq!(subset, HCons::new(2, HCons::new(true, HNil)));
    }
}

mod private {
    pub trait Sealed {}
}
