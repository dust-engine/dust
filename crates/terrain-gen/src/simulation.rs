use dust_utils::hlist::*;
use dust_utils::hlist_bound;

hlist_bound![PhaseHList: Phase];

trait Phase: Clone {
    type Inputs: PhaseHList;

    fn execute(&self, inputs: Self::Inputs);
}

trait SelfPhases<Tag, SubsetTag>: Phases<Self, Tag, SubsetTag> + Clone {
    fn execute_all(&self) {
        <Self as Phases<Self, Tag, SubsetTag>>::execute_all(self)
    }
}
impl<T, Tag, SubsetTag> SelfPhases<Tag, SubsetTag> for T where T: Phases<T, Tag, SubsetTag> + Clone {}

trait Phases<L: PhaseHList + Clone, Tag, SubsetTag>: PhaseHList + SubsetOf<L, SubsetTag> {
    fn execute_all(this: &L);
}

impl<L: PhaseHList + Clone> Phases<L, HNil, HNil> for HNil {
    fn execute_all(_this: &L) {}
}
impl<
        L: PhaseHList + Clone,
        H: Phase,
        T: Phases<L, TT, TST>,
        TT: HList,
        HT,
        ST,
        TST: HList,
        HST,
    > Phases<L, HCons<(HT, HST), TT>, HCons<ST, TST>> for HCons<H, T>
where
    H::Inputs: Phases<L, HT, HST>,
    L: HContains<H, ST>,
{
    fn execute_all(this: &L) {
        H::Inputs::execute_all(this);
        let head = this.h_get();
        head.execute(H::Inputs::subset(this.clone()));
        T::execute_all(this);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Debug, Copy, Clone)]
    struct First(u32);
    #[derive(Debug, Clone)]
    struct Second(String);
    #[derive(Debug, Copy, Clone)]
    struct Third(bool);

    impl Phase for First {
        type Inputs = HNil;

        fn execute(&self, _inputs: Self::Inputs) {
            println!("First: {:?}", self);
        }
    }
    impl Phase for Second {
        type Inputs = HCons<First, HNil>;

        fn execute(&self, inputs: Self::Inputs) {
            println!("Second: {:?}. Inputs: {:?}", self, inputs);
        }
    }
    impl Phase for Third {
        type Inputs = HCons<First, HCons<Second, HNil>>;

        fn execute(&self, inputs: Self::Inputs) {
            println!("Third: {:?}. Inputs: {:?}", self, inputs);
        }
    }

    #[test]
    fn test() {
        let list = HCons::new(
            First(10),
            HCons::new(Third(false), HCons::new(Second("thing".to_string()), HNil)),
        );
        list.execute_all();
    }
}
