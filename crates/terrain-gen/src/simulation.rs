use dust_utils::hlist::*;
use parking_lot::Mutex;

trait PhaseHList<'x>: HList + 'x {
    type State: StateHList<'x>;
    type ValueRef<'a>: HList
    where
        'x: 'a;
    type Ref<'a>: HList
    where
        'x: 'a;
}
impl<'x> PhaseHList<'x> for HNil {
    type State = HNil;
    type ValueRef<'a>
    where
        'x: 'a,
    = HNil;
    type Ref<'a>
    where
        'x: 'a,
    = HNil;
}
impl<'x, H: Phase<'x>, T: PhaseHList<'x>> PhaseHList<'x> for HCons<H, T> {
    type State = HCons<StateLock<'x, H>, T::State>;
    type ValueRef<'a>
    where
        'x: 'a,
    = HCons<&'a H::Value, T::ValueRef<'a>>;
    type Ref<'a>
    where
        'x: 'a,
    = HCons<&'a H, T::Ref<'a>>;
}

trait StateHList<'x>: HList + 'x {
    type Phase: PhaseHList<'x>;

    fn to_inputs<'a>(
        &'a self,
    ) -> (
        <Self::Phase as PhaseHList<'x>>::ValueRef<'a>,
        <Self::Phase as PhaseHList<'x>>::Ref<'a>,
    ) where 'x: 'a;
}
impl<'x> StateHList<'x> for HNil {
    type Phase = HNil;
    fn to_inputs<'a>(
        &'a self,
    ) -> (
        <Self::Phase as PhaseHList<'x>>::ValueRef<'a>,
        <Self::Phase as PhaseHList<'x>>::Ref<'a>,
    ) where 'x: 'a {
        (HNil, HNil)
    }
}
impl<'x, HP: Phase<'x>, T: StateHList<'x>> StateHList<'x> for HCons<StateLock<'x, HP>, T> {
    type Phase = HCons<HP, T::Phase>;
    fn to_inputs<'a>(
        &'a self,
    ) -> (
        <Self::Phase as PhaseHList<'x>>::ValueRef<'a>,
        <Self::Phase as PhaseHList<'x>>::Ref<'a>,
    ) where 'x: 'a {
        let HCons { head, tail } = self;
        let state = self
            .head
            .try_lock()
            .expect("Unable to unlock mutex. This is due to a cycle in the dependency graph");
        let (t_value_ref, t_ref) = tail.to_inputs();
        (HCons::new(&state.value.unwrap(), t_value_ref), HCons::new(&state.phase, t_ref))
    }
}

type StateLock<'x, P: Phase<'x>> = Mutex<State<'x, P>>;

struct State<'x, P: Phase<'x>> {
    phase: P,
    value: Option<P::Value>,
}

trait Phase<'x>: Clone + 'x {
    type Inputs: PhaseHList<'x>;
    type Value: 'x;

    fn execute(
        &mut self,
        inputs: <Self::Inputs as PhaseHList<'x>>::ValueRef<'_>,
        phase_inputs: <Self::Inputs as PhaseHList<'x>>::Ref<'_>,
    ) -> Self::Value;
}

trait SelfPhases<'x, Tag, SubsetTag>: Phases<'x, Self, Tag, SubsetTag> + Clone {
    fn execute_all(&self) {
        <Self as Phases<Self, Tag, SubsetTag>>::execute_all(self)
    }
}
impl<'x, T, Tag, SubsetTag> SelfPhases<'x, Tag, SubsetTag> for T where
    T: Phases<'x, T, Tag, SubsetTag> + Clone
{
}

trait Phases<'x, L: StateHList<'x> + Clone, Tag, SubsetTag>:
    StateHList<'x> + SubsetOf<L, SubsetTag>
{
    fn execute_all(this: &L);
}

impl<'x, L: StateHList<'x> + Clone> Phases<'x, L, HNil, HNil> for HNil {
    fn execute_all(_this: &L) {}
}
impl<
        'x,
        L: StateHList<'x> + Clone,
        HP: Phase<'x>,
        T: Phases<'x, L, TT, TST>,
        HT,
        TT: HList,
        ST,
        HST,
        TST: HList,
    > Phases<'x, L, HCons<(HT, HST), TT>, HCons<ST, TST>> for HCons<StateLock<'x, HP>, T>
where
    <HP::Inputs as PhaseHList<'x>>::State: Phases<'x, L, HT, HST>,
    L: HContains<StateLock<'x, HP>, ST>,
{
    fn execute_all(this: &L) {
        <HP::Inputs as PhaseHList>::State::execute_all(this);
        {
            let head = this
                .h_get()
                .try_lock()
                .expect("Unable to unlock mutex. This is due to a cycle in the dependency graph");
            if head.value.is_none() {
                head.value = Some(
                    head.phase
                        .execute(<HP::Inputs as PhaseHList<'x>>::State::subset(this.clone())),
                );
            }
        }
        T::execute_all(this);
    }
}

#[cfg(never_compile)]
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
