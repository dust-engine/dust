use dust_utils::hlist::*;
use parking_lot::MappedRwLockReadGuard;
use parking_lot::RwLock;
use parking_lot::RwLockReadGuard;
use std::any::type_name;

trait PhaseHList<'x>: HList + 'x {
    type State: StateHList<'x>;
    type StateRef<'a>: StateRefHList<'x, 'a>
    where
        'x: 'a;
    type ValueRef<'a>: HList
    where
        'x: 'a;
    type Ref<'a>: HList
    where
        'x: 'a;

    fn to_state(self) -> Self::State;
}
impl<'x> PhaseHList<'x> for HNil {
    type State = HNil;
    type StateRef<'a>
    where
        'x: 'a,
    = HNil;
    type ValueRef<'a>
    where
        'x: 'a,
    = HNil;
    type Ref<'a>
    where
        'x: 'a,
    = HNil;

    fn to_state(self) -> Self::State {
        HNil
    }
}
impl<'x, H: Phase<'x>, T: PhaseHList<'x>> PhaseHList<'x> for HCons<H, T> {
    type State = HCons<StateLock<'x, H>, T::State>;
    type StateRef<'a>
    where
        'x: 'a,
    = HCons<&'a StateLock<'x, H>, T::StateRef<'a>>;
    type ValueRef<'a>
    where
        'x: 'a,
    = HCons<MappedRwLockReadGuard<'a, H::Value>, T::ValueRef<'a>>;
    type Ref<'a>
    where
        'x: 'a,
    = HCons<RwLockReadGuard<'a, H>, T::Ref<'a>>;

    fn to_state(self) -> Self::State {
        HCons::new(
            StateLock {
                phase: RwLock::new(self.head),
                value: RwLock::new(None),
            },
            self.tail.to_state(),
        )
    }
}

trait StateHList<'x>: HList + 'x {
    type Ref<'a>: StateRefHList<'x, 'a>
    where
        'x: 'a;

    fn to_ref<'a>(&'a self) -> Self::Ref<'a>
    where
        'x: 'a;
}
impl<'x> StateHList<'x> for HNil {
    type Ref<'a>
    where
        'x: 'a,
    = HNil;

    fn to_ref<'a>(&'a self) -> Self::Ref<'a>
    where
        'x: 'a,
    {
        HNil
    }
}
impl<'x, HP: Phase<'x>, T: StateHList<'x>> StateHList<'x> for HCons<StateLock<'x, HP>, T> {
    type Ref<'a>
    where
        'x: 'a,
    = HCons<&'a StateLock<'x, HP>, T::Ref<'a>>;

    fn to_ref<'a>(&'a self) -> Self::Ref<'a>
    where
        'x: 'a,
    {
        HCons::new(&self.head, self.tail.to_ref())
    }
}

trait StateRefHList<'x, 'a>: HList + 'a + Copy + Clone {
    type Phase: PhaseHList<'x>;
}
impl<'x, 'a> StateRefHList<'x, 'a> for HNil {
    type Phase = HNil;
}
impl<'x, 'a, HP: Phase<'x>, T: StateRefHList<'x, 'a>> StateRefHList<'x, 'a>
    for HCons<&'a StateLock<'x, HP>, T>
{
    type Phase = HCons<HP, T::Phase>;
}

#[derive(Debug)]
struct StateLock<'x, P: Phase<'x>> {
    phase: RwLock<P>,
    value: RwLock<Option<P::Value>>,
}

trait Phase<'x>: 'x {
    type Inputs: PhaseHList<'x>;
    type Value: 'x;

    fn execute<'a>(
        &'a mut self,
        inputs: <Self::Inputs as PhaseHList<'x>>::ValueRef<'a>,
        phase_inputs: <Self::Inputs as PhaseHList<'x>>::Ref<'a>,
    ) -> Self::Value
    where
        'x: 'a;
}

trait SelfPhases<'x: 'a, 'a, Tag, SubsetTag>: Phases<'x, 'a, Self, Tag, SubsetTag> {
    fn execute_all(self) {
        <Self as Phases<Self, Tag, SubsetTag>>::execute_all(self);
    }
}
impl<'x: 'a, 'a, T, Tag, SubsetTag> SelfPhases<'x, 'a, Tag, SubsetTag> for T where
    T: Phases<'x, 'a, T, Tag, SubsetTag>
{
}

trait Phases<'x: 'a, 'a, L: StateRefHList<'x, 'a>, Tag, SubsetTag>:
    StateRefHList<'x, 'a> + SubsetOf<L, SubsetTag>
{
    fn execute_all(
        this: L,
    ) -> (
        <Self::Phase as PhaseHList<'x>>::ValueRef<'a>,
        <Self::Phase as PhaseHList<'x>>::Ref<'a>,
    );
}

impl<'x: 'a, 'a, L: StateRefHList<'x, 'a>> Phases<'x, 'a, L, HNil, HNil> for HNil {
    fn execute_all(
        _this: L,
    ) -> (
        <Self::Phase as PhaseHList<'x>>::ValueRef<'a>,
        <Self::Phase as PhaseHList<'x>>::Ref<'a>,
    ) {
        (HNil, HNil)
    }
}
impl<
        'x: 'a,
        'a,
        L: StateRefHList<'x, 'a>,
        HP: Phase<'x>,
        T: Phases<'x, 'a, L, TT, TST>,
        HT,
        TT: HList,
        ST,
        HST,
        TST: HList,
    > Phases<'x, 'a, L, HCons<(HT, HST), TT>, HCons<ST, TST>> for HCons<&'a StateLock<'x, HP>, T>
where
    <HP::Inputs as PhaseHList<'x>>::StateRef<'a>: Phases<'x, 'a, L, HT, HST>,
    L: HContains<&'a StateLock<'x, HP>, ST>,
{
    fn execute_all(
        this: L,
    ) -> (
        <Self::Phase as PhaseHList<'x>>::ValueRef<'a>,
        <Self::Phase as PhaseHList<'x>>::Ref<'a>,
    ) {
        let state = this.h_get();
        let mut phase_value = (
            state
                .phase
                .try_read()
                .expect("Unable to lock RwLock due to a cycle in the dependency graph."),
            state
                .value
                .try_read()
                .expect("Unable to lock RwLock due to a cycle in the dependency graph."),
        );

        if phase_value.1.is_none() {
            take_mut::take(&mut phase_value, |(phase, value)| {
                drop(phase);
                drop(value);
                let mut phase = state
                    .phase
                    .try_write()
                    .expect("Unable to lock RwLock due to a cycle in the dependency graph.");
                let mut value = state
                    .value
                    .try_write()
                    .expect("Unable to lock RwLock due to a cycle in the dependency graph.");
                let (inputs, phase_inputs) = <<HP::Inputs as PhaseHList<'x>>::StateRef<'a> as Phases<
                    'x,
                    'a,
                    L,
                    HT,
                    HST,
                >>::execute_all(this);
                *value = Some(
                    phase.execute(unsafe { transmute::transmute(inputs) }, unsafe {
                        transmute::transmute(phase_inputs)
                    }),
                );
                drop(phase);
                drop(value);
                (
                    state
                        .phase
                        .try_read()
                        .expect("Invalid state. This should never happen."),
                    state
                        .value
                        .try_read()
                        .expect("Invalid state. This should never happen."),
                )
            });
        }

        let (t_value_ref, t_ref) = T::execute_all(this);
        (
            HCons::new(
                RwLockReadGuard::map(phase_value.1, |x| x.as_ref().expect("Invalid state.")),
                t_value_ref,
            ),
            HCons::new(phase_value.0, t_ref),
        )
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

    impl<'x> Phase<'x> for First {
        type Inputs = HNil;
        type Value = String;

        fn execute<'a>(
            &'a mut self,
            _inputs: <Self::Inputs as PhaseHList<'x>>::ValueRef<'a>,
            _phase_inputs: <Self::Inputs as PhaseHList<'x>>::Ref<'a>,
        ) -> Self::Value
        where
            'x: 'a,
        {
            println!("First: {:?}", self);
            "abcde".to_string()
        }
    }
    impl<'x> Phase<'x> for Second {
        type Inputs = HCons<First, HNil>;
        type Value = bool;

        fn execute<'a>(
            &'a mut self,
            inputs: <Self::Inputs as PhaseHList<'x>>::ValueRef<'a>,
            _phase_inputs: <Self::Inputs as PhaseHList<'x>>::Ref<'a>,
        ) -> Self::Value
        where
            'x: 'a,
        {
            println!("Second: {:?}. Inputs: {:?}", self, inputs);
            false
        }
    }
    impl<'x> Phase<'x> for Third {
        type Inputs = HCons<First, HCons<Second, HNil>>;
        type Value = ();

        fn execute<'a>(
            &'a mut self,
            inputs: <Self::Inputs as PhaseHList<'x>>::ValueRef<'a>,
            _phase_inputs: <Self::Inputs as PhaseHList<'x>>::Ref<'a>,
        ) -> Self::Value
        where
            'x: 'a,
        {
            println!("Third: {:?}. Inputs: {:?}", self, inputs);
            ()
        }
    }

    #[test]
    fn test() {
        let list = HCons::new(
            First(10),
            HCons::new(Third(false), HCons::new(Second("thing".to_string()), HNil)),
        )
        .to_state();
        SelfPhases::execute_all(list.to_ref());
        println!("{:?}", list);
    }
}
