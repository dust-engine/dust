use dust_utils::hlist::*;
use dust_utils::hlist_bound;
use dust_utils::hkt::Mapping;

struct State<T>(T);

struct InputMapping;
impl Mapping for InputMapping {
    type Lift<T> = (State<T/*::Value*/>, T);
}

hlist_bound![HListPhaseBound: Phase];

trait Phase {
    type Value;
    type Inputs: HListPhaseBound;

    fn execute(&mut self, inputs: <Self::Inputs as HList>::Mapped<InputMapping>) -> State<Self::Value>;
}
