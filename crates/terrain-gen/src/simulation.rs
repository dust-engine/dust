use dust_utils::hlist::*;
use dust_utils::hkt::Mapping;

struct State<T>(T);

struct InputMapping;
impl Mapping for InputMapping {
    type Lift<T> = (State<T/*::Value*/>, T);
}

trait HListCopyBound: HList {}
impl HListCopyBound for HNil {}
impl<Head: Copy, Tail: HListCopyBound> HListCopyBound for HCons<Head, Tail> {}

trait Phase {
    type Value;
    type Inputs: HList;

    fn execute(&mut self, inputs: <Self::Inputs as HList>::Mapped<InputMapping>) -> State<Self::Value>;
}
