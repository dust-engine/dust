use dust_utils::hlist::*;
use dust_utils::hlist_bound;

hlist_bound![PhaseHList: Phase];

trait Phase: Clone {
    type Inputs: PhaseHList;

    fn execute(&mut self, inputs: Self::Inputs);
}

trait Phases<L: PhaseHList, Tag>: PhaseHList {
    fn run(&mut self);
}

impl<L: PhaseHList> Phases<L, HNil> for HNil {
    fn run(&mut self) {}
}
impl<L: PhaseHList, H: Phase, T: Phases<L, TT>, TT: HList, HT> Phases<L, HCons<HT, TT>>
    for HCons<H, T>
where
    H::Inputs: SubsetOf<L, HT>,
{
    fn run(&mut self) {}
}
