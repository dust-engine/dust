trait Phase {
    type Value;
    type Inputs: HList;

    fn execute(&mut self, inputs: Self::Inputs) -> State<Self::Value>;
}
