pub trait HMap {
    type Map<T>;
}

pub trait HValueMap: HMap {
    fn map<T>(t: T) -> Self::Map<T>;
}

pub trait HFunctor {
    type Mapped<M: HMap>;
    fn map<M: HValueMap>(self) -> Self::Mapped<M>;
}
