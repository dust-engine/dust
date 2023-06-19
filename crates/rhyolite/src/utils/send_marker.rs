use std::ops::{Deref, DerefMut};

pub struct SendMarker<T>(T);

unsafe impl<T> Send for SendMarker<T> {}
unsafe impl<T> Sync for SendMarker<T> {}
impl<T> SendMarker<T> {
    pub unsafe fn new(inner: T) -> Self {
        Self(inner)
    }

    pub unsafe fn unwrap(self) -> T {
        self.0
    }
}

impl<T> Deref for SendMarker<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T> DerefMut for SendMarker<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
