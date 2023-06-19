use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

pub struct Retainer<T> {
    inner: Arc<T>,
}
unsafe impl<T: Send> Send for Retainer<T> {}
unsafe impl<T: Sync> Sync for Retainer<T> {}

pub struct RetainerHandle<T> {
    inner: Arc<T>,
}

impl<T> Retainer<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
    pub fn handle(&self) -> RetainerHandle<T> {
        RetainerHandle {
            inner: self.inner.clone(),
        }
    }
}

impl<T> Deref for Retainer<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref()
    }
}

impl<T> DerefMut for Retainer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { Arc::get_mut_unchecked(&mut self.inner) }
    }
}
