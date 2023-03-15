use std::sync::Arc;

use bevy_ecs::prelude::Component;

struct SbtIndexInner {
    id: usize,
    sender: std::sync::mpsc::Sender<usize>,
}
impl Drop for SbtIndexInner {
    fn drop(&mut self) {
        self.sender.send(self.id).unwrap();
    }
}
unsafe impl Sync for SbtIndexInner {}

// This is to be included on the component of entities.
#[derive(Clone, Component)]
pub struct SbtIndex(Arc<SbtIndexInner>);

pub struct SbtManager {}

impl SbtManager {
    pub fn add(&mut self, _hitgroup_id: u32, _parameterse: &[u8]) {
        // for each unique combination of (hitgroup_id, parameters), return unique sbt index.
    }
}
