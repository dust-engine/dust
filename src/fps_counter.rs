use bevy::prelude::*;
use std::collections::VecDeque;

pub struct FPSCounter {
    events: VecDeque<u128>,
    first_record: Option<u128>,
}

impl Default for FPSCounter {
    fn default() -> Self {
        FPSCounter {
            events: VecDeque::with_capacity(150),
            first_record: None,
        }
    }
}

pub fn fps_counter(time: Res<Time>, mut counter: ResMut<FPSCounter>, mut windows: ResMut<Windows>) {
    let now = time.time_since_startup().as_millis();
    counter.events.push_back(now);
    if let Some(first_record) = counter.first_record {
        if now - first_record < 1000 {
            return;
        }
    } else {
        counter.first_record = Some(now);
        return;
    }
    let a_second_ago = now - 1000;
    while counter.events.front().map_or(false, |v| *v < a_second_ago) {
        counter.events.pop_front();
    }
    let fps = counter.events.len();

    let window = windows.get_primary_mut().unwrap();
    window.set_title(format!("Bevy Demo(fps: {})", fps));
}
