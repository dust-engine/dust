//! Player overlay: a small overlay reporting the player's position and look
//! orientation.

use bevy::prelude::*;
use dust_character::Character;
use gpui::{
    Context, InteractiveElement as _, IntoElement, ParentElement as _, Render, Styled as _, div, px,
};
use gpui_component::{ActiveTheme as _, h_flex, v_flex};

use crate::watch;

pub(crate) struct PlayerSnapshot {
    position: Vec3,
    /// In radians.
    yaw: f32,
    pitch: f32,
}

/// The player's position and orientation, when either changed.
fn player_snapshot(
    characters: Query<(&Character, &Transform), Or<(Changed<Character>, Changed<Transform>)>>,
) -> Option<PlayerSnapshot> {
    characters
        .iter()
        .next()
        .map(|(character, transform)| PlayerSnapshot {
            position: transform.translation,
            yaw: character.yaw,
            pitch: character.pitch,
        })
}

/// The overlay, semi-transparent, in the bottom-left corner of the scene.
pub(crate) struct PlayerOverlay {
    snapshot: Option<PlayerSnapshot>,
}

impl PlayerOverlay {
    pub fn new(_: &mut gpui::Window, _: &mut Context<Self>) -> Self {
        PlayerOverlay { snapshot: None }
    }
}

impl Render for PlayerOverlay {
    fn render(&mut self, window: &mut gpui::Window, cx: &mut Context<Self>) -> impl IntoElement {
        watch(window, cx, player_snapshot, |this, snapshot, _, _| {
            this.snapshot = Some(snapshot)
        });
        let mono = cx.theme().mono_font_family.clone();
        let row = |label: &'static str, value: String| {
            h_flex()
                .gap_2()
                .child(
                    div()
                        .w(px(80.))
                        .text_color(gpui::white().opacity(0.6))
                        .child(label),
                )
                .child(div().font_family(mono.clone()).child(value))
        };
        let contents = match &self.snapshot {
            Some(player) => {
                let Vec3 { x, y, z } = player.position;
                // `yaw` accumulates without bound as the mouse turns; wrap it
                // into a compass-style 0–360° range.
                let yaw = player.yaw.to_degrees().rem_euclid(360.0);
                let pitch = player.pitch.to_degrees();
                v_flex()
                    .gap_0p5()
                    .child(row("Position", format!("{x:.2}, {y:.2}, {z:.2}")))
                    .child(row(
                        "Orientation",
                        format!("yaw {yaw:.1}°, pitch {pitch:.1}°"),
                    ))
            }
            None => v_flex().child("No player."),
        };
        div()
            .id("player-overlay")
            // The position can be selected and copied, so the overlay takes the
            // mouse.
            .occlude()
            .p_2()
            .rounded_md()
            .bg(gpui::black().opacity(0.85))
            .text_sm()
            .text_color(gpui::white())
            .child(contents)
    }
}
