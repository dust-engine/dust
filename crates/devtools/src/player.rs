//! Player overlay: a small egui overlay reporting the player's position and
//! look orientation.

use bevy::prelude::*;
use dust_character::Character;
use pumicite_egui::{EguiContexts, EguiPrimaryContextPass, egui};

use crate::devtools::{DevTools, draw_devtools};

/// Adds the player overlay.
///
/// Like the other plugins in this crate, it draws into the primary egui
/// context set up by the renderer plugins and does not add `EguiPlugin` itself.
#[derive(Default)]
pub struct PlayerOverlayPlugin;

impl Plugin for PlayerOverlayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PlayerOverlay>();
        app.add_systems(EguiPrimaryContextPass, player_overlay.after(draw_devtools));
    }
}

/// Player overlay state.
#[derive(Resource)]
pub struct PlayerOverlay {
    /// Whether the overlay is shown. Toggled from the DevTools ⋮ menu; left
    /// public so an app can hide/show it programmatically.
    pub open: bool,
}

impl Default for PlayerOverlay {
    fn default() -> Self {
        Self { open: true }
    }
}

/// Draws the overlay, semi-transparent, in the bottom-left corner of
/// [`DevTools::scene_rect`].
fn player_overlay(
    mut contexts: EguiContexts,
    overlay: Res<PlayerOverlay>,
    characters: Query<(&Character, &Transform)>,
    devtools: Option<Res<DevTools>>,
) {
    if !overlay.open {
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    let scene = devtools.map_or_else(|| ctx.viewport_rect(), |devtools| devtools.scene_rect);
    egui::Area::new(egui::Id::new("player_overlay"))
        .pivot(egui::Align2::LEFT_BOTTOM)
        .fixed_pos(scene.left_bottom() + egui::vec2(8.0, -8.0))
        .show(ctx, |ui| {
            egui::Frame::new()
                .fill(egui::Color32::from_black_alpha(240))
                .corner_radius(egui::CornerRadius::same(6))
                .inner_margin(egui::Margin::same(8))
                .show(ui, |ui| match characters.iter().next() {
                    Some((character, transform)) => player_ui(ui, character, transform),
                    None => {
                        ui.weak("No player.");
                    }
                });
        });
}

/// Draws the overlay's contents.
fn player_ui(ui: &mut egui::Ui, character: &Character, transform: &Transform) {
    let Vec3 { x, y, z } = transform.translation;
    let position = format!("{x:.2}, {y:.2}, {z:.2}");
    egui::Grid::new("player_overlay_grid")
        .num_columns(2)
        .show(ui, |ui| {
            ui.label("Position");
            // A read-only `&str` buffer: the text can be selected and copied
            // but not edited.
            ui.add(
                egui::TextEdit::singleline(&mut position.as_str())
                    .font(egui::TextStyle::Monospace)
                    .desired_width(180.0),
            );
            ui.end_row();

            // `yaw` accumulates without bound as the mouse turns; wrap it into
            // a compass-style 0–360° range.
            let yaw = character.yaw.to_degrees().rem_euclid(360.0);
            let pitch = character.pitch.to_degrees();
            ui.label("Orientation");
            ui.label(egui::RichText::new(format!("yaw {yaw:.1}°, pitch {pitch:.1}°")).monospace());
            ui.end_row();
        });
}
