//! The "Graphics Debug" DevTools tab: the render stages on the left, in the
//! order they run within a frame, and the selected stage's debug toggles on
//! the right.

mod screen_trace;
mod sharc;
mod sky;

use bevy::ecs::system::SystemId;
use bevy::prelude::*;
use pumicite_egui::egui;

use crate::devtools::DevToolsAppExt;

/// Adds the "Graphics Debug" DevTools tab.
#[derive(Default)]
pub struct GraphicsDebugPlugin;

impl Plugin for GraphicsDebugPlugin {
    fn build(&self, app: &mut App) {
        let world = app.world_mut();
        let stages: Vec<RenderStage> = [
            // Drawn first: primary rays skip the pixels opaque UI covers.
            ("egui overlay", None),
            ("Sky atmosphere", Some(world.register_system(sky::egui_ui))),
            ("Primary rays", None),
            ("HZB build", None),
            ("Shadow rays", None),
            (
                "Screen trace",
                Some(world.register_system(screen_trace::screen_trace_ui)),
            ),
            ("Final gather", None),
            ("SHARC", Some(world.register_system(sharc::sharc_debug_ui))),
            ("Auto-exposure", None),
            ("Super-resolution", None),
            ("Tonemap", None),
        ]
        .into_iter()
        .map(|(name, toggles)| RenderStage { name, toggles })
        .collect();
        let selected = stages
            .iter()
            .position(|stage| stage.toggles.is_some())
            .unwrap_or(0);
        app.insert_resource(GraphicsDebug { stages, selected });
        sky::build(app);
        app.add_devtools_tab("Graphics Debug", graphics_debug_ui);
    }
}

struct RenderStage {
    name: &'static str,
    /// Draws the stage's debug toggles into the `Ui` it is given; `None` for a
    /// stage without any.
    toggles: Option<SystemId<InMut<'static, egui::Ui>>>,
}

#[derive(Resource)]
struct GraphicsDebug {
    stages: Vec<RenderStage>,
    /// Index into `stages` of the stage whose toggles are shown.
    selected: usize,
}

fn graphics_debug_ui(InMut(ui): InMut<egui::Ui>, world: &mut World) {
    world.resource_scope(|world, mut debug: Mut<GraphicsDebug>| {
        let debug = &mut *debug;
        egui::Panel::left("graphics_debug_stages")
            .resizable(true)
            .default_size(160.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.with_layout(egui::Layout::top_down_justified(egui::Align::Min), |ui| {
                        for (i, stage) in debug.stages.iter().enumerate() {
                            let mut name = egui::RichText::new(stage.name);
                            if stage.toggles.is_none() {
                                name = name.weak();
                            }
                            ui.selectable_value(&mut debug.selected, i, name);
                        }
                    });
                });
            });
        let toggles = debug
            .stages
            .get(debug.selected)
            .and_then(|stage| stage.toggles);
        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink(false)
                .show(ui, |ui| match toggles {
                    Some(system) => {
                        if let Err(err) = world.run_system_with(system, ui) {
                            ui.colored_label(ui.visuals().error_fg_color, err.to_string());
                        }
                    }
                    None => {
                        ui.weak("No debug toggles for this stage.");
                    }
                });
        });
    });
}
