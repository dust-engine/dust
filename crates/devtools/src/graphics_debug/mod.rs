//! The "Graphics Debug" DevTools tab: a sidebar of pages on the left and the
//! selected page on the right. The sidebar groups the pages into sections
//! split by separators: information pages such as "GPU Info" first, then the
//! render stages, in the order they run within a frame, each showing its
//! debug toggles.

mod gpu;
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
        let info = vec![("GPU Info", Some(world.register_system(gpu::gpu_info_ui)))];
        let render_pipeline = vec![
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
        ];
        let sections: Vec<Vec<Page>> = [info, render_pipeline]
            .into_iter()
            .map(|section| {
                section
                    .into_iter()
                    .map(|(name, contents)| Page { name, contents })
                    .collect()
            })
            .collect();
        app.insert_resource(GraphicsDebug {
            sections,
            // "GPU Info".
            selected: (0, 0),
        });
        sky::build(app);
        app.add_devtools_tab("Graphics Debug", graphics_debug_ui);
    }
}

struct Page {
    name: &'static str,
    /// Draws the page into the `Ui` it is given; `None` for a render stage
    /// without debug toggles.
    contents: Option<SystemId<InMut<'static, egui::Ui>>>,
}

#[derive(Resource)]
struct GraphicsDebug {
    /// The sidebar's sections, top to bottom.
    sections: Vec<Vec<Page>>,
    /// Section and page index of the shown page.
    selected: (usize, usize),
}

fn graphics_debug_ui(InMut(ui): InMut<egui::Ui>, world: &mut World) {
    world.resource_scope(|world, mut debug: Mut<GraphicsDebug>| {
        let debug = &mut *debug;
        egui::Panel::left("graphics_debug_pages")
            .resizable(true)
            .default_size(160.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.with_layout(egui::Layout::top_down_justified(egui::Align::Min), |ui| {
                        for (s, pages) in debug.sections.iter().enumerate() {
                            if s > 0 {
                                ui.separator();
                            }
                            for (p, page) in pages.iter().enumerate() {
                                let mut name = egui::RichText::new(page.name);
                                if page.contents.is_none() {
                                    name = name.weak();
                                }
                                ui.selectable_value(&mut debug.selected, (s, p), name);
                            }
                        }
                    });
                });
            });
        let (s, p) = debug.selected;
        let contents = debug
            .sections
            .get(s)
            .and_then(|pages| pages.get(p))
            .and_then(|page| page.contents);
        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink(false)
                .show(ui, |ui| match contents {
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
