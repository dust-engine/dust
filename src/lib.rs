use bevy::app::{PluginGroup, PluginGroupBuilder};

pub use dust_pbr as pbr;
pub use dust_vdb as vdb;
pub use dust_vox as vox;
pub use rhyolite;

pub struct DustPlugin;

impl PluginGroup for DustPlugin {
    fn build(self) -> PluginGroupBuilder {
        let mut group = PluginGroupBuilder::start::<Self>();

        group = group
            .add(bevy::app::PanicHandlerPlugin)
            .add(dust_log::LogPlugin)
            .add(bevy::core::TaskPoolPlugin::default())
            .add(bevy::core::TypeRegistrationPlugin)
            .add(bevy::core::FrameCountPlugin)
            .add(bevy::time::TimePlugin)
            .add(bevy::transform::TransformPlugin)
            .add(bevy::hierarchy::HierarchyPlugin)
            .add(bevy::diagnostic::DiagnosticsPlugin)
            .add(bevy::input::InputPlugin)
            .add(bevy::window::WindowPlugin::default())
            .add(bevy::a11y::AccessibilityPlugin)
            .add(bevy::asset::AssetPlugin::default())
            .add(bevy::scene::ScenePlugin)
            .add::<bevy::winit::WinitPlugin>(bevy::winit::WinitPlugin::default());

        /*
        #[cfg(feature = "physics")]
        {
            use bevy_rapier3d::parry::query::{DefaultQueryDispatcher, QueryDispatcher};
            use std::sync::Arc;
            let query_dispatcher = DefaultQueryDispatcher.chain(dust_vdb::VdbQueryDispatcher);
            group = group.add(
                bevy_rapier3d::plugin::RapierPhysicsPlugin::<()>::default()
                    .with_query_dispatcher(Arc::new(query_dispatcher))
                    .with_narrow_phase_dispatcher(Arc::new(query_dispatcher)),
            );

            #[cfg(feature = "debug")]
            {
                group = group.add(bevy_rapier3d::prelude::RapierDebugRenderPlugin::default());
            }
        }
        */

        group = group.add(rhyolite::SurfacePlugin::default());

        #[cfg(feature = "debug")]
        {
            group = group.add(rhyolite::debug::DebugUtilsPlugin::default());
        }

        group = group
            .add(rhyolite::RhyolitePlugin::default())
            .add(rhyolite::SwapchainPlugin::default());

        #[cfg(feature = "gizmos")]
        {
            group = group
                .add(bevy::gizmos::GizmoPlugin)
                .add(rhyolite_gizmos::GizmosPlugin);
        }

        group = group
            .add(dust_pbr::PbrRendererPlugin)
            .add(dust_vox::VoxPlugin);
        group
    }
}
