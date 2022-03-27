fn main() {
    let mut app = bevy_app::App::new();

    app.insert_resource(bevy_window::WindowDescriptor {
        title: "I am a window!".to_string(),
        width: 1280.,
        height: 800.,
        scale_factor_override: Some(1.0),
        ..Default::default()
    })
    .add_plugin(bevy_core::CorePlugin::default())
    .add_plugin(bevy_transform::TransformPlugin::default())
    .add_plugin(bevy_input::InputPlugin::default())
    .add_plugin(bevy_window::WindowPlugin::default())
    //.add_plugin(bevy::asset::AssetPlugin::default())
    //.add_plugin(dust_raytrace::DustPlugin::default())
    //add_plugin(bevy::scene::ScenePlugin::default())
    .add_plugin(bevy_winit::WinitPlugin::default())
    //.add_plugin(flycamera::FlyCameraPlugin)
    //.add_plugin(fps_counter::FPSCounterPlugin)
    .add_plugin(dust_render::RenderPlugin::default());

    {
        app.sub_app_mut(dust_render::RenderApp)
            .add_plugin(dust_render::swapchain::SwapchainPlugin::default());
    }
    app.run();
}
