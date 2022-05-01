use bevy_app::Plugin;

pub struct BlasHandle;

/// Resource in Render World.
pub struct BlasStore {

}

/// This plugin generates a BLAS for each unique combination of geometries.
pub struct GeometryBlasPlugin;

impl Plugin for GeometryBlasPlugin {
    fn build(&self, app: &mut bevy_app::App) {
    }
}


/// For each Renderable, look at all its children and their geometries.
/// Generate a unique BlasHandle for each unique combination of geometries.
fn geometry_blas_system() {

}


