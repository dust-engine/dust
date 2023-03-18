
#[derive(bevy_reflect::TypeUuid)]
#[uuid = "c7713cf2-527f-45ac-8eed-cbbcdc7302fd"]
pub struct VoxPalette(pub Box<[dot_vox::Color; 255]>);
