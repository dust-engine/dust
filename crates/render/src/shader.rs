use bevy_reflect::TypeUuid;

#[derive(TypeUuid)]
#[uuid = "ec052e5b-03ab-443f-9eac-b368526350fa"]
pub enum Shader {
    Spirv(Box<[u32]>),
    Glsl(String),
}
