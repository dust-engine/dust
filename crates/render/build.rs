const GLSL_SHADER_FILES: [&str; 2] = ["./src/ray.frag", "./src/ray.vert"];

fn main() {
    /*
    for shader_file_path in &GLSL_SHADER_FILES {
        let input = std::fs::read_to_string(shader_file_path)
            .unwrap();
        let stage = if shader_file_path.ends_with(".frag") {
            naga::ShaderStage::Fragment
        } else if shader_file_path.ends_with(".vert") {
            naga::ShaderStage::Vertex
        } else if shader_file_path.ends_with(".comp") {
            naga::ShaderStage::Compute
        } else {
            panic!("Unknown shader type: {}", shader_file_path)
        };
        let mut entry_points = naga::FastHashMap::default();
        entry_points.insert("main".to_string(), stage);
        let module = naga::front::glsl::parse_str(
            &input,
            &naga::front::glsl::Options {
                entry_points,
                defines: Default::default()
            }
        )
            .unwrap();
        let analysis = naga::proc::Validator::new().validate(&module).unwrap();
        naga::back::spv::write_vec(
            &module,
            &analysis,
            &naga::back::spv::Options::default()
        );
    }
     */
}
