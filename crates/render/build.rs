const GLSL_SHADER_FILES: [&str; 1] = ["./src/ray.comp"];

fn main() {
    use shaderc::{CompileOptions, Compiler, Error, ShaderKind};
    let mut compiler = Compiler::new().unwrap();
    let options = CompileOptions::new().unwrap();

    let mut out_path = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).to_path_buf();
    out_path.push("shaders");
    std::fs::create_dir_all(&out_path).unwrap();
    out_path.push("shader.spv");

    for &shader_file_path in &GLSL_SHADER_FILES {
        let path = std::path::Path::new(shader_file_path);
        println!("cargo:rerun-if-changed={}", path.to_str().unwrap());
        let input = std::fs::read_to_string(path).unwrap();

        let extension = path.extension().unwrap();
        let stage = match extension.to_str().unwrap() {
            "vert" => ShaderKind::Vertex,
            "frag" => ShaderKind::Fragment,
            "comp" => ShaderKind::Compute,
            _ => panic!("Extension {:?} not recognized", extension),
        };
        let binary_result = compiler.compile_into_spirv(
            &input,
            stage,
            path.file_name().unwrap().to_str().unwrap(),
            "main",
            Some(&options),
        );

        let binary_result = match binary_result {
            Ok(result) => result,
            Err(err) => match err {
                Error::CompilationError(ty, err) => {
                    panic!("Shader Error ({}): {}", ty, err)
                }
                Error::InternalError(err) => panic!("Shader Compilation Internal Error: {}", err),
                _ => panic!("Shader Compilation Failed: {:?}", err),
            },
        };

        assert_eq!(*binary_result.as_binary().first().unwrap(), 0x07230203);

        out_path.set_file_name(path.file_name().unwrap());
        out_path.set_extension(match stage {
            ShaderKind::Fragment => "frag.spv",
            ShaderKind::Vertex => "vert.spv",
            ShaderKind::Compute => "comp.spv",
            _ => unreachable!(),
        });
        std::fs::write(&out_path, &binary_result.as_binary_u8()).unwrap();
    }
}
