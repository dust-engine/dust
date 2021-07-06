use shaderc::{IncludeCallbackResult, IncludeType, ResolvedInclude};
use std::path::Path;

const GLSL_ENTRY_FILES: [&str; 1] = ["./src/shaders/ray.comp"];
const GLSL_INCLUDE_DIR: &str = "./src/shaders";
const MAX_INCLUDE_DEPTH: usize = 10;

fn include_callback(
    filename: &str,
    ty: IncludeType,
    _origin_filename: &str,
    depth: usize,
) -> IncludeCallbackResult {
    if depth > MAX_INCLUDE_DEPTH {
        panic!("Max include depth {} exceeded", MAX_INCLUDE_DEPTH);
    }
    let mut path = Path::new(GLSL_INCLUDE_DIR).to_path_buf();
    path.push(filename);
    match std::fs::read_to_string(&path) {
        Ok(content) => {
            println!("cargo:rerun-if-changed={}", path.to_str().unwrap());
            let resolved_name = path
                .into_os_string()
                .into_string()
                .expect("Path contains invalid characters");
            Ok(ResolvedInclude {
                resolved_name,
                content,
            })
        }
        Err(err) => Err(err.to_string()),
    }
}

fn main() {
    use shaderc::{CompileOptions, Compiler, Error, ShaderKind};
    let mut compiler = Compiler::new().unwrap();
    let mut options = CompileOptions::new().unwrap();
    options.set_target_spirv(shaderc::SpirvVersion::V1_3);
    options.set_include_callback(include_callback);

    let mut out_path = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).to_path_buf();
    out_path.push("shaders");
    std::fs::create_dir_all(&out_path).unwrap();

    for &shader_file_path in &GLSL_ENTRY_FILES {
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

        out_path.push(path.file_name().unwrap());
        out_path.set_extension(match stage {
            ShaderKind::Fragment => "frag.spv",
            ShaderKind::Vertex => "vert.spv",
            ShaderKind::Compute => "comp.spv",
            _ => unreachable!(),
        });
        std::fs::write(&out_path, &binary_result.as_binary_u8()).unwrap();
        out_path.pop();
    }
}
