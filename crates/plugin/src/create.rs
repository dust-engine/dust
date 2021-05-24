use cargo_metadata::Metadata;
use cargo_metadata::Package;
use serde_json::Value;

fn find_plugin_dependencies(metadata: &Metadata) -> Vec<Package> {
    metadata
        .packages
        .iter()
        .cloned()
        .filter(|package| {
            if let Value::Object(metadata) = &package.metadata {
                if let Some(Value::Object(dust_metadata)) = metadata.get("dust") {
                    if dust_metadata["plugin"] == Value::Bool(true) {
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            }
        })
        .collect()
}
