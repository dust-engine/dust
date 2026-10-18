//! The "GPU Info" page of the Graphics Debug tab: the physical device the
//! renderer runs on, its driver, memory heaps and the limits the renderer
//! cares about.

use bevy::prelude::*;
use bevy_pumicite::pumicite::ash::vk;
use bevy_pumicite::pumicite::physical_device::PhysicalDevice;
use pumicite_egui::egui;

/// A titled group of label / value rows.
pub(super) struct Section {
    title: &'static str,
    /// Whether the section starts expanded.
    open: bool,
    rows: Vec<(String, String)>,
}

pub(super) fn gpu_info_ui(
    InMut(ui): InMut<egui::Ui>,
    physical_device: Option<Res<PhysicalDevice>>,
    // The properties don't change while the app runs, so they're formatted
    // once.
    mut sections: Local<Vec<Section>>,
) {
    let Some(physical_device) = physical_device else {
        ui.weak("No Vulkan physical device.");
        return;
    };
    if sections.is_empty() {
        *sections = collect_sections(&physical_device);
    }

    if ui
        .button("Copy as text")
        .on_hover_text("Copy everything below, e.g. for a bug report.")
        .clicked()
    {
        ui.ctx().copy_text(sections_text(&sections));
    }
    ui.add_space(4.0);

    for section in sections.iter() {
        egui::CollapsingHeader::new(section.title)
            .default_open(section.open)
            .show(ui, |ui| {
                egui::Grid::new(("gpu_info", section.title))
                    .num_columns(2)
                    .striped(true)
                    .show(ui, |ui| {
                        for (label, value) in &section.rows {
                            ui.label(label);
                            ui.add(egui::Label::new(egui::RichText::new(value).monospace()).wrap());
                            ui.end_row();
                        }
                    });
            });
    }
}

fn sections_text(sections: &[Section]) -> String {
    let mut text = String::new();
    for section in sections {
        text.push_str(&format!("[{}]\n", section.title));
        for (label, value) in &section.rows {
            text.push_str(&format!("{label}: {value}\n"));
        }
        text.push('\n');
    }
    text
}

fn collect_sections(physical_device: &PhysicalDevice) -> Vec<Section> {
    let properties = physical_device.properties();
    let limits = &properties.limits;
    let driver = properties.get::<vk::PhysicalDeviceDriverProperties>();
    let subgroup = properties.get::<vk::PhysicalDeviceSubgroupProperties>();
    let subgroup_size_control = properties.get::<vk::PhysicalDeviceSubgroupSizeControlProperties>();
    let ray_tracing = properties.get::<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    let acceleration_structure =
        properties.get::<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();

    let device = vec![
        row("Name", properties.device_name().to_string_lossy()),
        row("Type", device_type_name(properties.device_type)),
        row(
            "Vendor",
            format!(
                "{} ({:#06x})",
                vendor_name(properties.vendor_id),
                properties.vendor_id
            ),
        ),
        row("Device ID", format!("{:#06x}", properties.device_id)),
        row("Vulkan", api_version(properties.api_version)),
    ];

    let conformance = driver.conformance_version;
    let driver = vec![
        row(
            "Name",
            driver
                .driver_name_as_c_str()
                .unwrap_or(c"unknown")
                .to_string_lossy(),
        ),
        row(
            "Info",
            driver
                .driver_info_as_c_str()
                .unwrap_or(c"unknown")
                .to_string_lossy(),
        ),
        row("ID", format!("{:?}", driver.driver_id)),
        row(
            "Version",
            driver_version(properties.vendor_id, properties.driver_version),
        ),
        row(
            "Conformance",
            format!(
                "{}.{}.{}.{}",
                conformance.major, conformance.minor, conformance.subminor, conformance.patch
            ),
        ),
    ];

    let heaps = properties
        .memory_heaps()
        .iter()
        .enumerate()
        .map(|(i, heap)| {
            let mut value = format_bytes(heap.size);
            if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                value.push_str(", device local");
            }
            row(format!("Heap {i}"), value)
        })
        .collect();
    let types = properties
        .memory_types()
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            row(
                format!("Type {i}"),
                format!("heap {}: {:?}", ty.heap_index, ty.property_flags),
            )
        })
        .collect();

    let [count_x, count_y, count_z] = limits.max_compute_work_group_count;
    let [size_x, size_y, size_z] = limits.max_compute_work_group_size;
    let limits = vec![
        row("Max image size (2D)", limits.max_image_dimension2_d),
        row("Max memory allocations", limits.max_memory_allocation_count),
        row(
            "Max storage buffer range",
            format_bytes(limits.max_storage_buffer_range.into()),
        ),
        row(
            "Max push constants",
            format_bytes(limits.max_push_constants_size.into()),
        ),
        row(
            "Max bound descriptor sets",
            limits.max_bound_descriptor_sets,
        ),
        row(
            "Max compute shared memory",
            format_bytes(limits.max_compute_shared_memory_size.into()),
        ),
        row(
            "Max workgroup count",
            format!("{count_x} × {count_y} × {count_z}"),
        ),
        row(
            "Max workgroup size",
            format!("{size_x} × {size_y} × {size_z}"),
        ),
        row(
            "Max workgroup invocations",
            limits.max_compute_work_group_invocations,
        ),
        row("Max sampler anisotropy", limits.max_sampler_anisotropy),
        row(
            "Timestamp period",
            format!("{} ns", limits.timestamp_period),
        ),
    ];

    let subgroup = vec![
        row("Size", subgroup.subgroup_size),
        row(
            "Size range",
            format!(
                "{}–{}",
                subgroup_size_control.min_subgroup_size, subgroup_size_control.max_subgroup_size
            ),
        ),
        row("Stages", format!("{:?}", subgroup.supported_stages)),
        row("Operations", format!("{:?}", subgroup.supported_operations)),
    ];

    let ray_tracing = vec![
        row("Max recursion depth", ray_tracing.max_ray_recursion_depth),
        row(
            "Max hit attribute size",
            format_bytes(ray_tracing.max_ray_hit_attribute_size.into()),
        ),
        row(
            "Max dispatch invocations",
            ray_tracing.max_ray_dispatch_invocation_count,
        ),
        row(
            "Shader group handle size",
            format_bytes(ray_tracing.shader_group_handle_size.into()),
        ),
        row(
            "Shader group handle alignment",
            format_bytes(ray_tracing.shader_group_handle_alignment.into()),
        ),
        row(
            "Shader group base alignment",
            format_bytes(ray_tracing.shader_group_base_alignment.into()),
        ),
        row(
            "Max BLAS geometries",
            acceleration_structure.max_geometry_count,
        ),
        row(
            "Max TLAS instances",
            acceleration_structure.max_instance_count,
        ),
        row(
            "Max BLAS primitives",
            acceleration_structure.max_primitive_count,
        ),
    ];

    vec![
        section("Device", true, device),
        section("Driver", true, driver),
        section("Memory heaps", true, heaps),
        section("Memory types", false, types),
        section("Limits", true, limits),
        section("Subgroups", false, subgroup),
        section("Ray tracing", false, ray_tracing),
    ]
}

fn section(title: &'static str, open: bool, rows: Vec<(String, String)>) -> Section {
    Section { title, open, rows }
}

fn row(label: impl Into<String>, value: impl ToString) -> (String, String) {
    (label.into(), value.to_string())
}

fn device_type_name(device_type: vk::PhysicalDeviceType) -> &'static str {
    match device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => "Discrete GPU",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "Integrated GPU",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "Virtual GPU",
        vk::PhysicalDeviceType::CPU => "CPU",
        _ => "Other",
    }
}

/// The PCI vendor IDs of common GPU vendors, and the Khronos vendor IDs of
/// vendors without one.
fn vendor_name(vendor_id: u32) -> &'static str {
    match vendor_id {
        0x1002 => "AMD",
        0x1010 => "Imagination",
        0x106b => "Apple",
        0x10de => "NVIDIA",
        0x13b5 => "ARM",
        0x5143 => "Qualcomm",
        0x8086 => "Intel",
        0x10005 => "Mesa",
        _ => "Unknown",
    }
}

fn api_version(version: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(version),
        vk::api_version_minor(version),
        vk::api_version_patch(version)
    )
}

/// `driverVersion` is vendor-specific; NVIDIA and Intel on Windows don't use
/// the Vulkan version encoding.
fn driver_version(vendor_id: u32, version: u32) -> String {
    match vendor_id {
        0x10de => format!(
            "{}.{}.{}.{}",
            version >> 22,
            (version >> 14) & 0xff,
            (version >> 6) & 0xff,
            version & 0x3f
        ),
        0x8086 if cfg!(windows) => format!("{}.{}", version >> 14, version & 0x3fff),
        _ => api_version(version),
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}
