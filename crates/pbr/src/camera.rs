use std::ops::Range;

use bevy::{ecs::component::Component, math::UVec2};

pub struct Viewport {
    origin: UVec2,
    size: UVec2,
}

/// Parameters based on physical camera characteristics for calculating EV100
/// values for use with [`Exposure`]. This is also used for depth of field.
#[derive(Clone, Copy)]
pub struct PhysicalCameraParameters {
    /// <https://en.wikipedia.org/wiki/F-number>
    pub aperture_f_stops: f32,
    /// <https://en.wikipedia.org/wiki/Shutter_speed>
    pub shutter_speed_s: f32,
    /// <https://en.wikipedia.org/wiki/Film_speed>
    pub sensitivity_iso: f32,
}

impl PhysicalCameraParameters {
    /// Calculate the [EV100](https://en.wikipedia.org/wiki/Exposure_value).
    pub fn ev100(&self) -> f32 {
        bevy::math::ops::log2(
            self.aperture_f_stops * self.aperture_f_stops * 100.0
                / (self.shutter_speed_s * self.sensitivity_iso),
        )
    }
}

#[derive(Component)]
pub struct Camera {
    pub viewport: Option<Viewport>,
    pub depth: Range<f32>,
    /// Exposure compensation in stops (EV), applied on top of the metered
    /// auto-exposure (see `autoexposure_pass`). 0.0 trusts the meter's 18%
    /// mid-gray target; positive values brighten, negative darken.
    pub exposure: f32,

    /// The height of the [image sensor format] in meters.
    ///
    /// Focal length is derived from the FOV and this value. The default is
    /// 36mm, matching a [Full Frame DLSR] camera.
    ///
    /// [image sensor format]: https://en.wikipedia.org/wiki/Image_sensor_format
    ///
    /// [Full Frame DLSR]: https://en.wikipedia.org/wiki/Full-frame_DSLR
    pub sensor_width: f32,

    /// The focal length of the camera in meters.
    pub focal_length: f32,
}
impl Camera {
    pub fn fov(&self) -> f32 {
        2.0 * (self.sensor_width / (2.0 * self.focal_length)).atan()
    }
    pub fn tan_half_fov(&self) -> f32 {
        self.sensor_width / (2.0 * self.focal_length)
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            viewport: None,
            depth: 0.1..10000.0,
            // No compensation: trust the auto-exposure meter.
            exposure: 0.0,
            sensor_width: 0.036,
            focal_length: 0.035,
        }
    }
}
