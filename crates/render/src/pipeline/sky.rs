use bevy_ecs::system::Resource;
use bevy_math::{Vec3, Vec3A, Vec4, Vec2};
use crevice::std430::AsStd430;

#[derive(Debug, Resource)]
pub struct Sunlight {
    // 1 - 10
    pub turbidity: f32,
    pub albedo: Vec3A,

    /// Direction from eye to sun
    pub direction: Vec3A,
}

impl Default for Sunlight {
    fn default() -> Self {
        Self {
            turbidity: 1.0,
            albedo: Vec3A::splat(0.2),
            direction: Vec3A::new(0.0, 0.80114365, -0.5984721),
        }
    }
}

mod dataset {
    use bevy_math::Vec3;

    #[repr(C)] // guarantee 'bytes' comes after '_align'
    struct AlignedTo<Align, Bytes: ?Sized> {
        _align: [Align; 0],
        bytes: Bytes,
    }

    const RAW_DAT: &'static AlignedTo<f32, [u8]> = &AlignedTo {
        _align: [],
        bytes: *include_bytes!("dataset.bin"),
    };
    const RAW_CONFIG: &'static [Vec3] =
        unsafe { std::slice::from_raw_parts(RAW_DAT.bytes.as_ptr() as *const Vec3, 1080) };
    pub const CONFIG_LOW_ALBEDO: &'static [[[Vec3; 6]; 9]] =
        unsafe { std::slice::from_raw_parts(RAW_CONFIG.as_ptr() as *const [[Vec3; 6]; 9], 10) };
    pub const CONFIG_HIGH_ALBEDO: &'static [[[Vec3; 6]; 9]] = unsafe {
        std::slice::from_raw_parts(
            RAW_CONFIG.as_ptr().add(6 * 9 * 10) as *const [[Vec3; 6]; 9],
            10,
        )
    };

    const RAW_RAD: &'static [Vec3] =
        unsafe { std::slice::from_raw_parts(RAW_CONFIG.as_ptr().add(1080), 120) };
    pub const RAD_LOW_ALBEDO: &'static [[Vec3; 6]] =
        unsafe { std::slice::from_raw_parts(RAW_RAD.as_ptr() as *const [Vec3; 6], 10) };
    pub const RAD_HIGH_ALBEDO: &'static [[Vec3; 6]] =
        unsafe { std::slice::from_raw_parts(RAW_RAD.as_ptr().add(60) as *const [Vec3; 6], 10) };
}

#[derive(Clone, Debug, AsStd430)]
pub struct SkyModelChannelState {
    pub config1: Vec4,
    pub config2: Vec4,
    pub config3: f32,
    pub radiance: f32,
    padding: Vec2,
}

/// This is what you want to send to the shader.
#[derive(Clone, Debug, AsStd430)]
pub struct SkyModelState {
    pub r: SkyModelChannelState,
    pub g: SkyModelChannelState,
    pub b: SkyModelChannelState,
    pub direction: Vec4,
}

impl Sunlight {
    /// albedo: Ground albedo value between [0, 1]
    /// solar_elevation: Solar elevation in radians
    pub fn bake(&self) -> SkyModelState {
        let configs = cook_config(self.turbidity, self.albedo, self.direction.y);
        let radiances = cook_radiance_config(self.turbidity, self.albedo, self.direction.y);
        let mut configs = configs.map(|config| SkyModelChannelState {
            config1: Vec4::new(config[0], config[1], config[2], config[3]),
            config2: Vec4::new(config[4], config[5], config[6], config[7]),
            config3: config[8],
            radiance: 0.0,
            padding: Vec2::new(0.0, 0.0)
        });
        configs[0].radiance = radiances.x;
        configs[1].radiance = radiances.y;
        configs[2].radiance = radiances.z;
        SkyModelState {
            r: configs[0].clone(),
            g: configs[1].clone(),
            b: configs[2].clone(),
            direction: Vec4::new(self.direction.x, self.direction.y, self.direction.z, 0.0),
        }
    }
}

fn coefficient(elev_matrix: &[Vec3; 6], solar_elevation: f32) -> Vec3A {
    let rev_solar_elevation = 1.0 - solar_elevation;
    rev_solar_elevation.powi(5) * Vec3A::from(elev_matrix[0])
        + 5.0 * rev_solar_elevation.powi(4) * solar_elevation * Vec3A::from(elev_matrix[1])
        + 10.0 * rev_solar_elevation.powi(3) * solar_elevation.powi(2) * Vec3A::from(elev_matrix[2])
        + 10.0 * rev_solar_elevation.powi(2) * solar_elevation.powi(3) * Vec3A::from(elev_matrix[3])
        + 5.0 * rev_solar_elevation * solar_elevation.powi(4) * Vec3A::from(elev_matrix[4])
        + solar_elevation.powi(5) * Vec3A::from(elev_matrix[5])
}

fn cook_radiance_config(turbidity: f32, albedo: Vec3A, solar_elevation: f32) -> Vec3A {
    let int_turbidity: usize = turbidity as usize;
    let turbidity_rem = turbidity - int_turbidity as f32;

    let mut res: Vec3A;
    let solar_elevation = (solar_elevation / std::f32::consts::FRAC_PI_2).powf(1.0 / 3.0);

    // alb 0 low turb
    res = (1.0 - albedo)
        * (1.0 - turbidity_rem)
        * coefficient(&dataset::RAD_LOW_ALBEDO[int_turbidity - 1], solar_elevation);

    // alb 1 low turb
    res += albedo
        * (1.0 - turbidity_rem)
        * coefficient(
            &dataset::RAD_HIGH_ALBEDO[int_turbidity - 1],
            solar_elevation,
        );

    if int_turbidity == 10 {
        return res;
    }

    // alb 0 high turb
    res += (1.0 - albedo)
        * turbidity_rem
        * coefficient(&dataset::RAD_LOW_ALBEDO[int_turbidity], solar_elevation);

    // alb 1 high turb
    res += albedo
        * turbidity_rem
        * coefficient(&dataset::RAD_HIGH_ALBEDO[int_turbidity], solar_elevation);
    return res;
}

fn cook_config(turbidity: f32, albedo: Vec3A, solar_elevation: f32) -> [[f32; 9]; 3] {
    let mut config = [[0_f32; 9]; 3];
    let int_turbidity: usize = turbidity as usize;
    let turbidity_rem = turbidity - int_turbidity as f32;
    let solar_elevation = (solar_elevation / std::f32::consts::FRAC_PI_2).powf(1.0 / 3.0);

    // alb 0 low turb
    for i in 0..9 {
        // alb 0 low turb
        let mut result = (1.0 - albedo)
            * (1.0 - turbidity_rem)
            * coefficient(
                &dataset::CONFIG_LOW_ALBEDO[int_turbidity - 1][i],
                solar_elevation,
            );

        // alb 1 low turb
        result += albedo
            * (1.0 - turbidity_rem)
            * coefficient(
                &dataset::CONFIG_HIGH_ALBEDO[int_turbidity - 1][i],
                solar_elevation,
            );

        if int_turbidity < 10 {
            result += (1.0 - albedo)
                * turbidity_rem
                * coefficient(
                    &dataset::CONFIG_LOW_ALBEDO[int_turbidity][i],
                    solar_elevation,
                );

            result += albedo
                * turbidity_rem
                * coefficient(
                    &dataset::CONFIG_HIGH_ALBEDO[int_turbidity][i],
                    solar_elevation,
                );
        }

        config[0][i] = result.x;
        config[1][i] = result.y;
        config[2][i] = result.z;
    }

    config
}
