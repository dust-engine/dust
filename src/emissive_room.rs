//! The `emissive_room` scene (`DUST_SCENE=emissive_room`): a dark ballroom lit
//! only by emissive voxels, for testing emissive lighting.
//!
//! The room is closed on all sides, so the sun and the sky add nothing: all of
//! its light comes from palette entries with an emission. The emitters cover
//! the range the final gather and SHARC have to handle, from wide, dim ones (the
//! rose window, the dance floor, the coffer panels) through mid-sized neon
//! strips and cove lights to hundreds of single-voxel fairy lights at high
//! intensity, in many colors. One light moves: the wisp, which circles the room.
//!
//! The room is drawn into a dense grid of palette values, then split into
//! models of at most 256³ voxels, the most one [`VoxGeometry`] holds.

use std::f32::consts::{PI, TAU};
use std::ops::Range;
use std::time::Instant;

use avian3d::prelude::*;
use bevy::math::{U8Vec4, ivec3};
use bevy::prelude::*;
use bevy::scene::CommandsSceneExt;
use bevy_pumicite::CreateDevice;
use dust_character::{Character, CharacterCamera};
use dust_vox::{
    VoxGeometry, VoxGpuMaterial, VoxInstance, VoxMaterial, VoxModel, VoxModelCollider, VoxPalette,
};
use pumicite::Allocator;

pub struct EmissiveRoomPlugin;

impl Plugin for EmissiveRoomPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_emissive_room.after(CreateDevice))
            .add_systems(Update, move_wisp);
    }
}

/// World units per voxel, as in the `.vox` assets: the character is 29 voxels tall.
const UNIT: f32 = 1.0 / 16.0;
/// The room's extent in voxels, shell included: 24 × 9.5 × 32 units.
const ROOM: IVec3 = IVec3::new(384, 152, 512);
/// Thickness of the walls, the floor and the ceiling slab.
const SHELL: i32 = 4;
/// The lowest empty layer, the floor's walking surface.
const FLOOR_TOP: i32 = SHELL;
/// The bottom layer of the ceiling slab, which the coffers recess up to.
const CEILING: i32 = ROOM.y - SHELL;
/// The bottom of the ceiling beams.
const BEAM_BOTTOM: i32 = CEILING - 12;
/// Spacing of the ceiling beams, the pilasters and the floor's brass inlay.
const BAY: i32 = 64;
const CENTER_X: i32 = ROOM.x / 2;
const CENTER_Z: i32 = ROOM.z / 2;
/// The extent of one model, the root node of the voxel tree.
const CHUNK: i32 = 256;
/// The wisp's model is `WISP_SIZE`³ voxels.
const WISP_SIZE: u32 = 16;

// Palette values. The tree stores `value - 1` as the palette index, and 0 means
// empty.

// Structure, all dark.
const FLOOR_DARK: u8 = 1;
const FLOOR_LIGHT: u8 = 2;
const WALL: u8 = 3;
const WALL_PANEL: u8 = 4;
const WAINSCOT: u8 = 5;
const STONE: u8 = 6;
const CEILING_PLASTER: u8 = 7;
const BRASS: u8 = 8;
const IRON: u8 = 9;
const WIRE: u8 = 10;
const WOOD: u8 = 11;
const CLOTH: u8 = 12;
const WAX: u8 = 13;

// Emitters.
const E_COVE: u8 = 32;
const E_NEON_MAGENTA: u8 = 33;
const E_NEON_CYAN: u8 = 34;
const E_SCONCE: u8 = 35;
const E_CANDLE: u8 = 36;
const E_BULB: u8 = 37;
const E_MARQUEE: u8 = 38;
const E_CRYSTAL: u8 = 39;
const E_COFFER: u8 = 40;
const E_ORB: u8 = 41;
const E_WATER: u8 = 42;
const E_AISLE: u8 = 43;
const E_STAGE: u8 = 44;
const E_LANTERN: u8 = 45;
const E_WISP: u8 = 46;
const E_SIGN: u8 = 47;
const E_GLASS_RUBY: u8 = 48;
const E_GLASS_SAPPHIRE: u8 = 49;
const E_GLASS_EMERALD: u8 = 50;
const E_GLASS_AMBER: u8 = 51;
const E_GLASS_VIOLET: u8 = 52;
const E_DANCE: [u8; 6] = [56, 57, 58, 59, 60, 61];
const E_FAIRY: [u8; 7] = [64, 65, 66, 67, 68, 69, 70];

/// How a palette entry reflects or emits light.
#[derive(Clone, Copy)]
enum Surface {
    /// Diffuse, with this roughness.
    Diffuse(f32),
    /// Metal, with this roughness.
    Metal(f32),
    /// Emits the color times this. The color is sRGB-encoded, and the shaders
    /// scale it as it is.
    Emit(f32),
}

/// Every palette entry the room uses: value, sRGB color, surface.
const MATERIALS: &[(u8, u32, Surface)] = &[
    (FLOOR_DARK, 0x0e0e11, Surface::Diffuse(0.15)),
    (FLOOR_LIGHT, 0x2a2a30, Surface::Diffuse(0.15)),
    (WALL, 0x3a1018, Surface::Diffuse(0.9)),
    (WALL_PANEL, 0x241018, Surface::Diffuse(0.9)),
    (WAINSCOT, 0x24160e, Surface::Diffuse(0.6)),
    (STONE, 0x34343a, Surface::Diffuse(0.5)),
    (CEILING_PLASTER, 0x1a1622, Surface::Diffuse(0.9)),
    (BRASS, 0xb08d3c, Surface::Metal(0.35)),
    (IRON, 0x1c1c1e, Surface::Metal(0.6)),
    (WIRE, 0x0c0c0c, Surface::Diffuse(0.8)),
    (WOOD, 0x3a2416, Surface::Diffuse(0.5)),
    (CLOTH, 0x5a0e1e, Surface::Diffuse(1.0)),
    (WAX, 0xe8e0cc, Surface::Diffuse(0.7)),
    // Wide and dim.
    (E_COFFER, 0x3048ff, Surface::Emit(1.5)),
    (E_WATER, 0x10c0b0, Surface::Emit(1.5)),
    (E_CRYSTAL, 0xb8e4ff, Surface::Emit(3.0)),
    (E_GLASS_RUBY, 0xd01030, Surface::Emit(3.0)),
    (E_GLASS_SAPPHIRE, 0x1838ff, Surface::Emit(3.0)),
    (E_GLASS_EMERALD, 0x10b048, Surface::Emit(3.0)),
    (E_GLASS_AMBER, 0xffa020, Surface::Emit(3.0)),
    (E_GLASS_VIOLET, 0x8a28e0, Surface::Emit(3.0)),
    (E_DANCE[0], 0xff2060, Surface::Emit(1.2)),
    (E_DANCE[1], 0xffa020, Surface::Emit(1.2)),
    (E_DANCE[2], 0x20ff60, Surface::Emit(1.2)),
    (E_DANCE[3], 0x20a0ff, Surface::Emit(1.2)),
    (E_DANCE[4], 0xa040ff, Surface::Emit(1.2)),
    (E_DANCE[5], 0xff40e0, Surface::Emit(1.2)),
    // Strips.
    (E_COVE, 0x18d8ff, Surface::Emit(4.0)),
    (E_NEON_MAGENTA, 0xff1ec8, Surface::Emit(6.0)),
    (E_NEON_CYAN, 0x20e8ff, Surface::Emit(6.0)),
    (E_STAGE, 0xffd070, Surface::Emit(6.0)),
    (E_SIGN, 0xff2a6a, Surface::Emit(8.0)),
    (E_AISLE, 0xffc050, Surface::Emit(8.0)),
    // Small and bright.
    (E_LANTERN, 0xffb860, Surface::Emit(12.0)),
    (E_SCONCE, 0xffa848, Surface::Emit(16.0)),
    (E_MARQUEE, 0xffcc70, Surface::Emit(20.0)),
    (E_ORB, 0xfff0d8, Surface::Emit(20.0)),
    (E_CANDLE, 0xff8c2a, Surface::Emit(24.0)),
    (E_WISP, 0x90e0ff, Surface::Emit(24.0)),
    (E_BULB, 0xffe0b0, Surface::Emit(32.0)),
    // Single voxels, brightest.
    (E_FAIRY[0], 0xff3030, Surface::Emit(48.0)),
    (E_FAIRY[1], 0xff9020, Surface::Emit(48.0)),
    (E_FAIRY[2], 0xffe040, Surface::Emit(48.0)),
    (E_FAIRY[3], 0x40ff60, Surface::Emit(48.0)),
    (E_FAIRY[4], 0x30e0ff, Surface::Emit(48.0)),
    (E_FAIRY[5], 0x4060ff, Surface::Emit(48.0)),
    (E_FAIRY[6], 0xff50d0, Surface::Emit(48.0)),
];

fn palette(allocator: Allocator) -> pumicite::ash::VkResult<VoxPalette> {
    VoxPalette::new(allocator, &palette_entries())
}

/// The palette's entries, indexed by `value - 1`.
fn palette_entries() -> Box<[VoxGpuMaterial; 256]> {
    let unorm = |x: f32| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
    let mut entries = Box::new([VoxGpuMaterial::default(); 256]);
    for &(value, rgb, surface) in MATERIALS {
        let entry = &mut entries[value as usize - 1];
        entry.color = U8Vec4::new((rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8, 255);
        // `packed` bytes, low to high: type, roughness, metalness, specular.
        let (packed, emission) = match surface {
            Surface::Diffuse(roughness) => (unorm(roughness) << 8, 0.0),
            Surface::Metal(roughness) => (1 | unorm(roughness) << 8 | 255 << 16, 0.0),
            Surface::Emit(emission) => (3, emission),
        };
        entry.packed = packed;
        entry.emission = emission as f16;
    }
    entries
}

/// The room's palette values, one byte per voxel.
struct Grid {
    cells: Vec<u8>,
}

impl Grid {
    fn new() -> Self {
        Self {
            cells: vec![0; (ROOM.x * ROOM.y * ROOM.z) as usize],
        }
    }

    fn index(p: IVec3) -> Option<usize> {
        (p.cmpge(IVec3::ZERO).all() && p.cmplt(ROOM).all())
            .then(|| ((p.z * ROOM.y + p.y) * ROOM.x + p.x) as usize)
    }

    fn get(&self, p: IVec3) -> u8 {
        Self::index(p).map_or(0, |i| self.cells[i])
    }

    /// Sets one voxel; 0 empties it. Voxels outside the room are ignored.
    fn set(&mut self, p: IVec3, value: u8) {
        if let Some(i) = Self::index(p) {
            self.cells[i] = value;
        }
    }

    /// Fills the box from `min` up to but not including `max`.
    fn fill(&mut self, min: IVec3, max: IVec3, value: u8) {
        for z in min.z..max.z {
            for y in min.y..max.y {
                for x in min.x..max.x {
                    self.set(ivec3(x, y, z), value);
                }
            }
        }
    }

    /// Fills the voxels whose centers are `radii` from the vertical axis
    /// through `(x, z)`, in the layers `y`: a disc, or a ring when the radii
    /// start above 0.
    fn cylinder(&mut self, x: f32, z: f32, radii: Range<f32>, y: Range<i32>, value: u8) {
        let reach = radii.end.ceil() as i32 + 1;
        let (x0, z0) = (x.floor() as i32, z.floor() as i32);
        for cz in z0 - reach..=z0 + reach {
            for cx in x0 - reach..=x0 + reach {
                let distance = Vec2::new(cx as f32 + 0.5 - x, cz as f32 + 0.5 - z).length();
                if radii.contains(&distance) {
                    for cy in y.clone() {
                        self.set(ivec3(cx, cy, cz), value);
                    }
                }
            }
        }
    }

    fn sphere(&mut self, center: Vec3, radius: f32, value: u8) {
        let min = (center - radius).floor().as_ivec3();
        let max = (center + radius).ceil().as_ivec3();
        for z in min.z..max.z {
            for y in min.y..max.y {
                for x in min.x..max.x {
                    let p = ivec3(x, y, z);
                    if (p.as_vec3() + 0.5 - center).length() < radius {
                        self.set(p, value);
                    }
                }
            }
        }
    }

    /// A line one voxel thick from `a` to `b`, in voxel coordinates.
    fn line(&mut self, a: Vec3, b: Vec3, value: u8) {
        let steps = ((b - a).length() * 2.0).ceil().max(1.0) as i32;
        for i in 0..=steps {
            let p = a.lerp(b, i as f32 / steps as f32);
            self.set(p.floor().as_ivec3(), value);
        }
    }
}

/// One of the four walls, in its own coordinates: `u` along it, `v` up from the
/// ground and `d` into the room from its face, where `d = 0` is the first layer
/// inside the room and negative `d` is in the wall.
#[derive(Clone, Copy)]
struct Wall {
    origin: IVec3,
    along: IVec3,
    inward: IVec3,
    len: i32,
}

impl Wall {
    const WEST: Wall = Wall {
        origin: IVec3::new(SHELL, 0, 0),
        along: IVec3::Z,
        inward: IVec3::X,
        len: ROOM.z,
    };
    const EAST: Wall = Wall {
        origin: IVec3::new(ROOM.x - SHELL - 1, 0, 0),
        along: IVec3::Z,
        inward: IVec3::NEG_X,
        len: ROOM.z,
    };
    /// The near wall, with the door.
    const SOUTH: Wall = Wall {
        origin: IVec3::new(0, 0, SHELL),
        along: IVec3::X,
        inward: IVec3::Z,
        len: ROOM.x,
    };
    /// The far wall, with the rose window.
    const NORTH: Wall = Wall {
        origin: IVec3::new(0, 0, ROOM.z - SHELL - 1),
        along: IVec3::X,
        inward: IVec3::NEG_Z,
        len: ROOM.x,
    };

    fn at(self, u: i32, v: i32, d: i32) -> IVec3 {
        self.origin + self.along * u + IVec3::Y * v + self.inward * d
    }

    fn fill(self, grid: &mut Grid, u: Range<i32>, v: Range<i32>, d: Range<i32>, value: u8) {
        for u in u {
            for v in v.clone() {
                for d in d.clone() {
                    grid.set(self.at(u, v, d), value);
                }
            }
        }
    }
}

fn build_room() -> Grid {
    let mut grid = Grid::new();
    let g = &mut grid;

    // The shell: floor, ceiling slab and walls, closed all round.
    g.fill(IVec3::ZERO, ivec3(ROOM.x, SHELL, ROOM.z), FLOOR_DARK);
    g.fill(ivec3(0, CEILING, 0), ROOM, CEILING_PLASTER);
    g.fill(IVec3::ZERO, ivec3(SHELL, ROOM.y, ROOM.z), WALL);
    g.fill(ivec3(ROOM.x - SHELL, 0, 0), ROOM, WALL);
    g.fill(IVec3::ZERO, ivec3(ROOM.x, ROOM.y, SHELL), WALL);
    g.fill(ivec3(0, 0, ROOM.z - SHELL), ROOM, WALL);

    floor(g);
    ceiling(g);
    walls(g);
    door(g);
    neon_sign(g);
    rose_window(g);
    stage(g);
    fountain(g);
    dance_floor(g);
    for x in [100, ROOM.x - 100] {
        for z in [96, 176, 256, 336, 400] {
            table(g, x, z);
        }
    }
    for z in [128, 384] {
        chandelier(g, CENTER_X, z);
    }
    string_lights(g);
    for (x, z) in [
        (24, 24),
        (ROOM.x - 24, 24),
        (24, ROOM.z - 24),
        (ROOM.x - 24, ROOM.z - 24),
    ] {
        floor_lantern(g, x, z);
    }
    grid
}

/// Black and charcoal marble checkerboard with a brass inlay on the bay grid,
/// and a line of lights down the aisle from the door to the fountain.
fn floor(g: &mut Grid) {
    let top = FLOOR_TOP - 1;
    for z in 0..ROOM.z {
        for x in 0..ROOM.x {
            let value = if x % BAY == 0 || z % BAY == 0 {
                BRASS
            } else if (x / 32 + z / 32) % 2 == 0 {
                FLOOR_DARK
            } else {
                FLOOR_LIGHT
            };
            g.set(ivec3(x, top, z), value);
        }
    }
    for z in (48..=208).step_by(32) {
        g.fill(
            ivec3(CENTER_X - 2, top, z - 2),
            ivec3(CENTER_X + 2, top + 1, z + 2),
            E_AISLE,
        );
    }
}

/// Beams on the bay grid with a brass stripe, and a recessed coffer between
/// them with a dim indigo panel in a brass frame.
fn ceiling(g: &mut Grid) {
    for z in 0..ROOM.z {
        for x in 0..ROOM.x {
            // 0..8 across a beam.
            let (bx, bz) = ((x + 4).rem_euclid(BAY), (z + 4).rem_euclid(BAY));
            if bx < 8 || bz < 8 {
                g.fill(
                    ivec3(x, BEAM_BOTTOM, z),
                    ivec3(x + 1, CEILING, z + 1),
                    CEILING_PLASTER,
                );
                if (3..5).contains(&bx) || (3..5).contains(&bz) {
                    g.set(ivec3(x, BEAM_BOTTOM, z), BRASS);
                }
            } else {
                // The coffer's center is at 32.
                let within = |c: i32, half: i32| (32 - half..32 + half).contains(&c);
                let (cx, cz) = (x.rem_euclid(BAY), z.rem_euclid(BAY));
                if within(cx, 6) && within(cz, 6) {
                    g.set(ivec3(x, CEILING, z), E_COFFER);
                } else if within(cx, 8) && within(cz, 8) {
                    g.set(ivec3(x, CEILING - 1, z), BRASS);
                }
            }
        }
    }
}

fn walls(g: &mut Grid) {
    for wall in [Wall::WEST, Wall::EAST, Wall::SOUTH, Wall::NORTH] {
        let u = SHELL..wall.len - SHELL;
        // Walnut wainscot.
        wall.fill(g, u.clone(), FLOOR_TOP..28, 0..2, WAINSCOT);
        // A cyan cove light along the floor, under a lip that hides it from
        // eye level: it grazes the floor.
        wall.fill(g, u.clone(), FLOOR_TOP..FLOOR_TOP + 3, 2..3, E_COVE);
        wall.fill(g, u.clone(), FLOOR_TOP + 3..FLOOR_TOP + 5, 0..6, STONE);
        // Brass chair rail.
        wall.fill(g, u, 28..30, 0..3, BRASS);
    }

    // The long walls: a pilaster under every beam, and a panel and a sconce in
    // every bay. The neon on the pilasters alternates, and is opposite across
    // the room.
    for (side, wall) in [Wall::WEST, Wall::EAST].into_iter().enumerate() {
        for bay in 0..ROOM.z / BAY {
            let start = bay * BAY;
            panel(g, wall, start + 14..start + 50);
            sconce(g, wall, start + 32);
            if bay > 0 {
                let neon = [E_NEON_MAGENTA, E_NEON_CYAN][(bay as usize + side) % 2];
                pilaster(g, wall, start, neon);
            }
        }
    }

    // The end walls: the outer bays only. The door and the rose window take the
    // middle.
    for wall in [Wall::SOUTH, Wall::NORTH] {
        for start in [0, ROOM.x - BAY] {
            panel(g, wall, start + 14..start + 50);
            sconce(g, wall, start + 32);
        }
        pilaster(g, wall, BAY, E_NEON_CYAN);
        pilaster(g, wall, ROOM.x - BAY, E_NEON_MAGENTA);
    }
    sconce(g, Wall::SOUTH, 112);
    sconce(g, Wall::SOUTH, ROOM.x - 112);
}

/// A velvet panel in a brass frame, above the chair rail.
fn panel(g: &mut Grid, wall: Wall, u: Range<i32>) {
    let v = 40..120;
    for pu in u.clone() {
        for pv in v.clone() {
            let edge = pu == u.start || pu == u.end - 1 || pv == v.start || pv == v.end - 1;
            if edge {
                wall.fill(g, pu..pu + 1, pv..pv + 1, 0..2, BRASS);
            } else {
                wall.fill(g, pu..pu + 1, pv..pv + 1, 0..1, WALL_PANEL);
            }
        }
    }
}

/// A stone pilaster centered at `u`, from the floor to the beam, with a neon
/// strip in a groove up its face.
fn pilaster(g: &mut Grid, wall: Wall, u: i32, neon: u8) {
    wall.fill(g, u - 10..u + 10, FLOOR_TOP..16, 0..10, STONE);
    wall.fill(g, u - 8..u + 8, 16..BEAM_BOTTOM, 0..8, STONE);
    wall.fill(g, u - 10..u + 10, 124..130, 0..10, BRASS);
    wall.fill(g, u - 2..u + 2, 20..118, 7..8, 0);
    wall.fill(g, u - 1..u + 1, 20..118, 6..7, neon);
}

/// A brass sconce centered at `u`: a backplate, an iron arm and a cup holding
/// a bare flame.
fn sconce(g: &mut Grid, wall: Wall, u: i32) {
    wall.fill(g, u - 3..u + 3, 44..58, 0..1, BRASS);
    wall.fill(g, u - 1..u + 1, 47..49, 1..6, IRON);
    wall.fill(g, u - 2..u + 2, 49..51, 3..7, BRASS);
    wall.fill(g, u - 1..u + 1, 51..55, 4..6, E_SCONCE);
}

/// Double doors in the near wall, in a stone frame ringed with marquee bulbs.
fn door(g: &mut Grid) {
    let w = Wall::SOUTH;
    let (u0, u1, top) = (CENTER_X - 24, CENTER_X + 24, 84);
    w.fill(g, u0 - 8..u1 + 8, FLOOR_TOP..top + 8, 0..4, STONE);
    w.fill(g, u0..u1, FLOOR_TOP..top, 0..4, 0);
    w.fill(g, u0..u1, FLOOR_TOP..top, 0..1, WOOD);
    w.fill(g, CENTER_X - 1..CENTER_X + 1, FLOOR_TOP..top, 0..1, IRON);
    w.fill(g, CENTER_X - 6..CENTER_X - 3, 40..50, 1..2, BRASS);
    w.fill(g, CENTER_X + 3..CENTER_X + 6, 40..50, 1..2, BRASS);

    let mut bulb = |u: i32, v: i32| w.fill(g, u - 1..u + 1, v - 1..v + 1, 4..5, E_MARQUEE);
    for v in (12..top + 4).step_by(8) {
        bulb(u0 - 4, v);
        bulb(u1 + 4, v);
    }
    for u in (u0 - 4..=u1 + 4).step_by(8) {
        bulb(u, top + 4);
    }
}

/// "DUST" in neon above the door, a little off the wall so it lights the wall
/// behind it.
fn neon_sign(g: &mut Grid) {
    #[rustfmt::skip]
    const GLYPHS: [[&str; 7]; 4] = [
        ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."],
        ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."],
        [".####", "#....", "#....", ".###.", "....#", "....#", "####."],
        ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."],
    ];
    const SCALE: i32 = 3;
    let advance = 7 * SCALE;
    let width = GLYPHS.len() as i32 * advance - 2 * SCALE;
    let (u0, v0) = (CENTER_X - width / 2, 100);
    for (i, glyph) in GLYPHS.iter().enumerate() {
        for (row, line) in glyph.iter().enumerate() {
            for (col, c) in line.bytes().enumerate() {
                if c == b'#' {
                    let u = u0 + i as i32 * advance + col as i32 * SCALE;
                    let v = v0 + (6 - row as i32) * SCALE;
                    Wall::SOUTH.fill(g, u..u + SCALE, v..v + SCALE, 2..3, E_SIGN);
                }
            }
        }
    }
}

/// A rose window of emissive glass in the far wall, behind stone tracery.
fn rose_window(g: &mut Grid) {
    let w = Wall::NORTH;
    let center = Vec2::new(CENTER_X as f32, 80.0);
    let (glass_radius, frame_radius) = (46.0, 54.0);
    let spoke = TAU / 12.0;
    let reach = frame_radius as i32;
    for u in CENTER_X - reach..CENTER_X + reach {
        for v in 80 - reach..80 + reach {
            let p = Vec2::new(u as f32 + 0.5, v as f32 + 0.5) - center;
            let r = p.length();
            if r >= frame_radius {
                continue;
            }
            if r >= glass_radius {
                let lip = r < glass_radius + 2.0;
                w.fill(g, u..u + 1, v..v + 1, 0..4, if lip { BRASS } else { STONE });
                continue;
            }
            let angle = p.y.atan2(p.x).rem_euclid(TAU);
            // Twelve petals around the center, then a ring of roundels set
            // between the spokes.
            let sector = (angle / spoke) as usize;
            let roundel_index = ((angle - spoke / 2.0) / spoke).round();
            let roundel_angle = roundel_index * spoke + spoke / 2.0;
            let roundel = (p - 38.5 * Vec2::from_angle(roundel_angle)).length();
            let glass = if r < 11.0 {
                E_GLASS_AMBER
            } else if r < 28.0 {
                [E_GLASS_RUBY, E_GLASS_VIOLET][sector % 2]
            } else if roundel < 5.0 {
                [E_GLASS_EMERALD, E_GLASS_AMBER][roundel_index as usize % 2]
            } else {
                E_GLASS_SAPPHIRE
            };
            w.fill(g, u..u + 1, v..v + 1, -2..0, glass);

            let off_spoke = ((angle + spoke / 2.0).rem_euclid(spoke) - spoke / 2.0) * r;
            let tracery = (11.0..14.0).contains(&r)
                || (28.0..31.0).contains(&r)
                || (r >= 11.0 && off_spoke.abs() < 1.2)
                || (r >= 31.0 && (roundel - 5.0).abs() < 0.9);
            if tracery {
                w.fill(g, u..u + 1, v..v + 1, 0..2, STONE);
            }
        }
    }
}

/// A dais under the rose window, up two steps with a light along each nosing,
/// and a candelabra on either side.
fn stage(g: &mut Grid) {
    let (x0, x1) = (96, ROOM.x - 96);
    let back = ROOM.z - SHELL;
    for (front, top) in [(428, 8), (436, 12), (444, 16)] {
        g.fill(ivec3(x0, FLOOR_TOP, front), ivec3(x1, top, back), STONE);
        g.fill(ivec3(x0, top - 1, front), ivec3(x1, top, back), WOOD);
        g.fill(
            ivec3(x0, top - 1, front),
            ivec3(x1, top, front + 1),
            E_STAGE,
        );
    }
    for x in [124, ROOM.x - 124] {
        candelabra(g, x, 16, 476);
    }
}

fn candelabra(g: &mut Grid, x: i32, base: i32, z: i32) {
    g.fill(
        ivec3(x - 3, base, z - 3),
        ivec3(x + 3, base + 2, z + 3),
        BRASS,
    );
    g.fill(
        ivec3(x - 1, base, z - 1),
        ivec3(x + 1, base + 48, z + 1),
        IRON,
    );
    let arm = base + 36;
    for (dx, dz) in [(10, 0), (-10, 0), (0, 10), (0, -10)] {
        let (ex, ez) = (x + dx, z + dz);
        // From the pole's 2×2 section to the candle's.
        g.fill(
            ivec3(x.min(ex) - 1, arm, z.min(ez) - 1),
            ivec3(x.max(ex) + 1, arm + 2, z.max(ez) + 1),
            BRASS,
        );
        candle(g, ex, arm + 2, ez, 5);
    }
    candle(g, x, base + 48, z, 7);
}

/// A candle centered on the corner at `(x, z)`, in a brass cup at `y`.
fn candle(g: &mut Grid, x: i32, y: i32, z: i32, wax: i32) {
    g.fill(ivec3(x - 2, y, z - 2), ivec3(x + 2, y + 1, z + 2), BRASS);
    g.fill(
        ivec3(x - 1, y + 1, z - 1),
        ivec3(x + 1, y + 1 + wax, z + 1),
        WAX,
    );
    g.fill(
        ivec3(x - 1, y + 1 + wax, z - 1),
        ivec3(x + 1, y + 4 + wax, z + 1),
        E_CANDLE,
    );
}

/// A round fountain in the middle of the room: glowing water in a stone basin,
/// a tiered column, and an orb on top. Bulbs ring the rim.
fn fountain(g: &mut Grid) {
    let (x, z) = (CENTER_X as f32, CENTER_Z as f32);
    g.cylinder(x, z, 0.0..36.0, FLOOR_TOP..10, STONE);
    g.cylinder(x, z, 0.0..36.0, 10..12, E_WATER);
    g.cylinder(x, z, 36.0..42.0, FLOOR_TOP..18, STONE);
    g.cylinder(x, z, 35.0..43.0, 18..19, BRASS);
    g.cylinder(x, z, 0.0..4.0, 12..56, STONE);
    g.cylinder(x, z, 0.0..12.0, 34..36, BRASS);
    g.cylinder(x, z, 10.5..12.0, 36..38, BRASS);
    g.cylinder(x, z, 4.0..10.5, 36..37, E_WATER);
    g.cylinder(x, z, 0.0..6.0, 56..59, BRASS);
    g.sphere(Vec3::new(x, 66.0, z), 7.0, E_ORB);
    for i in 0..12 {
        let p = Vec2::new(x, z) + 39.0 * Vec2::from_angle(TAU * i as f32 / 12.0);
        let (px, pz) = (p.x.round() as i32, p.y.round() as i32);
        g.fill(ivec3(px - 1, 19, pz - 1), ivec3(px + 1, 20, pz + 1), E_BULB);
    }
}

/// Light-up tiles in front of the stage, in six colors at low intensity.
fn dance_floor(g: &mut Grid) {
    const TILE: i32 = 16;
    let (min, max) = (ivec3(128, 0, 320), ivec3(ROOM.x - 128, 0, 416));
    for z in min.z..max.z {
        for x in min.x..max.x {
            let (lx, lz) = (x - min.x, z - min.z);
            let value = if lx % TILE == 0 || lz % TILE == 0 {
                FLOOR_DARK
            } else {
                let (tx, tz) = ((lx / TILE) as u32, (lz / TILE) as u32);
                let hash = tx.wrapping_mul(73856093) ^ tz.wrapping_mul(19349663);
                E_DANCE[(hash % 6) as usize]
            };
            g.set(ivec3(x, FLOOR_TOP - 1, z), value);
        }
    }
}

/// A round table with a cloth, and three candles of different heights.
fn table(g: &mut Grid, x: i32, z: i32) {
    let (fx, fz) = (x as f32, z as f32);
    g.cylinder(fx, fz, 0.0..6.0, FLOOR_TOP..6, IRON);
    g.cylinder(fx, fz, 0.0..2.5, FLOOR_TOP..20, IRON);
    g.cylinder(fx, fz, 0.0..13.0, 20..22, CLOTH);
    g.cylinder(fx, fz, 12.0..13.0, 14..20, CLOTH);
    candle(g, x - 4, 22, z + 2, 6);
    candle(g, x + 4, 22, z + 2, 4);
    candle(g, x, 22, z - 4, 8);
}

/// A three-tier brass chandelier hanging from the beams at `(x, z)`: bulbs on
/// each ring, glowing crystals hanging between them.
fn chandelier(g: &mut Grid, x: i32, z: i32) {
    let (fx, fz) = (x as f32, z as f32);
    g.fill(
        ivec3(x - 1, 112, z - 1),
        ivec3(x + 1, BEAM_BOTTOM, z + 1),
        IRON,
    );
    g.cylinder(fx, fz, 7.0..9.0, 112..113, BRASS);
    g.fill(ivec3(x - 2, 80, z - 2), ivec3(x + 2, 112, z + 2), BRASS);
    g.sphere(Vec3::new(fx, 77.0, fz), 3.5, E_CRYSTAL);

    for (tier, (radius, y, bulbs)) in [(16.0, 104, 8), (28.0, 96, 12), (40.0, 88, 18)]
        .into_iter()
        .enumerate()
    {
        g.cylinder(fx, fz, radius - 1.0..radius + 1.0, y..y + 1, BRASS);
        let arms = 6;
        for i in 0..arms {
            let dir = Vec2::from_angle(TAU * (i as f32 + 0.5 * tier as f32) / arms as f32);
            g.line(
                Vec3::new(fx + 2.0 * dir.x, (y + 8) as f32, fz + 2.0 * dir.y),
                Vec3::new(fx + radius * dir.x, y as f32 + 0.5, fz + radius * dir.y),
                BRASS,
            );
        }
        for i in 0..bulbs {
            let p = Vec2::new(fx, fz) + radius * Vec2::from_angle(TAU * i as f32 / bulbs as f32);
            let (px, pz) = (p.x.round() as i32, p.y.round() as i32);
            g.fill(
                ivec3(px - 1, y + 1, pz - 1),
                ivec3(px + 1, y + 2, pz + 1),
                BRASS,
            );
            g.fill(
                ivec3(px - 1, y + 2, pz - 1),
                ivec3(px + 1, y + 5, pz + 1),
                E_BULB,
            );

            let c = Vec2::new(fx, fz)
                + radius * Vec2::from_angle(TAU * (i as f32 + 0.5) / bulbs as f32);
            let length = 3 + (i % 3) * 2;
            let (cx, cz) = (c.x.floor() as i32, c.y.floor() as i32);
            g.fill(
                ivec3(cx, y - length, cz),
                ivec3(cx + 1, y, cz + 1),
                E_CRYSTAL,
            );
        }
    }
}

/// Strings of single-voxel lights across the room between the beams, sagging
/// in the middle: hundreds of tiny, bright emitters in seven colors.
fn string_lights(g: &mut Grid) {
    let half_span = (CENTER_X - SHELL) as f32;
    let height = |x: f32| {
        let t = (x - CENTER_X as f32) / half_span;
        128.0 - 12.0 * (1.0 - t * t)
    };
    for (row, z) in (BAY / 2..ROOM.z).step_by(BAY as usize).enumerate() {
        let mut x = SHELL as f32;
        while x < (ROOM.x - SHELL) as f32 {
            g.set(ivec3(x as i32, height(x) as i32, z), WIRE);
            x += 0.25;
        }
        for (i, x) in (12..ROOM.x - 12).step_by(8).enumerate() {
            let y = height(x as f32 + 0.5) as i32;
            g.set(ivec3(x, y - 1, z), E_FAIRY[(i + row) % E_FAIRY.len()]);
        }
    }
}

/// An iron post with a brass lantern on top, in a corner.
fn floor_lantern(g: &mut Grid, x: i32, z: i32) {
    g.fill(ivec3(x - 3, FLOOR_TOP, z - 3), ivec3(x + 3, 6, z + 3), IRON);
    g.fill(ivec3(x - 1, 6, z - 1), ivec3(x + 1, 44, z + 1), IRON);
    g.fill(ivec3(x - 5, 44, z - 5), ivec3(x + 5, 46, z + 5), BRASS);
    g.fill(ivec3(x - 5, 54, z - 5), ivec3(x + 5, 56, z + 5), BRASS);
    g.fill(ivec3(x - 3, 56, z - 3), ivec3(x + 3, 58, z + 3), BRASS);
    for (cx, cz) in [
        (x - 5, z - 5),
        (x + 3, z - 5),
        (x - 5, z + 3),
        (x + 3, z + 3),
    ] {
        g.fill(ivec3(cx, 46, cz), ivec3(cx + 2, 54, cz + 2), BRASS);
    }
    g.fill(ivec3(x - 2, 46, z - 2), ivec3(x + 2, 53, z + 2), E_LANTERN);
}

/// The wisp: a glowing core in two brass rings.
fn wisp_voxel(p: UVec3) -> u8 {
    let q = p.as_vec3() + 0.5 - WISP_SIZE as f32 / 2.0;
    let ring = |radial: f32, axial: f32| (radial - 6.5).abs() < 0.75 && axial.abs() < 0.75;
    if q.length() < 3.5 {
        E_WISP
    } else if ring(Vec2::new(q.x, q.z).length(), q.y) || ring(Vec2::new(q.x, q.y).length(), q.z) {
        BRASS
    } else {
        0
    }
}

/// Builds a model `extent` voxels in size from `voxel`, which gives the palette
/// value at each, or `None` if every voxel is empty.
fn build_model(
    allocator: &Allocator,
    extent: UVec3,
    voxel: impl Fn(UVec3) -> u8,
) -> Option<(VoxGeometry, VoxMaterial)> {
    let mut geometry = VoxGeometry::new(allocator.clone(), UNIT);
    let mut material = VoxMaterial::new(allocator.clone());
    let mut accessor = geometry.tree.accessor_mut(&mut material);
    let mut empty = true;
    // Leaf by leaf (4³), so consecutive writes land in the accessor's hot leaf.
    for bz in (0..extent.z).step_by(4) {
        for by in (0..extent.y).step_by(4) {
            for bx in (0..extent.x).step_by(4) {
                for z in bz..(bz + 4).min(extent.z) {
                    for y in by..(by + 4).min(extent.y) {
                        for x in bx..(bx + 4).min(extent.x) {
                            let p = UVec3::new(x, y, z);
                            let value = voxel(p);
                            if value != 0 {
                                accessor.set(p, value);
                                empty = false;
                            }
                        }
                    }
                }
            }
        }
    }
    drop(accessor);
    (!empty).then_some((geometry, material))
}

/// The room voxel coordinates of each chunk's minimum corner: the room is split
/// into models of at most `CHUNK`³ voxels.
fn chunk_origins() -> Vec<IVec3> {
    let mut origins = Vec::new();
    for z in (0..ROOM.z).step_by(CHUNK as usize) {
        for y in (0..ROOM.y).step_by(CHUNK as usize) {
            for x in (0..ROOM.x).step_by(CHUNK as usize) {
                origins.push(ivec3(x, y, z));
            }
        }
    }
    origins
}

/// World position of a point in room voxel coordinates. The room is centered on
/// the origin horizontally, with its floor at y = 0.
fn room_to_world(p: Vec3) -> Vec3 {
    (p - Vec3::new(CENTER_X as f32, 0.0, CENTER_Z as f32)) * UNIT
}

/// Moves the wisp around the room.
#[derive(Component)]
struct Wisp;

fn setup_emissive_room(
    mut commands: Commands,
    allocator: Res<Allocator>,
    mut geometries: ResMut<Assets<VoxGeometry>>,
    mut materials: ResMut<Assets<VoxMaterial>>,
    mut palettes: ResMut<Assets<VoxPalette>>,
    camera: Single<(Entity, &mut dust_pbr::camera::Camera)>,
) {
    let started = Instant::now();
    let grid = build_room();
    let palette = palettes.add(palette(allocator.clone()).expect("emissive room palette"));

    // One model per chunk, built in parallel.
    let origins = chunk_origins();
    let chunks: Vec<_> = std::thread::scope(|scope| {
        let grid = &grid;
        let allocator = &*allocator;
        let builds: Vec<_> = origins
            .iter()
            .map(|&origin| {
                let extent = (ROOM - origin).min(IVec3::splat(CHUNK)).as_uvec3();
                scope.spawn(move || {
                    let model = build_model(allocator, extent, |p| grid.get(origin + p.as_ivec3()));
                    (origin, model)
                })
            })
            .collect();
        builds
            .into_iter()
            .map(|build| build.join().unwrap())
            .collect()
    });

    // The chunks' colliders attach to this one static body, as the castle's do
    // to its scene root. The body can't go on the instances themselves: their
    // colliders already have a `Position` by then, and avian writes a new body's
    // `Position` over its `Transform`, which would move every chunk to the
    // origin.
    let room = commands
        .spawn((
            Name::new("Emissive room"),
            Transform::default(),
            RigidBody::Static,
        ))
        .id();
    for (origin, model) in chunks {
        let Some((mut geometry, material)) = model else {
            continue;
        };
        let collider = geometry.collider();
        let geometry = geometries.add(geometry);
        let material = materials.add(material);
        let model = commands
            .spawn_scene(bsn! {
                Name({format!("Emissive room model {origin}")})
                VoxModel {
                    geometry: {geometry},
                    material: {material},
                    palette: {palette.clone()},
                }
                ~{ VoxModelCollider(collider) }
            })
            .id();
        let translation = room_to_world(origin.as_vec3());
        commands
            .spawn_scene(bsn! {
                Name({format!("Emissive room {origin}")})
                Transform::from_translation(translation)
                VoxInstance {
                    model: model
                }
            })
            .insert(ChildOf(room));
    }

    let (geometry, material) =
        build_model(&allocator, UVec3::splat(WISP_SIZE), wisp_voxel).expect("the wisp has voxels");
    let geometry = geometries.add(geometry);
    let material = materials.add(material);
    let model = commands
        .spawn_scene(bsn! {
            Name("Wisp model")
            VoxModel {
                geometry: {geometry},
                material: {material},
                palette: {palette.clone()},
            }
        })
        .id();
    commands
        .spawn_scene(bsn! {
            Name("Wisp")
            VoxInstance {
                model: model
            }
        })
        .insert(Wisp);

    info!("Built the emissive room in {:.0?}", started.elapsed());

    // The player, just inside the door, facing the rose window.
    let character = Character {
        yaw: PI,
        ..default()
    };
    let collider = character.collider();
    let spawn = room_to_world(Vec3::new(CENTER_X as f32, 20.0, 40.0));
    let character = commands
        .spawn((character, collider, Transform::from_translation(spawn)))
        .observe(super::on_view_changed)
        .id();
    let (camera, mut camera_settings) = camera.into_inner();
    commands.entity(camera).insert(CharacterCamera(character));
    // Auto-exposure meters the room to mid-gray; hold it a stop and a half
    // under so it stays a dark room with bright lights in it.
    camera_settings.exposure = -1.5;
}

/// Circles the wisp around the fountain, bobbing and turning.
fn move_wisp(time: Res<Time>, mut wisps: Query<&mut Transform, With<Wisp>>) {
    let t = time.elapsed_secs();
    let angle = t * TAU / 24.0;
    let center = Vec3::new(
        7.0 * angle.cos(),
        3.5 + 0.4 * (t * 1.3).sin(),
        7.0 * angle.sin(),
    );
    let half_extent = Vec3::splat(WISP_SIZE as f32 * UNIT / 2.0);
    for mut transform in &mut wisps {
        transform.rotation = Quat::from_rotation_y(t * 0.8) * Quat::from_rotation_x(0.4);
        transform.translation = center - transform.rotation * half_extent;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every voxel on the outside of the room is solid, so no sky gets in.
    #[test]
    fn room_is_closed() {
        let grid = build_room();
        let max = ROOM - 1;
        for z in 0..ROOM.z {
            for y in 0..ROOM.y {
                for x in 0..ROOM.x {
                    let p = ivec3(x, y, z);
                    if (p.cmpeq(IVec3::ZERO) | p.cmpeq(max)).any() {
                        assert_ne!(grid.get(p), 0, "hole in the shell at {p}");
                    }
                }
            }
        }
    }

    /// Every value the room and the wisp use has a palette entry; a value
    /// without one would render black.
    #[test]
    fn palette_covers_room() {
        let n = WISP_SIZE;
        let wisp = (0..n * n * n).map(|i| wisp_voxel(UVec3::new(i % n, i / n % n, i / n / n)));
        let mut counts = [0usize; 256];
        for value in build_room().cells.into_iter().chain(wisp) {
            counts[value as usize] += 1;
        }
        for value in 1..=255u8 {
            if counts[value as usize] > 0 {
                assert!(
                    MATERIALS.iter().any(|&(v, ..)| v == value),
                    "value {value} has no palette entry"
                );
            }
        }
        let emitters: usize = MATERIALS
            .iter()
            .filter(|(.., surface)| matches!(surface, Surface::Emit(_)))
            .map(|&(value, ..)| counts[value as usize])
            .sum();
        println!("{emitters} emissive voxels");
        assert!(emitters > 0);
    }

    /// The chunks' emitter tables hold every emissive face the room exposes.
    /// The only faces they add are on a chunk's boundary, where a chunk can't
    /// see that a voxel of the next chunk covers them.
    #[test]
    fn emitter_tables_cover_the_room() {
        use std::collections::HashSet;

        use dust_pbr::emitters::{EmitterTable, LUMINANCE};
        use dust_vox::emitters::emissive_faces;

        let grid = build_room();
        let entries = palette_entries();
        let emits = |value: u8| value != 0 && entries[value as usize - 1].emission > 0.0;
        let directions = [
            IVec3::X,
            IVec3::NEG_X,
            IVec3::Y,
            IVec3::NEG_Y,
            IVec3::Z,
            IVec3::NEG_Z,
        ];

        // The faces the whole room exposes, by room coordinate and direction.
        let mut emitters = Vec::new();
        let mut exposed = HashSet::new();
        for z in 0..ROOM.z {
            for y in 0..ROOM.y {
                for x in 0..ROOM.x {
                    let p = ivec3(x, y, z);
                    if !emits(grid.get(p)) {
                        continue;
                    }
                    emitters.push(p);
                    for (direction, step) in directions.iter().enumerate() {
                        let q = p + step;
                        if Grid::index(q).is_some() && grid.get(q) == 0 {
                            exposed.insert((p, direction as u8));
                        }
                    }
                }
            }
        }

        let mut found = HashSet::new();
        let (mut table_power, mut face_power) = (0.0f64, 0.0f64);
        for origin in chunk_origins() {
            // The chunk's model, as `build_model` makes it: the chunk's voxels,
            // in a tree `CHUNK` voxels on a side.
            let extent = (ROOM - origin).min(IVec3::splat(CHUNK));
            let inside = |p: IVec3| p.cmpge(IVec3::ZERO).all() && p.cmplt(extent).all();
            let faces = emissive_faces(
                emitters
                    .iter()
                    .map(|&p| p - origin)
                    .filter(|&p| inside(p))
                    .map(|p| p.as_uvec3()),
                |p| {
                    let p = p.as_ivec3();
                    let value = if inside(p) { grid.get(origin + p) } else { 0 };
                    (value != 0).then(|| value - 1)
                },
                &entries,
                UVec3::splat(CHUNK as u32),
            );
            let table = EmitterTable::build(UNIT, &faces);
            assert_eq!(table.faces.len(), faces.len(), "an emitter without luminance");
            table_power += table.header.power as f64;
            for face in &faces {
                face_power += (face.radiance.dot(LUMINANCE) * UNIT * UNIT) as f64;
                let p = origin + face.coords.as_ivec3();
                if !exposed.contains(&(p, face.direction)) {
                    let across = face.coords.as_ivec3() + directions[face.direction as usize];
                    assert!(!inside(across), "face {} of {p} is covered in its chunk", face.direction);
                }
                found.insert((p, face.direction));
            }
        }
        let missing = exposed.difference(&found).count();
        assert_eq!(missing, 0, "exposed faces missing from the tables");
        assert!((table_power - face_power).abs() < 1e-3 * face_power);
        println!(
            "{} faces exposed in the room, {} in the chunks' tables",
            exposed.len(),
            found.len()
        );
    }
}
