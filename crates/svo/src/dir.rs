//          Cell Corners
//
//       2-------------------6
//      /|                  /|
//     / |      0          / |
//    /  |                /  |
//   3-------------------7   |
//   |   |         5     |   |
//   |   |               |   |
//   | 2 |               | 3 |
//   |   |     4         |   |
//   |   0---------------|---4
//   |  /                |  /
//   | /         1       | /
//   |/                  |/
//   1-------------------5
//

//      +y
//       |
//       |
//       --------- +x
//      /
//     /
//   -z

#[repr(u8)]
#[derive(PartialEq, Eq, Copy, Clone)]
pub enum Quadrant {
    LeftBottom = 0b00,
    RightBottom = 0b10,
    LeftTop = 0b01,
    RightTop = 0b11,
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Corner {
    RearLeftBottom = 0b000,
    FrontLeftBottom = 0b001,
    RearLeftTop = 0b010,
    FrontLeftTop = 0b011,
    RearRightBottom = 0b100,
    FrontRightBottom = 0b101,
    RearRightTop = 0b110,
    FrontRightTop = 0b111,
}

impl Corner {
    #[inline]
    pub fn is_on_face(&self, face: Face) -> bool {
        let n = *self as u8;
        match face {
            Face::Front => n & 0b001 != 0,
            Face::Rear => n & 0b001 == 0,
            Face::Left => n & 0b100 == 0,
            Face::Right => n & 0b100 != 0,
            Face::Top => n & 0b010 != 0,
            Face::Bottom => n & 0b010 == 0,
        }
    }
    #[inline]
    pub fn opposite(&self) -> Self {
        (7 - *self as u8).into()
    }

    #[inline]
    pub fn position_offset(&self) -> (u8, u8, u8) {
        // instead of using a "match" statement, this reduces the overall build time by 10%
        let a = *self as u8;
        let x = a >> 2;
        let y = (a & 0b010) >> 1;
        let z = (!a) & 0b1;
        (x, y, z)
    }

    pub fn all() -> AllDirectionIterator {
        AllDirectionIterator { current: 0 }
    }

    // Internal, Face, Quadrant
    pub fn subdivided_surfaces(&self) -> [(bool, Face, Quadrant); 6] {
        use Corner::*;
        use Face::*;
        use Quadrant::*;
        match self {
            RearLeftBottom => [
                (true, Bottom, LeftTop),
                (false, Bottom, LeftTop),
                (false, Left, RightBottom),
                (true, Left, RightBottom),
                (true, Rear, LeftBottom),
                (false, Rear, LeftBottom),
            ],
            FrontLeftBottom => [
                (true, Bottom, LeftBottom),
                (false, Bottom, LeftBottom),
                (false, Left, LeftBottom),
                (true, Left, LeftBottom),
                (false, Front, LeftBottom),
                (true, Front, LeftBottom),
            ],
            RearLeftTop => [
                (false, Top, LeftTop),
                (true, Top, LeftTop),
                (false, Left, RightTop),
                (true, Left, RightTop),
                (true, Rear, LeftTop),
                (false, Rear, LeftTop),
            ],
            FrontLeftTop => [
                (false, Top, LeftBottom),
                (true, Top, LeftBottom),
                (false, Left, LeftTop),
                (true, Left, LeftTop),
                (false, Front, LeftTop),
                (true, Front, LeftTop),
            ],
            RearRightBottom => [
                (true, Bottom, RightTop),
                (false, Bottom, RightTop),
                (true, Right, RightBottom),
                (false, Right, RightBottom),
                (true, Rear, RightBottom),
                (false, Rear, RightBottom),
            ],
            FrontRightBottom => [
                (true, Bottom, RightBottom),
                (false, Bottom, RightBottom),
                (true, Right, LeftBottom),
                (false, Right, LeftBottom),
                (false, Front, RightBottom),
                (true, Front, RightBottom),
            ],
            RearRightTop => [
                (false, Top, RightTop),
                (true, Top, RightTop),
                (true, Right, RightTop),
                (false, Right, RightTop),
                (true, Rear, RightTop),
                (false, Rear, RightTop),
            ],
            FrontRightTop => [
                (false, Top, RightBottom),
                (true, Top, RightBottom),
                (true, Right, LeftTop),
                (false, Right, LeftTop),
                (false, Front, RightTop),
                (true, Front, RightTop),
            ],
        }
    }
}

pub struct AllDirectionIterator {
    current: u8,
}

impl Iterator for AllDirectionIterator {
    type Item = Corner;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= 8 {
            None
        } else {
            let item: Corner = self.current.into();
            self.current += 1;
            Some(item)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (8, Some(8))
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        8
    }
}
impl From<u8> for Corner {
    fn from(num: u8) -> Self {
        assert!(num < 8);
        unsafe { std::mem::transmute(num) }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Edge {
    LowerFar = 0,
    LowerRight = 1,
    LowerNear = 2,
    LowerLeft = 3,

    UpperFar = 4,
    UpperRight = 5,
    UpperNear = 6,
    UpperLeft = 7,

    VerticalRearLeft = 8,
    VerticalRearRight = 9,
    VerticalFrontRight = 10,
    VerticalFrontLeft = 11,
}
impl From<u8> for Edge {
    fn from(val: u8) -> Self {
        assert!(val < 12);
        unsafe { std::mem::transmute(val) }
    }
}

impl Edge {
    pub fn vertices(&self) -> (Corner, Corner) {
        match self {
            Edge::LowerFar => (Corner::RearLeftBottom, Corner::RearRightBottom),
            Edge::LowerRight => (Corner::RearRightBottom, Corner::FrontRightBottom),
            Edge::LowerNear => (Corner::FrontLeftBottom, Corner::FrontRightBottom),
            Edge::LowerLeft => (Corner::RearLeftBottom, Corner::FrontLeftBottom),

            Edge::UpperFar => (Corner::RearLeftTop, Corner::RearRightTop),
            Edge::UpperRight => (Corner::RearRightTop, Corner::FrontRightTop),
            Edge::UpperNear => (Corner::FrontLeftTop, Corner::FrontRightTop),
            Edge::UpperLeft => (Corner::RearLeftTop, Corner::FrontLeftTop),

            Edge::VerticalRearLeft => (Corner::RearLeftBottom, Corner::RearLeftTop),
            Edge::VerticalRearRight => (Corner::RearRightBottom, Corner::RearRightTop),
            Edge::VerticalFrontRight => (Corner::FrontRightBottom, Corner::FrontRightTop),
            Edge::VerticalFrontLeft => (Corner::FrontLeftBottom, Corner::FrontLeftTop),
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Face {
    Top = 0,
    Bottom = 1,
    Left = 2,
    Right = 3,
    Front = 4,
    Rear = 5,
}

impl From<u8> for Face {
    fn from(val: u8) -> Self {
        assert!(val < 6);
        unsafe { std::mem::transmute(val) }
    }
}

impl Face {
    pub fn vertices(&self) -> [[Corner; 2]; 2] {
        use Corner::*;
        use Face::*;
        match self {
            Front => [
                [FrontLeftBottom, FrontLeftTop],
                [FrontRightBottom, FrontRightTop],
            ],
            Rear => [
                [RearLeftBottom, RearLeftTop],
                [RearRightBottom, RearRightTop],
            ],
            Left => [
                [RearLeftBottom, FrontLeftBottom],
                [RearLeftTop, FrontLeftTop],
            ],
            Right => [
                [RearRightBottom, FrontRightBottom],
                [RearRightTop, FrontRightTop],
            ],
            Top => [[RearLeftTop, FrontLeftTop], [RearRightTop, FrontRightTop]],
            Bottom => [
                [RearLeftBottom, FrontLeftBottom],
                [RearRightBottom, FrontRightBottom],
            ],
        }
    }
}
