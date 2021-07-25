use crate::octree::Node;
use super::super::Octree;
use crate::alloc::Handle;

use crate::Voxel;
/*
fn set_recursive<T: Voxel>(
    octree: &mut Octree<T>,
    handle: Handle,
    mut x: u32,
    mut y: u32,
    mut z: u32,
    mut gridsize: u32,
    occupancy: bool,
) -> (bool, u8, bool) {
    gridsize = gridsize / 2;
    let mut corner: u8 = 0;
    if x >= gridsize {
        corner |= 0b100;
        x -= gridsize;
    }
    if y >= gridsize {
        corner |= 0b010;
        y -= gridsize;
    }
    if z >= gridsize {
        corner |= 0b001;
        z -= gridsize;
    }
    if gridsize <= 1 {
        // is leaf node
        let node_ref = unsafe { octree.arena.get_mut(handle).node };
        if !occupancy {
            node_ref.occupancy &= !(1 << corner);
        } else {
            node_ref.occupancy |= 1 << corner;
        }
        if node_ref.sizemask & (1 << (2 * corner)) != 0 {
            // has children. Cut them off.
            todo!()
        }
    } else {
        let node_ref = unsafe { octree.arena.get_mut(handle).node };
        let sizemask = node_ref.sizemask;
        if sizemask & (1 << (2 * corner)) == 0 {
            // no children
            octree.reshape(handle, sizemask | (1 << (2 * corner)));
        }

        let new_handle = unsafe { octree.arena.get(handle).node }.child_handle(corner.into());
        let (avg, all, collapsed) = set_recursive(octree, new_handle, x, y, z, gridsize, occupancy);

        let node_ref = unsafe { octree.arena.get_mut(handle).node };
        let sizemask = node_ref.sizemask;
        if !avg {
            node_ref.occupancy &= !(1 << corner);
        } else {
            node_ref.occupancy |= 1 << corner;
        }
        node_ref.extended_occupancy[corner as usize] = all;
        if collapsed {
            octree.reshape(handle, sizemask & !(1 << (2 * corner)));
        }
    }

    let node_ref = unsafe { octree.arena.get_mut(handle).node };
    if node_ref.sizemask == 0 {
        // node has no children
        if (node_ref.occupancy == 255 || node_ref.occupancy == 1)
            && node_ref.occupancy >> 7 == occupancy as u8
        {
            // collapse node
            return (occupancy, node_ref.occupancy, true);
        }
    }
    let avg = node_ref.occupancy != 0;
    let all = node_ref.occupancy;
    octree.arena.changed(handle);
    return (avg, all, false);
}
*/

fn set_recursive<T: Voxel>(
    octree: &mut Octree<T>,
    mut current: (Node<T>, Option<[u8; 8]>),
    mut x: u32,
    mut y: u32,
    mut z: u32,
    mut gridsize: u32,
    value: bool,
) -> Result<(Node<T>, Option<[u8; 8]>), bool> {
    gridsize = gridsize / 2;
    let mut corner: u8 = 0;
    if x >= gridsize {
        corner |= 0b100;
        x -= gridsize;
    }
    if y >= gridsize {
        corner |= 0b010;
        y -= gridsize;
    }
    if z >= gridsize {
        corner |= 0b001;
        z -= gridsize;
    }
    if gridsize <= 1 {
        // setting this node
        if value {
            current.0.occupancy |= 1 << corner;
        } else {
            current.0.occupancy &= !(1 << corner);
        }
        // TODO: if have children cut them off
    } else {
        let child_has_extended_occupancy = (current.0.sizemask & (3 << (2 * corner))) == 3; // 11
        let result = {
            let has_child = current.0.sizemask & (1 << (2 * corner)) != 0;
            let child_handle = current.0.child_handle(corner.into());
            let child_node = if has_child {
                unsafe { octree.arena.get(child_handle).node }
            } else {
                // TODO: THIS IS WRONG IN CASES WHERE WRITING AIR
                Default::default()
            };
            let child_extended_occupancy = if child_has_extended_occupancy {
                Some(unsafe { octree.arena.get(child_handle.offset(1)).extended_occupancy })
            } else {
                None
            };
            set_recursive(octree, (child_node, child_extended_occupancy), x, y, z, gridsize, value)
        };
        match result {
            Ok(child) => {
                if current.1.is_none() {
                    // TODO: Should be something else, in this scenario its fine though.
                    current.1 = Some([0; 8]);
                }
                // is not leaf
                if current.0.sizemask & (1 << (2 * corner)) == 0 {
                    // missing child
                    // println!("Missing child: Reshaping");
                    let val = current.0.sizemask | (1 << (2 * corner));
                    octree.reshape(&mut current.0, val);
                }
                // has eo but doesn't want it, vice versa
                if !child_has_extended_occupancy {
                    if let Some(_child_extended_occupancy) = child.1 {
                        // println!("Missing occupancy: Reshaping");
                        let val = current.0.sizemask | (2 << (2 * corner));
                        octree.reshape(&mut current.0, val);
                    }
                }
                if child_has_extended_occupancy && child.1.is_none() {
                    // println!("unncessary occupancy: reshaping");
                    let val = current.0.sizemask & !(2 << (2 * corner));
                    octree.reshape(&mut current.0, val);
                }
                let child_handle = current.0.child_handle(corner.into());
                octree.arena.get_mut(child_handle).node = child.0;
                // set extended occupancy
                if let Some(child_extended_occupancy) = child.1 {
                    octree.arena.get_mut(child_handle.offset(1)).extended_occupancy = child_extended_occupancy;
                }
                // Setting occupancy of parent
                if child.0.occupancy == 0 {
                    current.0.occupancy &= !(1 << corner);
                } else {
                    current.0.occupancy |= 1 << corner;
                }
                current.1.as_mut().unwrap()[corner as usize] = child.0.occupancy;
            },
            Err(value) => {
                // is leaf
                
                // remove child
                let val = current.0.sizemask & !(3 << (2 * corner));
                octree.reshape(&mut current.0, val);
                if value {
                    current.0.occupancy |= 1 << corner;
                } else {    
                    current.0.occupancy &= !(1 << corner);
                }
            },
        }
    }
    // Turn into leaf if occupancy all equal
    if current.0.sizemask == 0 {
        // node has no children
        if (current.0.occupancy == 255 || current.0.occupancy == 0)
            // Figure out why this is necessary
            && current.0.occupancy >> 7 == value as u8
        {
            // collapse node
            return Err(value);
        }
    }

    // println!("Current: {:?}", current);

    Ok(current)
}


pub fn get<T: Voxel>(
    octree: &Octree<T>,
    mut x: u32,
    mut y: u32,
    mut z: u32,
    mut gridsize: u32,
) -> bool {
    let mut handle = octree.root;
    while gridsize > 2 {
        gridsize = gridsize / 2;
        let mut corner: u8 = 0;
        if x >= gridsize {
            corner |= 0b100;
            x -= gridsize;
        }
        if y >= gridsize {
            corner |= 0b010;
            y -= gridsize;
        }
        if z >= gridsize {
            corner |= 0b001;
            z -= gridsize;
        }
        
        let node_ref = unsafe { octree.arena.get(handle).node };
        println!("Getting Node: {:?}", node_ref);
        println!("Occupancy: {:#b}", node_ref.occupancy);
        println!("Sizemask: {:#b}", node_ref.sizemask);
        println!("Extended occupancy: {:?}", unsafe { octree.arena.get(handle.offset(1)).extended_occupancy });
        println!("Child: {:?}", unsafe { octree.arena.get(node_ref.children).node });
        if node_ref.sizemask & (1 << (2 * corner)) == 0 {
            return node_ref.occupancy & (1 << corner) != 0;
        }
        if node_ref.sizemask & (2 << (2 * corner)) != 0 {
            println!("Next has extended occupancy");
        }

        handle = node_ref.child_handle(corner.into());
    }
    // gridsize is now equal to 2
    debug_assert_eq!(gridsize, 2);
    let mut corner: u8 = 0;
    if x >= 1 {
        corner |= 0b100;
    }
    if y >= 1 {
        corner |= 0b010;
    }
    if z >= 1 {
        corner |= 0b001;
    }
    unsafe { octree.arena.get(handle).node }.occupancy & (1 << corner) != 0
}

pub struct RandomAccessor<'a, T: Voxel> {
    pub octree: &'a Octree<T>,
}

impl<'a, T: Voxel> RandomAccessor<'a, T> {
    pub fn get(&self, x: u32, y: u32, z: u32, gridsize: u32) -> bool {
        get(self.octree, x, y, z, gridsize)
    }
}

pub struct RandomMutator<'a, T: Voxel> {
    pub octree: &'a mut Octree<T>,
}

impl<'a, T: Voxel> RandomMutator<'a, T> {
    pub fn get(&self, x: u32, y: u32, z: u32, gridsize: u32) -> bool {
        get(self.octree, x, y, z, gridsize)
    }
    pub fn set(&mut self, x: u32, y: u32, z: u32, gridsize: u32, item: bool) {
        let root_node = unsafe { self.octree.arena.get(self.octree.root).node };
        let root_extended_occupancy = if root_node.sizemask != 0 {
            Some(unsafe { self.octree.arena.get(self.octree.root.offset(1)).extended_occupancy })
        } else {
            None
        };
        let result = set_recursive(self.octree, (root_node, root_extended_occupancy), x, y, z, gridsize, item);
        match result {
            Ok(child) => {
                let child_handle = self.octree.root;
                self.octree.arena.get_mut(child_handle).node = child.0;
                // set extended occupancy
                if let Some(child_extended_occupancy) = child.1 {
                    self.octree.arena.get_mut(child_handle.offset(1)).extended_occupancy = child_extended_occupancy;
                }
            },
            Err(value) => {
                panic!("Invalid state");
            }
        }
        // println!("{:?}", self.octree.root);
        // self.octree.root_occupancy = data;
        // println!("End setting");
    }
}

impl<T: Voxel> Octree<T> {
    pub fn get_random_accessor(&self) -> RandomAccessor<T> {
        RandomAccessor { octree: self }
    }
    pub fn get_random_mutator(&mut self) -> RandomMutator<T> {
        RandomMutator { octree: self }
    }
}
