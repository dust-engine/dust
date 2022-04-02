use bevy_asset::HandleUntyped;
use bevy_utils::HashMap;

pub struct AccelerationStructureStore {
    accel_structs: HashMap<HandleUntyped, dustash::accel_struct::AccelerationStructure>,
    pending_accel_structs: HashMap<HandleUntyped, dustash::accel_struct::AccelerationStructure>,
}

fn build_tlas_system(

) {
    // how do you keep track of / release those BLAS used? can't increment thousands of Arcs each frmae.
}