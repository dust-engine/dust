/**
A supertree is an octree of octrees.

Each octree has a dedicated ArenaAllocator backed by the same Arc<dyn BlockAllocator>.
The memory used by a single octree is therefore
A single octree is the minimal unit when doing LOD

*/
pub struct Supertree {
}
