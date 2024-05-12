Loading and displaying MagicaVoxel assets.
Each MagicaVoxel model will be imported as a separate model.

ECS Models:
Model::
- VoxModel
- Handle<Material>
- Handle<Geometry>
- Handle<Palette>
- BLAS built for this(Automatically added)
- SbtIndex(Automatically added)

Instances:
- VoxInstance(Entity: pointing to a Model)
- Transform
- GlobalTransform

Another formulation for GLTF triangles:
Nodes:
- GlobalTransform
- Transform
- Mesh(Entity: pointing to a Mesh)

Mesh: Children<Primitive>. BLAS built for this.

Primitive:
- IndexBuffer { Handle<Buffer>, stride }
- VertexBuffer { Handle<Buffer>, stride }
- Handle<Material>
