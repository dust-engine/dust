.PHONY: default
default: blas_builder_copy_coords.spv

blas_builder_copy_coords.spv: blas_builder_copy_coords.slang
	slangc -O3 -matrix-layout-row-major -target spirv -o blas_builder_copy_coords.spv blas_builder_copy_coords.slang

pbr.spv: pbr.slang
	slangc -O3 -matrix-layout-row-major -target spirv -o pbr.spv pbr.slang
