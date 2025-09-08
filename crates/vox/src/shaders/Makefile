.PHONY: default
default: blas_builder_copy_coords.spv

blas_builder_copy_coords.spv: blas_builder_copy_coords.slang
	slangc -O3 -target spirv -o blas_builder_copy_coords.spv blas_builder_copy_coords.slang
