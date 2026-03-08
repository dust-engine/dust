.PHONY: default
default: pbr.spv

pbr.spv: pbr.slang
	slangc -O3 -matrix-layout-row-major -target spirv -o pbr.spv pbr.slang
