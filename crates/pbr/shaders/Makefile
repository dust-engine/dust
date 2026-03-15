.PHONY: default
default: pbr.spv tonemap.spv

pbr.spv: pbr.slang
	slangc -O3 -matrix-layout-row-major -target spirv -o pbr.spv pbr.slang

tonemap.spv: tonemap.slang
	slangc -O3 -target spirv -o tonemap.spv tonemap.slang
