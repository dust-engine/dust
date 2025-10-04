.PHONY: default
default: pbr.spv

pbr.spv: pbr.slang
	slangc -O3 -target spirv -o pbr.spv pbr.slang
