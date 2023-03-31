FLAGS="--target-env=vulkan1.3 -O"
glslc primary.rgen ${FLAGS} -fshader-stage=rgen -o primary.rgen.spv
