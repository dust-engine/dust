FLAGS="--target-env=vulkan1.3 -O"
glslc primary.rgen ${FLAGS} -fshader-stage=rgen -o primary.rgen.spv
glslc hit.rchit ${FLAGS} -fshader-stage=rchit -o hit.rchit.spv
glslc miss.rmiss ${FLAGS} -fshader-stage=rmiss -o miss.rmiss.spv
glslc hit.rint ${FLAGS} -fshader-stage=rint -o hit.rint.spv
