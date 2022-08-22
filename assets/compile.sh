FLAGS="--target-env=vulkan1.3 -O"
glslc primary.rgen ${FLAGS} -fshader-stage=rgen -o primary.rgen.spv
glslc dda.rint ${FLAGS} -fshader-stage=rint -o dda.rint.spv
glslc plain.rchit ${FLAGS} -fshader-stage=rchit -o plain.rchit.spv
glslc sky.rmiss ${FLAGS} -fshader-stage=rmiss -o sky.rmiss.spv
