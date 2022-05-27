glslc primary.rgen --target-env=vulkan1.3 -O -fshader-stage=rgen -o primary.rgen.spv
glslc dda.rint --target-env=vulkan1.3 -O -fshader-stage=rint -o dda.rint.spv
glslc plain.rchit --target-env=vulkan1.3 -O -fshader-stage=rchit -o plain.rchit.spv
glslc sky.rmiss --target-env=vulkan1.3 -O -fshader-stage=rmiss -o sky.rmiss.spv
