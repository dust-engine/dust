FLAGS="--target-env=vulkan1.3 -O -g"
glslc primary.rgen ${FLAGS} -fshader-stage=rgen -o primary.rgen.spv
glslc hit.rchit ${FLAGS} -fshader-stage=rchit -o hit.rchit.spv
glslc miss.rmiss ${FLAGS} -fshader-stage=rmiss -o miss.rmiss.spv
glslc hit.rint ${FLAGS} -fshader-stage=rint -o hit.rint.spv


glslc photon.rgen ${FLAGS} -fshader-stage=rgen -o photon.rgen.spv
glslc photon.rchit ${FLAGS} -fshader-stage=rchit -o photon.rchit.spv
glslc final_gather.rchit ${FLAGS} -fshader-stage=rchit -o final_gather.rchit.spv
glslc final_gather.rgen ${FLAGS} -fshader-stage=rgen -o final_gather.rgen.spv
glslc final_gather.rmiss ${FLAGS} -fshader-stage=rmiss -o final_gather.rmiss.spv
glslc shadow.rmiss ${FLAGS} -fshader-stage=rmiss -o shadow.rmiss.spv
glslc shadow.rgen ${FLAGS} -fshader-stage=rgen -o shadow.rgen.spv
glslc auto_exposure.comp ${FLAGS} -fshader-stage=comp -o auto_exposure.comp.spv
glslc auto_exposure_avg.comp ${FLAGS} -fshader-stage=comp -o auto_exposure_avg.comp.spv
glslc tone_map.comp ${FLAGS} -fshader-stage=comp -o tone_map.comp.spv
glslc asvgf/temporal.comp ${FLAGS} -fshader-stage=comp -o asvgf/temporal.comp.spv
