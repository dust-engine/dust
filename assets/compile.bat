set SHADERS[1]=auto_exposure_avg.comp
set SHADERS[2]=auto_exposure.comp
set SHADERS[3]=final_gather.rchit
set SHADERS[4]=final_gather.rgen
set SHADERS[5]=final_gather.rmiss
set SHADERS[6]=hit.rchit
set SHADERS[7]=hit.rint
set SHADERS[8]=miss.rmiss
set SHADERS[9]=photon.rchit
set SHADERS[10]=photon.rgen
set SHADERS[11]=primary.rgen
set SHADERS[12]=shadow.rgen
set SHADERS[13]=shadow.rmiss
set SHADERS[0]=tone_map.comp

(for /L %%i in (0, 1, 14) do (call glslc %%SHADERS[%%i]%% --target-env=vulkan1.3 -O -g -o %%SHADERS[%%i]%%.spv))

