.PHONY: default clean

SLANGC = slangc
SLANGFLAGS = -O3 -target spirv

SHADERS = transmittance_lut.spv multi_scattering.spv sky_view_lut.spv sky_render.spv

default: $(SHADERS)

transmittance_lut.spv: transmittance_lut.slang atmosphere_common.slang
	$(SLANGC) $(SLANGFLAGS) -o $@ transmittance_lut.slang

multi_scattering.spv: multi_scattering.slang atmosphere_common.slang
	$(SLANGC) $(SLANGFLAGS) -o $@ multi_scattering.slang

sky_view_lut.spv: sky_view_lut.slang atmosphere_common.slang
	$(SLANGC) $(SLANGFLAGS) -o $@ sky_view_lut.slang

sky_render.spv: sky_render.slang atmosphere_common.slang
	$(SLANGC) $(SLANGFLAGS) -o $@ sky_render.slang

clean:
	rm -f $(SHADERS)
