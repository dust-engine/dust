
/// Eric Heitz, Laurent Belcour, Victor Ostromoukhov, David Coeurjolly and
/// Jean-Claude Iehl. 2019. A Low-Discrepancy Sampler that Distributes Monte
/// Carlo Errors as a Blue Noise in Screen Space. In Proceedings of SIGGRAPH ’19
/// Talks. ACM, New York, NY, USA, 2 pages. https://doi.org/10.1145/3306307.3328191
///
/// https://crates.io/crates/blue-noise-sampler
///
/// Usage: 
float heitz_blue_noise_sampler(
    int pixel_i,
    int pixel_j,
    int sampleIndex,
    int sampleDimension,
) {
	// wrap arguments
	pixel_i = pixel_i & 127;
	pixel_j = pixel_j & 127;
	sampleIndex = sampleIndex & 255;
	sampleDimension = sampleDimension & 255;

	// xor index based on optimized ranking
	// jb: 1spp blue noise has all 0 in ranking_tile_buf so we can skip the load
	int rankedSampleIndex = sampleIndex ^ ranking_tile_buf[sampleDimension + (pixel_i + pixel_j*128)*8];

	// fetch value in sequence
	int value = sobol_buf[sampleDimension + rankedSampleIndex*256];

	// If the dimension is optimized, xor sequence value based on optimized scrambling
	value = value ^ scambling_tile_buf[(sampleDimension%8) + (pixel_i + pixel_j*128)*8];

	// convert to float and return
	float v = (0.5f+value)/256.0f;
	return v;
}
