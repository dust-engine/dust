layout(set = 0, binding = 2) readonly buffer HeitzBlueNoiseStorageBuffer{
	uint8_t sobol[256*256];
	uint8_t scrambling_tile[8*128*128];
	uint8_t ranking_tile[8*128*128];
} heitz_blue_noise_buffers;


/// Eric Heitz, Laurent Belcour, Victor Ostromoukhov, David Coeurjolly and
/// Jean-Claude Iehl. 2019. A Low-Discrepancy Sampler that Distributes Monte
/// Carlo Errors as a Blue Noise in Screen Space. In Proceedings of SIGGRAPH ’19
/// Talks. ACM, New York, NY, USA, 2 pages. https://doi.org/10.1145/3306307.3328191
///
/// https://crates.io/crates/blue-noise-sampler
///
/// Usage: 
float heitz_blue_noise_sampler(
	u32vec2 pixel,
    uint sampleIndex,
    uint sampleDimension
) {
	// wrap arguments
	pixel = pixel & 127;
	sampleIndex = sampleIndex & 255;
	sampleDimension = sampleDimension & 255;

	// xor index based on optimized ranking
	// jb: 1spp blue noise has all 0 in ranking_tile_buf so we can skip the load
	uint rankedSampleIndex = sampleIndex ^ heitz_blue_noise_buffers.ranking_tile[sampleDimension + (pixel.x + pixel.y*128)*8];

	// fetch value in sequence
	uint value = heitz_blue_noise_buffers.sobol[sampleDimension + rankedSampleIndex*256];

	// If the dimension is optimized, xor sequence value based on optimized scrambling
	value = value ^ heitz_blue_noise_buffers.scrambling_tile[(sampleDimension%8) + (pixel.x + pixel.y*128)*8];

	// convert to float and return
	float v = (0.5f+value)/256.0f;
	return v;
}
