#include <math.h>
#include <stdint.h>

#define A_CPU 1
#include "ffx_a.h"

namespace {
#include "ffx_lpm.h"
}

// Matches `DustLpmPreset` on the Rust side. Order is load-bearing.
enum DustLpmPreset : uint32_t {
    DUST_LPM_709_709 = 0,
    DUST_LPM_709_P3 = 1,
    DUST_LPM_709_2020 = 2,
    DUST_LPM_FS2RAW_709 = 3,
    DUST_LPM_FS2SCRGB_709 = 4,
    DUST_LPM_HDR10RAW_709 = 5,
    DUST_LPM_HDR10SCRGB_709 = 6,
    DUST_LPM_FS2RAW_P3 = 7,
    DUST_LPM_FS2SCRGB_P3 = 8,
    DUST_LPM_HDR10RAW_P3 = 9,
    DUST_LPM_HDR10SCRGB_P3 = 10,
    DUST_LPM_FS2RAW_2020 = 11,
    DUST_LPM_FS2SCRGB_2020 = 12,
    DUST_LPM_HDR10RAW_2020 = 13,
    DUST_LPM_HDR10SCRGB_2020 = 14,
};

extern "C" void dust_lpm_setup(
    uint32_t *ctl,
    uint32_t preset,
    const float *p_saturation, // float[3]
    const float *p_crosstalk,  // float[3]
    const float *p_fs2_red,    // float[2], only used by FS2* presets
    const float *p_fs2_green,  // float[2], only used by FS2* presets
    const float *p_fs2_blue,   // float[2], only used by FS2* presets
    const float *p_fs2_white,  // float[2], only used by FS2* presets
    float fs2_scalar,          // only used by FS2SCRGB_* presets
    float hdr10_scalar,        // only used by HDR10* presets
    float soft_gap,
    float hdr_max,
    float exposure,
    float contrast,
    float shoulder_contrast,
    bool shoulder) {
    float saturation[3] = {p_saturation[0], p_saturation[1], p_saturation[2]};
    float crosstalk[3] = {p_crosstalk[0], p_crosstalk[1], p_crosstalk[2]};
    float fs2R[2] = {p_fs2_red[0], p_fs2_red[1]};
    float fs2G[2] = {p_fs2_green[0], p_fs2_green[1]};
    float fs2B[2] = {p_fs2_blue[0], p_fs2_blue[1]};
    float fs2W[2] = {p_fs2_white[0], p_fs2_white[1]};
    float fs2S = fs2_scalar;
    float hdr10S = hdr10_scalar;

#define DUST_LPM_CALL(CONFIG_MACRO, COLORS_MACRO)          \
    LpmSetup(                                              \
        ctl,                                               \
        shoulder,                                          \
        CONFIG_MACRO,                                      \
        COLORS_MACRO,                                      \
        soft_gap,                                          \
        hdr_max,                                           \
        exposure,                                          \
        contrast,                                          \
        shoulder_contrast,                                 \
        saturation,                                        \
        crosstalk)

    switch (preset) {
    case DUST_LPM_709_709:
        DUST_LPM_CALL(LPM_CONFIG_709_709, LPM_COLORS_709_709);
        break;
    case DUST_LPM_709_P3:
        DUST_LPM_CALL(LPM_CONFIG_709_P3, LPM_COLORS_709_P3);
        break;
    case DUST_LPM_709_2020:
        DUST_LPM_CALL(LPM_CONFIG_709_2020, LPM_COLORS_709_2020);
        break;
    case DUST_LPM_FS2RAW_709:
        DUST_LPM_CALL(LPM_CONFIG_FS2RAW_709, LPM_COLORS_FS2RAW_709);
        break;
    case DUST_LPM_FS2SCRGB_709:
        DUST_LPM_CALL(LPM_CONFIG_FS2SCRGB_709, LPM_COLORS_FS2SCRGB_709);
        break;
    case DUST_LPM_HDR10RAW_709:
        DUST_LPM_CALL(LPM_CONFIG_HDR10RAW_709, LPM_COLORS_HDR10RAW_709);
        break;
    case DUST_LPM_HDR10SCRGB_709:
        DUST_LPM_CALL(LPM_CONFIG_HDR10SCRGB_709, LPM_COLORS_HDR10SCRGB_709);
        break;
    case DUST_LPM_FS2RAW_P3:
        DUST_LPM_CALL(LPM_CONFIG_FS2RAW_P3, LPM_COLORS_FS2RAW_P3);
        break;
    case DUST_LPM_FS2SCRGB_P3:
        DUST_LPM_CALL(LPM_CONFIG_FS2SCRGB_P3, LPM_COLORS_FS2SCRGB_P3);
        break;
    case DUST_LPM_HDR10RAW_P3:
        DUST_LPM_CALL(LPM_CONFIG_HDR10RAW_P3, LPM_COLORS_HDR10RAW_P3);
        break;
    case DUST_LPM_HDR10SCRGB_P3:
        DUST_LPM_CALL(LPM_CONFIG_HDR10SCRGB_P3, LPM_COLORS_HDR10SCRGB_P3);
        break;
    case DUST_LPM_FS2RAW_2020:
        DUST_LPM_CALL(LPM_CONFIG_FS2RAW_2020, LPM_COLORS_FS2RAW_2020);
        break;
    case DUST_LPM_FS2SCRGB_2020:
        DUST_LPM_CALL(LPM_CONFIG_FS2SCRGB_2020, LPM_COLORS_FS2SCRGB_2020);
        break;
    case DUST_LPM_HDR10RAW_2020:
        DUST_LPM_CALL(LPM_CONFIG_HDR10RAW_2020, LPM_COLORS_HDR10RAW_2020);
        break;
    case DUST_LPM_HDR10SCRGB_2020:
        DUST_LPM_CALL(LPM_CONFIG_HDR10SCRGB_2020, LPM_COLORS_HDR10SCRGB_2020);
        break;
    }

#undef DUST_LPM_CALL
}
