// Portable-only fused BGR -> exact quantization LUT -> compiler input tiles.
// No vendor headers, allocations, rounding, resizing, or model changes.
#include <cstddef>
#include <cstdint>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
namespace {
constexpr std::size_t side=1280, tile_width=64, bytes=side*side*3;
bool valid(const std::uint8_t* src,std::size_t n,const std::uint8_t* lut,std::size_t l,std::uint8_t* dst,std::size_t d) {
    return src && lut && dst && src!=dst && n==bytes && l==256 && d==bytes;
}
}
extern "C" unsigned portable_pack_abi() { return 1; }
extern "C" unsigned portable_pack_has_openmp() {
#if defined(_OPENMP)
    return 1;
#else
    return 0;
#endif
}
extern "C" int portable_pack_parallel(const std::uint8_t* src,std::size_t n,const std::uint8_t* lut,std::size_t l,std::uint8_t* dst,std::size_t d,unsigned threads) {
    if (!valid(src,n,lut,l,dst,d) || threads<1 || threads>16) return -1;
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int y=0;y<static_cast<int>(side);++y)
        for (std::size_t tile=0;tile<side/tile_width;++tile) {
            std::size_t out=(tile*side*tile_width+static_cast<std::size_t>(y)*tile_width)*3;
            const auto* row=src+(static_cast<std::size_t>(y)*side+tile*tile_width)*3;
            for (std::size_t x=0;x<tile_width;++x) {
                dst[out++]=lut[row[3*x+2]];dst[out++]=lut[row[3*x+1]];dst[out++]=lut[row[3*x]];
            }
        }
    return 0;
#else
    return -3;
#endif
}
extern "C" unsigned portable_pack_has_neon() {
#if defined(__aarch64__)
    return 1;
#else
    return 0;
#endif
}
extern "C" int portable_pack_scalar(const std::uint8_t* src,std::size_t n,const std::uint8_t* lut,std::size_t l,std::uint8_t* dst,std::size_t d) {
    if (!valid(src,n,lut,l,dst,d)) return -1;
    for (std::size_t y=0;y<side;++y)
        for (std::size_t tile=0;tile<side/tile_width;++tile) {
            std::size_t out=(tile*side*tile_width+y*tile_width)*3;
            const auto* row=src+(y*side+tile*tile_width)*3;
            for (std::size_t x=0;x<tile_width;++x) {
                dst[out++]=lut[row[3*x+2]];
                dst[out++]=lut[row[3*x+1]];
                dst[out++]=lut[row[3*x]];
            }
        }
    return 0;
}
#if defined(__aarch64__)
namespace {
inline uint8x16_t lookup(uint8x16_t index,const uint8x16x4_t& a,const uint8x16x4_t& b,const uint8x16x4_t& c,const uint8x16x4_t& d) {
    auto r=vqtbl4q_u8(a,index);
    r=vorrq_u8(r,vqtbl4q_u8(b,vsubq_u8(index,vdupq_n_u8(64))));
    r=vorrq_u8(r,vqtbl4q_u8(c,vsubq_u8(index,vdupq_n_u8(128))));
    return vorrq_u8(r,vqtbl4q_u8(d,vsubq_u8(index,vdupq_n_u8(192))));
}
}
#endif
extern "C" int portable_pack_neon(const std::uint8_t* src,std::size_t n,const std::uint8_t* lut,std::size_t l,std::uint8_t* dst,std::size_t d) {
    if (!valid(src,n,lut,l,dst,d)) return -1;
#if defined(__aarch64__)
    const uint8x16x4_t a={{vld1q_u8(lut),vld1q_u8(lut+16),vld1q_u8(lut+32),vld1q_u8(lut+48)}};
    const uint8x16x4_t b={{vld1q_u8(lut+64),vld1q_u8(lut+80),vld1q_u8(lut+96),vld1q_u8(lut+112)}};
    const uint8x16x4_t c={{vld1q_u8(lut+128),vld1q_u8(lut+144),vld1q_u8(lut+160),vld1q_u8(lut+176)}};
    const uint8x16x4_t e={{vld1q_u8(lut+192),vld1q_u8(lut+208),vld1q_u8(lut+224),vld1q_u8(lut+240)}};
    for (std::size_t y=0;y<side;++y)
        for (std::size_t tile=0;tile<side/tile_width;++tile) {
            std::size_t out=(tile*side*tile_width+y*tile_width)*3;
            const auto* row=src+(y*side+tile*tile_width)*3;
            for (std::size_t x=0;x<tile_width;x+=16) {
                const auto input=vld3q_u8(row+x*3);
                uint8x16x3_t output={{lookup(input.val[2],a,b,c,e),lookup(input.val[1],a,b,c,e),lookup(input.val[0],a,b,c,e)}};
                vst3q_u8(dst+out,output);out+=48;
            }
        }
    return 0;
#else
    return -2; // Never silently substitute a different requested kernel.
#endif
}
