#pragma once
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <immintrin.h>

struct BasicBloomFilter
{
    BasicBloomFilter(int n, float eps) : n(n), epsilon(eps)
    {
        m = ComputeNumBits();
        k = ComputeNumHashFns();
        bv.resize((m + 7) / 8);
    }

    int ComputeNumBits()
    {
        return static_cast<int>(-1.44 * n * std::log2(epsilon) + 0.5);
    }

    int ComputeNumHashFns()
    {
        return static_cast<int>(-std::log2(epsilon) + 0.5);
    }

    void Insert(uint32_t h1, uint32_t h2)
    {
        for(int i = 0; i < k; i++)
        {
            uint32_t hash = (h1 + i * h2) % m;
            uint64_t bit_idx = hash % 8;
            uint64_t byte_idx = hash / 8;
            bv[byte_idx] |= (1 << bit_idx);
        }
    }

    bool Query(uint32_t h1, uint32_t h2)
    {
        bool result = true;
        for(int i = 0; i < k; i++)
        {
            uint32_t hash = (h1 + i * h2) % m;
            uint64_t bit_idx = hash % 8;
            uint64_t byte_idx = hash / 8;
            result &= (bv[byte_idx] >> bit_idx) & 1;
        }
        return result;
    }

    void Reset()
    {
        std::fill(bv.begin(), bv.end(), 0);
    }

    int n;
    float epsilon;

    int m;
    int k;
    std::vector<uint8_t> bv;
};

constexpr int CACHE_LINE_BITS = 256;
constexpr int CACHE_LINE_BYTES = CACHE_LINE_BITS / 8;

struct BlockedBloomFilter
{
    BlockedBloomFilter(int n, float eps) : n(n), epsilon(eps)
    {
        m = ComputeNumBits();
        k = ComputeNumHashFns();
        num_blocks = (m + CACHE_LINE_BITS - 1) / CACHE_LINE_BITS;
        bv.resize(num_blocks * CACHE_LINE_BYTES);
    }

    int ComputeNumBits()
    {
        return static_cast<int>(-1.44 * n * std::log2(epsilon) + 0.5);
    }

    int ComputeNumHashFns()
    {
        return static_cast<int>(-std::log2(epsilon) + 0.5);
    }

    uint8_t *GetBlock(uint32_t h1, uint32_t h2)
    {
        uint32_t block_idx = h1 % num_blocks;
        uint32_t byte_idx = block_idx * CACHE_LINE_BYTES;
        return bv.data() + byte_idx;
    }

    void Insert(uint32_t h1, uint32_t h2)
    {
        uint8_t *block = GetBlock(h1, h2);
        for(int i = 1; i < k; i++)
        {
            uint32_t bit_pos = (h1 + i * h2) % CACHE_LINE_BITS;
            uint64_t bit_idx = bit_pos % 8;
            uint64_t byte_idx = bit_pos / 8;
            block[byte_idx] |= (1 << bit_idx);
        }
    }

    bool Query(uint32_t h1, uint32_t h2)
    {
        bool result = true;
        uint8_t *block = GetBlock(h1, h2);
        for(int i = 1; i < k; i++)
        {
            uint32_t bit_pos = (h1 + i * h2) % CACHE_LINE_BITS;
            uint64_t bit_idx = bit_pos % 8;
            uint64_t byte_idx = bit_pos / 8;
            result &= (bv[byte_idx] >> bit_idx) & 1;
        }
        return result;
    }

    void Reset()
    {
        std::fill(bv.begin(), bv.end(), 0);
    }

    int n;
    float epsilon;

    int num_blocks;
    int m;
    int k;
    std::vector<uint8_t> bv;
};

template <int Compensation>
struct RegisterBlockedBloomFilter
{
    RegisterBlockedBloomFilter(int n, float eps) : n(n), epsilon(eps)
    {
        m = ComputeNumBits();
        k = ComputeNumHashFns();
        num_blocks = (m + 64 - 1) / 64;
        bv.resize(num_blocks);
    }

    int ComputeNumBits()
    {
        auto bits_per_val = -1.44 * std::log2(epsilon) + Compensation;
        return static_cast<int>(bits_per_val * n + 0.5);
    }

    int ComputeNumHashFns()
    {
        return static_cast<int>(-std::log2(epsilon) + 0.5);
    }

    uint64_t *GetBlock(uint32_t h1, uint32_t h2)
    {
        uint32_t block_idx = h1 % num_blocks;
        return &bv[block_idx];
    }

    uint64_t ConstructMask(uint32_t h1, uint32_t h2)
    {
        uint64_t mask = 0;
        for(int i = 1; i < k; i++)
        {
            uint32_t bit_pos = (h1 + i * h2) % 64;
            mask |= (1ull << bit_pos);
        }
        return mask;
    }

    void Insert(uint32_t h1, uint32_t h2)
    {
        uint64_t *block = GetBlock(h1, h2);
        *block |= ConstructMask(h1, h2);
    }

    bool Query(uint32_t h1, uint32_t h2)
    {
        uint64_t *block = GetBlock(h1, h2);
        uint64_t mask = ConstructMask(h1, h2);
        return (*block & mask) == mask;
    }

    void Reset()
    {
        std::fill(bv.begin(), bv.end(), 0);
    }

    int n;
    float epsilon;

    int num_blocks;
    int m;
    int k;
    std::vector<uint64_t> bv;
};

struct SimdBloomFilter
{
    SimdBloomFilter(int n, float eps) : n(n), epsilon(eps)
    {
        m = ComputeNumBits();
        k = ComputeNumHashFns();
        int log_num_blocks = 64 - __builtin_clz(m) - 6;
        num_blocks = (1 << log_num_blocks);
        bv.resize(num_blocks);
    }

    uint64_t ComputeNumBits()
    {
        double bits_per_val = -1.44 * std::log2(epsilon);
        return static_cast<uint64_t>(bits_per_val * n + 0.5);
    }

    int ComputeNumHashFns()
    {
        return static_cast<int>(-std::log2(epsilon) + 0.5);
    }

    void GetBlockIdx(__m256i &vecBlockIdx, __m256i &vecH1, __m256i &vecH2)
    {
        __m256i vecNumBlocksMask = _mm256_set1_epi64x(num_blocks - 1);
        vecBlockIdx = _mm256_and_si256(vecH1, vecNumBlocksMask);
    }

    void ConstructMask(
        __m256i &vecMask,
        __m256i &vecH1,
        __m256i &vecH2)
    {
        __m256i vecShiftMask = _mm256_set1_epi64x((1 << 6) - 1);
        __m256i vecOnes = _mm256_set1_epi64x(1);
        for(int i = 1; i < k; i++)
        {
            __m256i vecI = _mm256_set1_epi64x(i);
            __m256i vecMulH2 = _mm256_mul_epi32(vecI, vecH2);
            __m256i vecHash = _mm256_add_epi64(vecH1, vecMulH2);
            __m256i vecShift = _mm256_and_si256(vecHash, vecShiftMask);
            __m256i vecPartial = _mm256_sllv_epi64(vecOnes, vecShift);
            vecMask = _mm256_or_si256(vecMask, vecPartial);
        }
    }

    void Insert(uint32_t *h1, uint32_t *h2)
    {
        __m256i vecH1A = _mm256_cvtepi32_epi64(_mm_loadu_si128(reinterpret_cast<__m128i *>(h1 + 0)));
        __m256i vecH1B = _mm256_cvtepi32_epi64(_mm_loadu_si128(reinterpret_cast<__m128i *>(h1 + 4)));
        __m256i vecH2A = _mm256_cvtepi32_epi64(_mm_loadu_si128(reinterpret_cast<__m128i *>(h2 + 0)));
        __m256i vecH2B = _mm256_cvtepi32_epi64(_mm_loadu_si128(reinterpret_cast<__m128i *>(h2 + 4)));

        __m256i vecMaskA = _mm256_setzero_si256();
        __m256i vecMaskB = _mm256_setzero_si256();
        ConstructMask(vecMaskA, vecH1A, vecH2A);
        ConstructMask(vecMaskB, vecH1B, vecH2B);

        __m256i vecBlockIdxA;
        __m256i vecBlockIdxB;
        GetBlockIdx(vecBlockIdxA, vecH1A, vecH2A);
        GetBlockIdx(vecBlockIdxB, vecH1B, vecH2B);

        uint64_t block0 = _mm256_extract_epi64(vecBlockIdxA, 0);
        uint64_t block1 = _mm256_extract_epi64(vecBlockIdxA, 1);
        uint64_t block2 = _mm256_extract_epi64(vecBlockIdxA, 2);
        uint64_t block3 = _mm256_extract_epi64(vecBlockIdxA, 3);
        uint64_t block4 = _mm256_extract_epi64(vecBlockIdxB, 0);
        uint64_t block5 = _mm256_extract_epi64(vecBlockIdxB, 1);
        uint64_t block6 = _mm256_extract_epi64(vecBlockIdxB, 2);
        uint64_t block7 = _mm256_extract_epi64(vecBlockIdxB, 3);

        // Uncomment to generate histogram of block distribution
        // printf("%d, %d, %d, %d, %d, %d, %d, %d,\n", block0, block1, block2, block3, block4, block5, block6, block7);

        bv[block0] |= _mm256_extract_epi64(vecMaskA, 0);
        bv[block1] |= _mm256_extract_epi64(vecMaskA, 1);
        bv[block2] |= _mm256_extract_epi64(vecMaskA, 2);
        bv[block3] |= _mm256_extract_epi64(vecMaskA, 3);
        bv[block4] |= _mm256_extract_epi64(vecMaskB, 0);
        bv[block5] |= _mm256_extract_epi64(vecMaskB, 1);
        bv[block6] |= _mm256_extract_epi64(vecMaskB, 2);
        bv[block7] |= _mm256_extract_epi64(vecMaskB, 3);
    }

    uint8_t Query(uint32_t *h1, uint32_t *h2)
    {
        __m256i vecH1A = _mm256_cvtepi32_epi64(_mm_loadu_si128(reinterpret_cast<__m128i *>(h1 + 0)));
        __m256i vecH1B = _mm256_cvtepi32_epi64(_mm_loadu_si128(reinterpret_cast<__m128i *>(h1 + 4)));
        __m256i vecH2A = _mm256_cvtepi32_epi64(_mm_loadu_si128(reinterpret_cast<__m128i *>(h2 + 0)));
        __m256i vecH2B = _mm256_cvtepi32_epi64(_mm_loadu_si128(reinterpret_cast<__m128i *>(h2 + 4)));

        __m256i vecMaskA = _mm256_setzero_si256();
        __m256i vecMaskB = _mm256_setzero_si256();
        ConstructMask(vecMaskA, vecH1A, vecH2A);
        ConstructMask(vecMaskB, vecH1B, vecH2B);

        __m256i vecBlockIdxA;
        __m256i vecBlockIdxB;
        GetBlockIdx(vecBlockIdxA, vecH1A, vecH2A);
        GetBlockIdx(vecBlockIdxB, vecH1B, vecH2B);

        __m256i vecBloomA = _mm256_i64gather_epi64((const long long *)bv.data(), vecBlockIdxA, sizeof(uint64_t));
        __m256i vecBloomB = _mm256_i64gather_epi64((const long long *)bv.data(), vecBlockIdxB, sizeof(uint64_t));
        __m256i vecCmpA = _mm256_cmpeq_epi64(_mm256_and_si256(vecMaskA, vecBloomA), vecMaskA);
        __m256i vecCmpB = _mm256_cmpeq_epi64(_mm256_and_si256(vecMaskB, vecBloomB), vecMaskB);
        uint32_t res_a = static_cast<uint32_t>(_mm256_movemask_epi8(vecCmpA));
        uint32_t res_b = static_cast<uint32_t>(_mm256_movemask_epi8(vecCmpB));
        uint64_t res_bytes = res_a | (static_cast<uint64_t>(res_b) << 32);
        uint8_t res_bits = static_cast<uint8_t>(_mm256_movemask_epi8(_mm256_set1_epi64x(res_bytes)) & 0xff);
        return res_bits;
    }

    void Reset()
    {
        std::fill(bv.begin(), bv.end(), 0);
    }

    int n;
    float epsilon;

    uint64_t num_blocks;
    int m;
    int k;
    std::vector<uint64_t> bv;
};

struct MaskTable
{
    MaskTable()
    {
        std::memset(masks, 0, sizeof(masks));
        std::random_device rd;
        std::default_random_engine gen(rd());
        std::uniform_int_distribution<int> first_mask_distrib(min_bits_set, max_bits_set);
        std::uniform_int_distribution<int> bit_pos_distrib(0, bits_per_mask - 1);
        std::uniform_int_distribution<int> bit_set_distrib(0, bits_per_mask * 2 - 1);

        int num_set_in_first_mask = first_mask_distrib(gen);
        for(int i = 0; i < num_set_in_first_mask; i++)
        {
            int bit_pos;
            do
            {
                bit_pos = bit_pos_distrib(gen);
            } while((masks[bit_pos / 8] >> (bit_pos % 8)) & 1);
            masks[bit_pos / 8] |= (1 << (bit_pos) % 8);
        }

        int total_bits = num_masks + bits_per_mask - 1;
        int num_set_in_current_mask = num_set_in_first_mask;
        for(int i = bits_per_mask; i < total_bits; i++)
        {
            int leaving_bit_idx = i - bits_per_mask;
            int leaving_bit = (masks[leaving_bit_idx / 8] >> (leaving_bit_idx % 8)) & 1;
            if(leaving_bit == 1 && num_set_in_current_mask == min_bits_set)
            {
                masks[i / 8] |= (1 << (i % 8));
                continue;
            }
            if(leaving_bit == 0 && num_set_in_current_mask == max_bits_set)
            {
                continue;
            }

            if(bit_set_distrib(gen) < min_bits_set + max_bits_set)
            {
                masks[i / 8] |= (1 << (i % 8));
                if(leaving_bit == 0)
                    num_set_in_current_mask += 1;
            }
            else
            {
                if(leaving_bit == 1)
                    num_set_in_current_mask -= 1;
            }
        }
    }

    static constexpr int bits_per_mask = 57;
    static constexpr int min_bits_set = 4;
    static constexpr int max_bits_set = 5;

    static constexpr int log_num_masks = 10;
    static constexpr int num_masks = 1 << log_num_masks;
    static constexpr int mask_bytes = (num_masks + 64) / 8;
    uint8_t masks[mask_bytes];
};

struct PatternedSimdBloomFilter
{
    PatternedSimdBloomFilter(int n, float eps) : n(n), epsilon(eps)
    {
        m = ComputeNumBits();
        int log_num_blocks = 64 - __builtin_clz(m) - 6;
        num_blocks = (1 << log_num_blocks);
        bv.resize(num_blocks);
    }

    uint64_t ComputeNumBits()
    {
        return std::max(512, 8 * n);
    }

    void GetBlockIdx(__m256i &vecBlockIdx, __m256i &vecHash)
    {
        __m256i vecNumBlocksMask = _mm256_set1_epi64x(num_blocks - 1);
        vecBlockIdx = _mm256_srli_epi64(vecHash, mask_idx_bits + rotate_bits);
        vecBlockIdx = _mm256_and_si256(vecBlockIdx, vecNumBlocksMask);
    }

    void ConstructMask(
        __m256i &vecMask,
        __m256i &vecHash)
    {
        __m256i vecMaskIdxMask = _mm256_set1_epi64x((1 << mask_idx_bits) - 1);
        __m256i vecMaskMask = _mm256_set1_epi64x((1ull << MaskTable::bits_per_mask) - 1);
        __m256i vec64 = _mm256_set1_epi64x(64);

        __m256i vecMaskIdx = _mm256_and_si256(vecHash, vecMaskIdx);
        __m256i vecMaskByteIdx = _mm256_srli_epi64(vecMaskIdx, 3);
        __m256i vecMaskBitIdx = _mm256_and_si256(vecMaskIdx, _mm256_set1_epi64x(0x7));
        __m256i vecRawMasks = _mm256_i64gather_epi64((const long long *)masks.masks, vecMaskByteIdx, 1);
        __m256i vecUnrotated = _mm256_and_si256(_mm256_srlv_epi64(vecRawMasks, vecMaskBitIdx), vecMaskMask);

        __m256i vecRotation = _mm256_and_si256(_mm256_srli_epi64(vecHash, mask_idx_bits), _mm256_set1_epi64x((1 << rotate_bits) - 1));
        __m256i vecShiftUp = _mm256_sllv_epi64(vecUnrotated, vecRotation);
        __m256i vecShiftDown = _mm256_srlv_epi64(vecUnrotated, _mm256_sub_epi64(vec64, vecRotation));
        vecMask = _mm256_or_si256(vecShiftDown, vecShiftUp);
    }

    void Insert(uint64_t *hash)
    {
        __m256i vecHashA = _mm256_loadu_si256(reinterpret_cast<__m256i *>(hash + 0));
        __m256i vecHashB = _mm256_loadu_si256(reinterpret_cast<__m256i *>(hash + 4));

        __m256i vecMaskA = _mm256_setzero_si256();
        __m256i vecMaskB = _mm256_setzero_si256();
        ConstructMask(vecMaskA, vecHashA);
        ConstructMask(vecMaskB, vecHashB);

        __m256i vecBlockIdxA;
        __m256i vecBlockIdxB;
        GetBlockIdx(vecBlockIdxA, vecHashA);
        GetBlockIdx(vecBlockIdxB, vecHashB);

        uint64_t block0 = _mm256_extract_epi64(vecBlockIdxA, 0);
        uint64_t block1 = _mm256_extract_epi64(vecBlockIdxA, 1);
        uint64_t block2 = _mm256_extract_epi64(vecBlockIdxA, 2);
        uint64_t block3 = _mm256_extract_epi64(vecBlockIdxA, 3);
        uint64_t block4 = _mm256_extract_epi64(vecBlockIdxB, 0);
        uint64_t block5 = _mm256_extract_epi64(vecBlockIdxB, 1);
        uint64_t block6 = _mm256_extract_epi64(vecBlockIdxB, 2);
        uint64_t block7 = _mm256_extract_epi64(vecBlockIdxB, 3);

        // Uncomment to generate histogram of block distribution
        // printf("%d, %d, %d, %d, %d, %d, %d, %d,\n", block0, block1, block2, block3, block4, block5, block6, block7);

        bv[block0] |= _mm256_extract_epi64(vecMaskA, 0);
        bv[block1] |= _mm256_extract_epi64(vecMaskA, 1);
        bv[block2] |= _mm256_extract_epi64(vecMaskA, 2);
        bv[block3] |= _mm256_extract_epi64(vecMaskA, 3);
        bv[block4] |= _mm256_extract_epi64(vecMaskB, 0);
        bv[block5] |= _mm256_extract_epi64(vecMaskB, 1);
        bv[block6] |= _mm256_extract_epi64(vecMaskB, 2);
        bv[block7] |= _mm256_extract_epi64(vecMaskB, 3);
    }

    uint8_t Query(uint64_t *hash)
    {
        __m256i vecHashA = _mm256_loadu_si256(reinterpret_cast<__m256i *>(hash + 0));
        __m256i vecHashB = _mm256_loadu_si256(reinterpret_cast<__m256i *>(hash + 4));

        __m256i vecMaskA = _mm256_setzero_si256();
        __m256i vecMaskB = _mm256_setzero_si256();
        ConstructMask(vecMaskA, vecHashA);
        ConstructMask(vecMaskB, vecHashB);

        __m256i vecBlockIdxA;
        __m256i vecBlockIdxB;
        GetBlockIdx(vecBlockIdxA, vecHashA);
        GetBlockIdx(vecBlockIdxB, vecHashB);

        __m256i vecBloomA = _mm256_i64gather_epi64((const long long *)bv.data(), vecBlockIdxA, sizeof(uint64_t));
        __m256i vecBloomB = _mm256_i64gather_epi64((const long long *)bv.data(), vecBlockIdxB, sizeof(uint64_t));
        __m256i vecCmpA = _mm256_cmpeq_epi64(_mm256_and_si256(vecMaskA, vecBloomA), vecMaskA);
        __m256i vecCmpB = _mm256_cmpeq_epi64(_mm256_and_si256(vecMaskB, vecBloomB), vecMaskB);
        uint32_t res_a = static_cast<uint32_t>(_mm256_movemask_epi8(vecCmpA));
        uint32_t res_b = static_cast<uint32_t>(_mm256_movemask_epi8(vecCmpB));
        uint64_t res_bytes = res_a | (static_cast<uint64_t>(res_b) << 32);
        uint8_t res_bits = static_cast<uint8_t>(_mm256_movemask_epi8(_mm256_set1_epi64x(res_bytes)) & 0xff);
        return res_bits;
    }

    void Reset()
    {
        std::fill(bv.begin(), bv.end(), 0);
    }

    int n;
    float epsilon;

    uint64_t num_blocks;
    uint64_t m;
    MaskTable masks;
    std::vector<uint64_t> bv;    

    static constexpr int mask_idx_bits = MaskTable::log_num_masks;
    static constexpr int rotate_bits = 6;
};
