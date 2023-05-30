#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_set>

#include "bloom_filters.h"

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

template <typename BloomType>
void FprBloom(int num_values, const char *label)
{
    ankerl::nanobench::Rng rng;

    std::vector<uint32_t> h1(num_values);
    std::vector<uint32_t> h2(num_values);

    std::unordered_set<uint64_t> taken;
    for(int i = 0; i < num_values; i++)
    {
        uint64_t h = rng();
        taken.insert(h);
        h1[i] = h & ((1ull << 32) - 1);
        h2[i] = h >> 32;
    }

    BloomType bloom(num_values, 0.01f);
    for(int i = 0; i < num_values; i++)
    {
        bloom.Insert(h1[i], h2[i]);
    }

    uint64_t num_false_positives = 0;
    for(int i = 0; i < num_values; i++)
    {
        uint64_t h;
        do
        {
            h = rng();
        } while(taken.find(h) != taken.end());
        h1[i] = h & ((1ull << 32) - 1);
        h2[i] = h >> 32;
    }
    for(int i = 0; i < num_values; i++)
    {
        if(bloom.Query(h1[i], h2[i]))
            num_false_positives++;
    }
    std::cout << label << ", " << num_values << ", " << (static_cast<double>(num_false_positives) / static_cast<double>(num_values)) << std::endl;
}

template <>
void FprBloom<SimdBloomFilter>(int num_values, const char *label)
{
    ankerl::nanobench::Rng rng;

    std::vector<uint32_t> h1(num_values);
    std::vector<uint32_t> h2(num_values);

    std::unordered_set<uint64_t> taken;
    for(int i = 0; i < num_values; i++)
    {
        uint64_t h = rng();
        taken.insert(h);
        h1[i] = h & ((1ull << 32) - 1);
        h2[i] = h >> 32;
    }

    SimdBloomFilter bloom(num_values, 0.01f);
    for(int i = 0; i < num_values; i += 8)
    {
        bloom.Insert(&h1[i], &h2[i]);
    }

    uint64_t num_false_positives = 0;
    for(int i = 0; i < num_values; i++)
    {
        uint64_t h;
        do
        {
            h = rng();
        } while(taken.find(h) != taken.end());
        h1[i] = h & ((1ull << 32) - 1);
        h2[i] = h >> 32;
    }
    for(int i = 0; i < num_values; i += 8)
    {
        num_false_positives += __builtin_popcount(bloom.Query(&h1[i], &h2[i]));
    }
    std::cout << label << ", " << num_values << ", " << (static_cast<double>(num_false_positives) / static_cast<double>(num_values)) << std::endl;
}

template <>
void FprBloom<PatternedSimdBloomFilter>(int num_values, const char *label)
{
    ankerl::nanobench::Rng rng;

    std::vector<uint64_t> hash(num_values);

    std::unordered_set<uint64_t> taken;
    for(int i = 0; i < num_values; i++)
        hash[i] = rng();

    PatternedSimdBloomFilter bloom(num_values, 0.01f);
    for(int i = 0; i < num_values; i += 8)
        bloom.Insert(&hash[i]);

    uint64_t num_false_positives = 0;
    for(int i = 0; i < num_values; i++)
    {
        uint64_t h;
        do
        {
            h = rng();
        } while(taken.find(h) != taken.end());
        hash[i] = h;
    }
    for(int i = 0; i < num_values; i += 8)
    {
        num_false_positives += __builtin_popcount(bloom.Query(&hash[i]));
    }
    std::cout << label << ", " << num_values << ", " << (static_cast<double>(num_false_positives) / static_cast<double>(num_values)) << std::endl;
}

template <typename BloomType>
void RunFpr(const char *label)
{
    for(int64_t i = 1024; i <= (1 << 26); i *= 4)
        FprBloom<BloomType>(i, label);
}

int main()
{
    std::cout << "Filter, NumValues, FPR\n";
    //RunFpr<RegisterBlockedBloomFilter<0>>("register_blocked");
    //RunFpr<RegisterBlockedBloomFilter<4>>("register_blocked_compensated");
    //RunFpr<SimdBloomFilter>("register_blocked_simd");
    RunFpr<PatternedSimdBloomFilter>("patterned_register_blocked_simd");
    //FprBloom<SimdBloomFilter>(1024, "register_blocked_simd");
    //FprBloom<PatternedSimdBloomFilter>(1024, "patterned_register_blocked_simd");
    return 0;
}
