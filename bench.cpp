#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>

#include "bloom_filters.h"

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

const char *output_template = "{{#result}}{{name}}, {{batch}}, {{median(cpucycles)}}{{/result}}\n";

template <typename BloomType>
void BenchmarkBloom(int num_values, const char *label)
{
    ankerl::nanobench::Rng rng;

    std::vector<uint32_t> h1(num_values);
    std::vector<uint32_t> h2(num_values);
    for(int i = 0; i < num_values; i++)
    {
        uint64_t h = rng();
        h1[i] = h & ((1ull << 32) - 1);
        h2[i] = h >> 32;
    }

    BloomType bloom(num_values, 0.01f); // 1% false positive rate
    ankerl::nanobench::Bench()
        .batch(num_values)
        .output(nullptr)
        .run(std::string(label) + "_build", [&]()
    {
        for(int i = 0; i < num_values; i++)
        {
            bloom.Insert(h1[i], h2[i]);
        }
    })
        .render(output_template, std::cout);
    std::cout << std::flush;

    for(int i = 0; i < num_values; i++)
    {
        uint64_t h = rng();
        h1[i] = h & ((1ull << 32) - 1);
        h2[i] = h >> 32;
    }
    std::vector<uint8_t> output((num_values + 7) / 8);
    ankerl::nanobench::Bench()
        .batch(num_values)
        .output(nullptr)
        .run(std::string(label) + "_probe", [&]()
    {
        for(int i = 0; i < num_values; i++)
        {
            bool exists = bloom.Query(h1[i], h2[i]);
            output[i / 8] |= (exists << (i % 8));
        }
    })
        .render(output_template, std::cout);
    std::cout << std::flush;
}


template <>
void BenchmarkBloom<SimdBloomFilter>(int num_values, const char *label)
{
    ankerl::nanobench::Rng rng;

    std::vector<uint32_t> h1(num_values);
    std::vector<uint32_t> h2(num_values);
    for(int i = 0; i < num_values; i++)
    {
        uint64_t h = rng();
        h1[i] = h & ((1ull << 32) - 1);
        h2[i] = h >> 32;
    }

    SimdBloomFilter bloom(num_values, 0.01f); // 1% false positive rate
    ankerl::nanobench::Bench()
        .batch(num_values)
        .output(nullptr)
        .run(std::string(label) + "_build", [&]()
    {
        for(int i = 0; i < num_values; i += 8)
        {
            bloom.Insert(&h1[i], &h2[i]);
        }
    })
        .render(output_template, std::cout);
    std::cout << std::flush;

    for(int i = 0; i < num_values; i++)
    {
        uint64_t h = rng();
        h1[i] = h & ((1ull << 32) - 1);
        h2[i] = h >> 32;
    }
    std::vector<uint8_t> output((num_values + 7) / 8);
    ankerl::nanobench::Bench()
        .batch(num_values)
        .output(nullptr)
        .run(std::string(label) + "_probe", [&]()
    {
        for(int i = 0; i < num_values; i += 8)
        {
            output[i / 8] = bloom.Query(&h1[i], &h2[i]);
        }
    })
        .render(output_template, std::cout);
    std::cout << std::flush;
}

template <>
void BenchmarkBloom<PatternedSimdBloomFilter>(int num_values, const char *label)
{
    ankerl::nanobench::Rng rng;

    std::vector<uint64_t> hash(num_values);
    for(int i = 0; i < num_values; i++)
        hash[i] = rng();

    PatternedSimdBloomFilter bloom(num_values, 0.01f); // 1% false positive rate
    ankerl::nanobench::Bench()
        .batch(num_values)
        .output(nullptr)
        .run(std::string(label) + "_build", [&]()
    {
        for(int i = 0; i < num_values; i += 8)
        {
            bloom.Insert(&hash[i]);
        }
    })
        .render(output_template, std::cout);
    std::cout << std::flush;

    for(int i = 0; i < num_values; i++)
        hash[i] = rng();

    std::vector<uint8_t> output((num_values + 7) / 8);
    ankerl::nanobench::Bench()
        .batch(num_values)
        .output(nullptr)
        .run(std::string(label) + "_probe", [&]()
    {
        for(int i = 0; i < num_values; i += 8)
        {
            output[i / 8] = bloom.Query(&hash[i]);
        }
    })
        .render(output_template, std::cout);
    std::cout << std::flush;
}

template <typename BloomType>
void RunBenchmarks(const char *label)
{
    for(int64_t i = 1024; i <= (1 << 26); i *= 4)
        BenchmarkBloom<BloomType>(i, label);
}

int main()
{
    std::cout << "Operation, NumValues, Cycles/Value" << std::endl;
    //RunBenchmarks<BasicBloomFilter>("basic");
    //RunBenchmarks<BlockedBloomFilter>("blocked");
    //RunBenchmarks<RegisterBlockedBloomFilter<0>>("register_blocked");
    //RunBenchmarks<RegisterBlockedBloomFilter<4>>("register_blocked_compensated");
    //RunBenchmarks<SimdBloomFilter>("register_blocked_simd");
    RunBenchmarks<PatternedSimdBloomFilter>("patterned_register_blocked_simd");
//    BenchmarkBasic(1024);
    return 0;
}
