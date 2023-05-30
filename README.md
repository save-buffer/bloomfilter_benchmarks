# Bloomfilter Benchmarks

I implemented a bunch of Bloom filters and benchmarked them. Benchmarking code is in `bench.cpp`, False Positive Rate-measuring is in
`fpr.cpp`, and implementations are in `bloom_filters.h`. To build, run `build.sh`, and then run `./bench` or `./fpr`. Both will output
CSV to stdout, which you can then graph with `graph.py`. 