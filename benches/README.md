# Benchmarking
> currently using the Criterion crate for my benchmark framework (analytical ; runs the benchmark multiple times and averages the results with other derived statisitcs)

> might use the iai crate later on (depends on valgrind) for larger benchmarks (non-analytical ; one-shot benchmarks) and profiling

main thing to remember is to use black_box for all constants and literals so the benchmark doesn't get optimized away

run benchmarks with `cargo bench`