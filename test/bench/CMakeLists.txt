add_library(random_gen batched_data_gen.cpp)
cuda_add_executable(benchmark main_benchmarks.cu
			OPTIONS ${GENCODE} )

target_link_libraries(benchmark random_gen)
