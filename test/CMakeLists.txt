cuda_add_executable(test_cpu_input test_cpu_input.cu OPTIONS ${GENCODE})
cuda_add_executable(test_gpu_input test_gpu_input.cu OPTIONS ${GENCODE})
cuda_add_executable(test_indexer test_indexer.cu OPTIONS ${GENCODE})
