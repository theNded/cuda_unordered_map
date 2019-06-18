cuda_add_executable(concurrent_map concurrent_map.cu
        OPTIONS ${GENCODE})

cuda_add_executable(concurrent_set concurrent_set.cu
        OPTIONS ${GENCODE})

cuda_add_executable(iterator iterator.cu
        OPTIONS ${GENCODE})
