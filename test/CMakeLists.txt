cuda_add_executable(concurrent_map concurrent_map.cu
        OPTIONS ${GENCODE})
cuda_add_executable(concurrent_map_multi_batch concurrent_map_multi_batch.cu
        OPTIONS ${GENCODE})
cuda_add_executable(concurrent_map_conflict_key concurrent_map_conflict_key.cu
        OPTIONS ${GENCODE})
#cuda_add_executable(iterator iterator.cu
#        OPTIONS ${GENCODE})
