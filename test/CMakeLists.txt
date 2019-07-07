# cuda_add_executable(concurrent_map_delete concurrent_map_delete.cu
#         OPTIONS ${GENCODE})
# cuda_add_executable(concurrent_map_multi_batch concurrent_map_multi_batch.cu
#         OPTIONS ${GENCODE})
# cuda_add_executable(concurrent_map_conflict_key concurrent_map_conflict_key.cu
#         OPTIONS ${GENCODE})
#cuda_add_executable(iterator iterator.cu
#        OPTIONS ${GENCODE})
cuda_add_executable(uint7_delete uint7_delete.cu OPTIONS ${GENCODE})
cuda_add_executable(uint7_multi_batch uint7_multi_batch.cu OPTIONS ${GENCODE})
cuda_add_executable(uint7_conflicts uint7_conflicts.cu OPTIONS ${GENCODE})
cuda_add_executable(kernel kernel.cu)