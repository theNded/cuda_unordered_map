cuda_add_executable(test_cpu_input legacy/test_cpu_input.cu OPTIONS ${GENCODE})
cuda_add_executable(test_gpu_input legacy/test_gpu_input.cu OPTIONS ${GENCODE})
cuda_add_executable(test_thrust_input legacy/test_thrust_input.cu OPTIONS ${GENCODE})

cuda_add_executable(test_unique test_unique.cu OPTIONS ${GENCODE})

