cuda_add_executable(test_cpu_input test_cpu_input.cu OPTIONS ${GENCODE})
cuda_add_executable(test_gpu_input test_gpu_input.cu OPTIONS ${GENCODE})
cuda_add_executable(test_thrust_input test_thrust_input.cu OPTIONS ${GENCODE})
cuda_add_executable(test_api_readwrite test_api_readwrite.cu OPTIONS ${GENCODE})
cuda_add_executable(test_api_readonly test_api_readonly.cu OPTIONS ${GENCODE})

