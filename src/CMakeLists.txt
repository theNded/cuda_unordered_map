set(CUHFILES
        slab_hash_config.cuh
        slab_hash_instantiate.cuh)

cuda_add_library(slab_hash STATIC
	${CUHFILES}
	${CUFILES}
	OPTIONS ${GENCODE} ${VERBOSE_PTXAS})
