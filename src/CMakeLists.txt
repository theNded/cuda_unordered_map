set(CUHFILES
		slab_hash/config.cuh
		slab_hash/instantiate.cuh)

cuda_add_library(slab_hash STATIC
	${CUHFILES}
	${CUFILES}
	OPTIONS ${GENCODE} ${VERBOSE_PTXAS})
