set(CUHFILES
        memory_alloc/config.h
        )

cuda_add_library(slab_hash STATIC
	${CUHFILES}
	${CUFILES}
	OPTIONS ${GENCODE} ${VERBOSE_PTXAS})
