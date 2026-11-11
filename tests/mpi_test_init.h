#ifndef DDLA_TEST_MPI_INIT_H
#define DDLA_TEST_MPI_INIT_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* Numerical workers may run concurrently; only the initializing thread calls MPI. */
static inline int ddla_test_mpi_init(int* argc, char*** argv)
{
    int provided = MPI_THREAD_SINGLE;
    const int status = MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
    if (status != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init_thread(FUNNELED) failed with status %d\n", status);
        exit(EXIT_FAILURE);
    }
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED required; MPI provided %d\n", provided);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        abort();
    }
    return MPI_SUCCESS;
}

#endif
