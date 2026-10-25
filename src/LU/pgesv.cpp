#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include <vector>
#include "ddla_connector.h"
#include <stdexcept>
#include <string>
#include "ddla_stream_impl.h"
#include "require_backend.h"

namespace ddla{

template <DdlaBackend Backend, typename T>
void pgesv(
    const DdlaHandle_t& handle, char side, char trans, int n, int nrhs,
    T* A, const int* array_descA,
    T* B, const int* array_descB
)
{
    check_desc(array_descB, handle);
    check_desc(array_descA, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);


    DdlaHandle_t ddla_handle = handle;
    detail::require_backend<Backend>(ddla_handle, "pgesv");
    assert(n <= array_descA[DDLA_M_] && n <= array_descA[DDLA_N_]);
    const int n_loc_A = num_loc(n, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
    std::vector<int> ipiv(n_loc_A);
    int info = 0;
    pgetrf<Backend>(handle, 
        n, n,
        A, 1, 1, array_descA,
        ipiv.data(),
        info
    );
    if(info !=0){
        throw std::runtime_error("pgesv: pgetrf returned info = " + std::to_string(info));
    }
    pgetrs<Backend>(handle, 
        side, trans, n, nrhs,
        A, array_descA,
        ipiv.data(),
        B, array_descB
    );
}

#define INSTANTIATE_PGESV(BACKEND, TYPE)                                   \
    template void pgesv<BACKEND, TYPE>(const DdlaHandle_t&, char side, char trans, int n, int nrhs, \
    TYPE* A, const int* array_descA, \
    TYPE* B, const int* array_descB)

#if DDLA_HAS_CPU
INSTANTIATE_PGESV(DdlaBackend::CPU, float);
INSTANTIATE_PGESV(DdlaBackend::CPU, double);
INSTANTIATE_PGESV(DdlaBackend::CPU, std::complex<float>);
INSTANTIATE_PGESV(DdlaBackend::CPU, std::complex<double>);
#endif

#if DDLA_HAS_GPU
INSTANTIATE_PGESV(DdlaBackend::GPU, float);
INSTANTIATE_PGESV(DdlaBackend::GPU, double);
INSTANTIATE_PGESV(DdlaBackend::GPU, std::complex<float>);
INSTANTIATE_PGESV(DdlaBackend::GPU, std::complex<double>);
#endif

#undef INSTANTIATE_PGESV


}
