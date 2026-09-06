#include <ddla/ddla.h>
#include "ddla_desc.h"
#include <cassert>
#include <vector>
#include "ddla_connector.h"
#include <stdexcept>
#include <string>
#include "ddla_stream_impl.h"
#include "require_gpu.h"

namespace ddla{

template <typename T>
void pgesv(
    const DdlaHandle_t& handle, const char& side, const char& trans, const int& n, const int& nrhs,
    T* d_A, const int* array_descA,
    T* d_B, const int* array_descB
)
{
    check_desc(array_descB, handle);
    check_desc(array_descA, handle);
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);


    DdlaHandle_t ddla_handle = handle;
    detail::require_gpu_backend(ddla_handle, "pgesv");
    assert(n <= array_descA[DDLA_M_] && n <= array_descA[DDLA_N_]);
    const int n_loc_A = num_loc(n, array_descA[DDLA_MB_], myprow, array_descA[DDLA_RSRC_], nprows);
    std::vector<int> ipiv(n_loc_A);
    int info = 0;
    pgetrf(handle, 
        n, n,
        d_A, array_descA,
        ipiv.data(),
        info
    );
    if(info !=0){
        throw std::runtime_error("pgesv: pgetrf returned info = " + std::to_string(info));
    }
    pgetrs(handle, 
        side, trans, n, nrhs,
        d_A, array_descA,
        ipiv.data(),
        d_B, array_descB
    );
}

template void pgesv<float>(
    const DdlaHandle_t&, const char& side, const char& trans, const int& n, const int& nrhs,
    float* d_A, const int* array_descA,
    float* d_B, const int* array_descB
);

template void pgesv<double>(
    const DdlaHandle_t&, const char& side, const char& trans, const int& n, const int& nrhs,
    double* d_A, const int* array_descA,
    double* d_B, const int* array_descB
);

template void pgesv<std::complex<float>>(
    const DdlaHandle_t&, const char& side, const char& trans, const int& n, const int& nrhs,
    std::complex<float>* d_A, const int* array_descA,
    std::complex<float>* d_B, const int* array_descB
);

template void pgesv<std::complex<double>>(
    const DdlaHandle_t&, const char& side, const char& trans, const int& n, const int& nrhs,
    std::complex<double>* d_A, const int* array_descA,
    std::complex<double>* d_B, const int* array_descB
);

}
