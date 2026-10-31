#!/usr/bin/env python3
"""Generate the LibDDLA Fortran binding layer from one signature table.

Outputs (next to this script):
  ddla_c_api.cpp  -- extern "C" C-ABI shim over the C++ templates of ddla.h
  ddla.f90        -- Fortran 2008 module `ddla` with bind(C) interfaces

The table covers the public API committed at HEAD 9ccc614.  Scalar types are
bound LAPACK-style with a type prefix (ddla_s_/d_/c_/z_ = float/double/
complex<float>/complex<double>); no generic names (most routines take only
c_ptr matrix arguments, which cannot distinguish the scalar type in a Fortran
generic resolution).
"""

TYPES = [
    # suffix, C++ scalar,    Fortran kind,  complex?, real kind
    ("s", "float",              "c_float",  False, "c_float"),
    ("d", "double",             "c_double", False, "c_double"),
    ("c", "std::complex<float>",  "c_float",  True, "c_float"),
    ("z", "std::complex<double>", "c_double", True, "c_double"),
]

# arg kinds:
#   h      handle (c_ptr value / void*)
#   c      character flag by value
#   i      int by value
#   i64    int64 by value
#   s      scalar of the matrix type (real: by ref; complex: bind(C) struct by ref)
#   sr     scalar of the REAL type of the matrix type (herk/her2k)
#   pm_io  backend pointer, written
#   pm_i   backend pointer, read-only
#   ph_in  host int array, read-only (ipiv of pgetrf/plapiv)
#   ph_io  host int array, written
#   pt     host array of the matrix type, written (pgeqrf tau)
#   d      ScaLAPACK int[9] descriptor, read-only
#   ip     host int, written (info, iamax result)
#   b      int boolean by value (0/1)
#   pwork  generic backend pointer (iwork; pass c_null_ptr)

FUNCS = [
    # ---------------- distributed: triangular solves ----------------
    ("ptrtrs", "all4", [("h","handle"),("c","side"),("c","uplo"),("c","trans"),("c","diag"),
                        ("i","m"),("i","n"),("pm_io","A"),("d","descA"),
                        ("pm_io","B"),("d","descB")], {}),
    ("ptrmm", "all4", [("h","handle"),("c","side"),("c","uplo"),("c","trans"),("c","diag"),
                       ("i","m"),("i","n"),("s","alpha"),("pm_i","A"),("d","descA"),
                       ("pm_io","B"),("d","descB")], {}),
    ("plapiv", "all4", [("h","handle"),("c","direc"),("c","rowcol"),("c","pivroc"),
                        ("i","m"),("i","n"),("pm_io","A"),("d","descA"),
                        ("ph_in","ipiv"),("d","descIP"),("pi_io","iwork")], {}),
    ("pswap", "all4", [("h","handle"),("i","N"),("pm_io","A"),("i","ia"),("i","ja"),
                       ("d","descA"),("i","inca"),("pm_io","B"),("i","ib"),("i","jb"),
                       ("d","descB"),("i","incb")], {}),
    ("gather", "all4", [("h","handle"),("i","m"),("i","n"),("pm_i","A"),("i","ia"),
                        ("i","ja"),("d","descA"),("i","root"),("pm_io","dst"),("i","lddst")], {}),
    ("scatter", "all4", [("h","handle"),("i","m"),("i","n"),("pm_io","A"),("i","ia"),
                         ("i","ja"),("d","descA"),("i","root"),("pm_i","src"),("i","ldsrc")], {}),
    # ---------------- distributed: LU family ----------------
    ("pgetrf", "all4", [("h","handle"),("i","m"),("i","n"),("pm_io","A"),("i","ia"),
                        ("i","ja"),("d","descA"),("ph_io","ipiv"),("ip","info")], {}),
    ("pgetrf_nopiv", "all4", [("h","handle"),("i","m"),("i","n"),("pm_io","A"),
                              ("d","descA"),("ip","info")], {}),
    ("pgetrs", "all4", [("h","handle"),("c","side"),("c","trans"),("i","n"),("i","nrhs"),
                        ("pm_io","A"),("d","descA"),("ph_in","ipiv"),
                        ("pm_io","B"),("d","descB")], {}),
    ("pgetrs_nopiv", "all4", [("h","handle"),("c","side"),("c","trans"),("i","n"),("i","nrhs"),
                              ("pm_io","A"),("d","descA"),("pm_io","B"),("d","descB")], {}),
    ("pgesv", "all4", [("h","handle"),("c","side"),("c","trans"),("i","n"),("i","nrhs"),
                       ("pm_io","A"),("d","descA"),("pm_io","B"),("d","descB")], {}),
    ("pgesv_nopiv", "all4", [("h","handle"),("c","side"),("c","trans"),("i","n"),("i","nrhs"),
                             ("pm_io","A"),("d","descA"),("pm_io","B"),("d","descB")], {}),
    ("pgetrf_bpiv", "all4", [("h","handle"),("i","m"),("i","n"),("pm_io","A"),("d","descA"),
                             ("pi_io","pivots"),("ip","info")], {}),
    ("pgetrs_bpiv", "all4", [("h","handle"),("c","side"),("c","trans"),("i","n"),("i","nrhs"),
                             ("pm_io","A"),("d","descA"),("pi_io","pivots"),
                             ("pm_io","B"),("d","descB")], {}),
    ("pgesv_bpiv", "all4", [("h","handle"),("c","side"),("c","trans"),("i","n"),("i","nrhs"),
                            ("pm_io","A"),("d","descA"),("pm_io","B"),("d","descB")], {}),
    # ---------------- distributed: multiply / add / QR / misc ----------------
    ("pgemm", "all4", [("h","handle"),("c","transa"),("c","transb"),("i","m"),("i","n"),
                       ("i","k"),("s","alpha"),("pm_i","A"),("d","descA"),("pm_i","B"),
                       ("d","descB"),("s","beta"),("pm_io","C"),("d","descC")], {}),
    ("pgeqrf", "all4", [("h","handle"),("c","uplo"),("i","m"),("i","n"),("pm_io","A"),
                        ("d","descA"),("pt","tau"),("ip","info")], {}),
    ("pgeadd", "all4", [("h","handle"),("c","transa"),("c","transb"),("i","m"),("i","n"),
                        ("s","alpha"),("pm_i","A"),("i","ia"),("i","ja"),("d","descA"),
                        ("s","beta"),("pm_i","B"),("i","ib"),("i","jb"),("d","descB"),
                        ("pm_io","C"),("i","ic"),("i","jc"),("d","descC")], {}),
    ("pdam", "all4", [("h","handle"),("s","alpha"),("pm_io","A"),("d","descA"),("i","n")], {}),
    # ---------------- distributed: Cholesky family ----------------
    ("ppotrf", "all4", [("h","handle"),("c","uplo"),("i","n"),("pm_io","A"),("i","ia"),
                        ("i","ja"),("d","descA"),("ip","info"),("ip","is_nega")],
     {"ret_bool": "is_nega", "drop_args": ["is_head", "location"]}),
    ("potrf_bottom_right", "all4", [("h","handle"),("c","uplo"),("i","n"),("pm_io","A"),
                                    ("i","lda"),("ip","info")],
     {"cpp_order": [1, 2, 3, 4, 5, 0]}),   # C++ API takes the handle LAST
    ("ppotrf_bottom_right", "all4", [("h","handle"),("c","uplo"),("i","n"),("pm_io","A"),
                                     ("d","descA"),("ip","info")], {}),
    ("ppotrs", "all4", [("h","handle"),("c","side"),("c","uplo"),("c","trans"),("i","n"),
                        ("i","nrhs"),("pm_io","A"),("d","descA"),("pm_io","B"),("d","descB"),
                        ("b","is_nega"),("b","location")], {}),
    ("pposv", "all4", [("h","handle"),("c","side"),("c","uplo"),("c","trans"),("i","n"),
                       ("i","nrhs"),("pm_io","A"),("i","ia"),("i","ja"),("d","descA"),
                       ("pm_io","B"),("i","ib"),("i","jb"),("d","descB"),("ip","info")],
     {"drop_args": ["is_head", "location"]}),
    # ---------------- local BLAS ----------------
    ("gemm", "all4", [("h","handle"),("c","transa"),("c","transb"),("i","m"),("i","n"),
                      ("i","k"),("s","alpha"),("pm_i","A"),("i","lda"),("pm_i","B"),
                      ("i","ldb"),("s","beta"),("pm_io","C"),("i","ldc")], {}),
    ("gemv", "all4", [("h","handle"),("c","trans"),("i","m"),("i","n"),("s","alpha"),
                      ("pm_i","A"),("i","lda"),("pm_i","x"),("i","incx"),("s","beta"),
                      ("pm_io","y"),("i","incy")], {}),
    ("trmv", "all4", [("h","handle"),("c","uplo"),("c","trans"),("c","diag"),("i","n"),
                      ("pm_i","A"),("i","lda"),("pm_io","x"),("i","incx")], {}),
    ("trsm", "all4", [("h","handle"),("c","side"),("c","uplo"),("c","trans"),("c","diag"),
                      ("i","m"),("i","n"),("s","alpha"),("pm_io","A"),("i","lda"),
                      ("pm_io","B"),("i","ldb")], {}),
    ("trmm", "all4", [("h","handle"),("c","side"),("c","uplo"),("c","trans"),("c","diag"),
                      ("i","m"),("i","n"),("s","alpha"),("pm_i","A"),("i","lda"),
                      ("pm_io","B"),("i","ldb")], {}),
    ("syrk", "sd", [("h","handle"),("c","uplo"),("c","trans"),("i","n"),("i","k"),
                    ("s","alpha"),("pm_i","A"),("i","lda"),("s","beta"),("pm_io","C"),
                    ("i","ldc")], {}),
    ("trtrm", "all4", [("h","handle"),("c","uplo"),("i","n"),("s","alpha"),("pm_i","A"),
                       ("i","lda"),("pm_i","B"),("i","ldb"),("s","beta"),("pm_io","C"),
                       ("i","ldc")], {}),
    ("geam", "all4", [("h","handle"),("c","transa"),("c","transb"),("i","m"),("i","n"),
                      ("s","alpha"),("pm_i","A"),("i","lda"),("s","beta"),("pm_i","B"),
                      ("i","ldb"),("pm_io","C"),("i","ldc")], {}),
    ("herk", "all4", [("h","handle"),("c","uplo"),("c","trans"),("i","n"),("i","k"),
                      ("sr","alpha"),("pm_i","A"),("i","lda"),("sr","beta"),("pm_io","C"),
                      ("i","ldc")], {}),
    ("her2k", "all4", [("h","handle"),("c","uplo"),("c","trans"),("i","n"),("i","k"),
                       ("s","alpha"),("pm_i","A"),("i","lda"),("pm_i","B"),("i","ldb"),
                       ("sr","beta"),("pm_io","C"),("i","ldc")], {}),
    ("hemv", "all4", [("h","handle"),("c","uplo"),("i","n"),("s","alpha"),("pm_i","A"),
                      ("i","lda"),("pm_i","x"),("i","incx"),("s","beta"),("pm_io","y"),
                      ("i","incy")], {}),
    ("hemm", "all4", [("h","handle"),("c","side"),("c","uplo"),("i","m"),("i","n"),
                      ("s","alpha"),("pm_i","A"),("i","lda"),("pm_i","B"),("i","ldb"),
                      ("s","beta"),("pm_io","C"),("i","ldc")], {}),
    ("her2", "all4", [("h","handle"),("c","uplo"),("i","n"),("s","alpha"),("pm_i","x"),
                      ("i","incx"),("pm_i","y"),("i","incy"),("pm_io","A"),("i","lda")], {}),
    # ---------------- local BLAS-1 ----------------
    ("scal", "all4", [("h","handle"),("i","n"),("s","alpha"),("pm_io","x"),("i","incx")], {}),
    ("axpy", "all4", [("h","handle"),("i","n"),("s","alpha"),("pm_i","x"),("i","incx"),
                      ("pm_io","y"),("i","incy")], {}),
    ("swap", "all4", [("h","handle"),("i","n"),("pm_io","x"),("i","incx"),("pm_io","y"),
                      ("i","incy")], {}),
    ("geru", "all4", [("h","handle"),("i","m"),("i","n"),("s","alpha"),("pm_i","x"),
                      ("i","incx"),("pm_i","y"),("i","incy"),("pm_io","A"),("i","lda")], {}),
    ("gerc", "all4", [("h","handle"),("i","m"),("i","n"),("s","alpha"),("pm_i","x"),
                      ("i","incx"),("pm_i","y"),("i","incy"),("pm_io","A"),("i","lda")], {}),
    ("iamax", "all4", [("h","handle"),("i","n"),("pm_i","x"),("i","incx"),("ip","result")], {}),
    ("nrm2", "all4", [("h","handle"),("i","n"),("pm_i","x"),("i","incx"),("pr_io","result")], {}),
    ("dotc", "all4", [("h","handle"),("i","n"),("pm_i","x"),("i","incx"),("pm_i","y"),
                      ("i","incy"),("pm_io","result")], {}),
    # ---------------- local LAPACK wrappers ----------------
    ("getrf", "all4", [("h","handle"),("i","m"),("i","n"),("pm_io","A"),("i","lda"),
                       ("pi_io","ipiv"),("pi_io","info")], {}),
    ("potrf", "all4", [("h","handle"),("c","uplo"),("i","n"),("pm_io","A"),("i","lda"),
                       ("pi_io","info")], {}),
    ("laswp", "all4", [("h","handle"),("i","n"),("pm_io","A"),("i","lda"),("i","k1"),
                       ("i","k2"),("pi_i","ipiv"),("i","incx")], {}),
    ("geqrf", "all4", [("h","handle"),("c","uplo"),("i","m"),("i","n"),("pm_io","A"),
                       ("i","lda"),("pm_io","tau")], {}),
    ("larft", "all4", [("h","handle"),("c","direct"),("c","storev"),("i","n"),("i","k"),
                       ("pm_io","V"),("i","ldv"),("pm_i","tau"),("pm_io","Tmat"),
                       ("i","ldt")], {}),
]

STATUS_ENUM = """enum {
    DDLA_C_STATUS_SUCCESS = 0,
    DDLA_C_STATUS_INVALID_HANDLE = 1,     // null handle
    DDLA_C_STATUS_INVALID_VALUE = 2,      // bad argument (mirrors ddlaStatus_t)
    DDLA_C_STATUS_ALLOC_FAILED = 3,
    DDLA_C_STATUS_BACKEND_UNAVAILABLE = 4,// backend not compiled into this build
    DDLA_C_STATUS_INTERNAL_ERROR = 5,     // vendor / device failure (mirrors ddlaStatus_t)
    DDLA_C_STATUS_EXCEPTION = 6           // a C++ exception escaped the library call
};"""

PROLOGUE = r"""// ---------------------------------------------------------------------------
// ddla_c_api.cpp -- C ABI over the LibDDLA C++ API, the base of the Fortran
// bindings (fortran/ddla.f90).  GENERATED by fortran/gen_bindings.py from the
// signature table there; edit that table, not this file.
//
// Conventions:
//   * Every function returns an int status (DDLA_C_STATUS_* below); 0 is
//     success.  A C++ exception escaping the library call is caught and
//     reported as DDLA_C_STATUS_EXCEPTION with its what() text retrievable
//     from ddla_c_error_string(DDLA_C_STATUS_EXCEPTION).
//   * The handle travels as void* (the C++ DdlaHandle_t is an opaque
//     DdlaStream*).  ddla_c_init / ddla_c_destroy hand the pointer itself
//     back and forth through void**.
//   * Integers are C int (32-bit, the ScaLAPACK convention), sizes long long.
//     Character flags travel by value.
//   * Scalar arguments are passed by reference: real ones as const RT*,
//     complex ones as const ddla_c_float_complex* / ddla_c_double_complex*
//     (a two-field {re, im} struct with the layout of the Fortran
//     complex(c_float) / complex(c_double) dummy of the same intent).
//   * Matrices and vectors are void*: host memory for a CPU handle, device
//     memory for a GPU handle, exactly as the C++ API documents each argument.
//   * The backend is taken from the handle at run time (ddlaGetBackend) and
//     dispatched to the matching template instantiation; requesting a backend
//     this build does not contain reports DDLA_C_STATUS_BACKEND_UNAVAILABLE.
//
// The MPI communicator of ddla_c_set arrives as the Fortran handle and is
// converted with MPI_Comm_f2c.
// ---------------------------------------------------------------------------
#include <ddla/ddla.h>

#include <mpi.h>

#include <exception>
#include <string>
#include <type_traits>

// Two-field complex scalars; ddla.f90 declares the same layouts as
// type, bind(C) :: ddla_complex_float / ddla_complex_double.
typedef struct { float re, im; } ddla_c_float_complex;
typedef struct { double re, im; } ddla_c_double_complex;

""" + STATUS_ENUM + r"""

namespace {

thread_local std::string g_ddla_c_last_error;

void ddla_c_record_error(const char* what)
{
    try { g_ddla_c_last_error = what ? what : ""; }
    catch (...) { g_ddla_c_last_error.clear(); }
}

// Run one library call on the handle's backend.  body is a template lambda
// invoked with std::integral_constant<DdlaBackend, ...>; only the branches
// this build actually contains are compiled (DDLA_HAS_CPU / DDLA_HAS_GPU from
// ddla_config.h), so every lane -- CPU-only, GPU-only, dual -- links.
template <typename F>
int ddla_c_run(void* handle, F&& body)
{
    if (handle == nullptr)
        return DDLA_C_STATUS_INVALID_HANDLE;
    try {
        const ddla::DdlaHandle_t hh = static_cast<ddla::DdlaHandle_t>(handle);
        switch (ddla::ddlaGetBackend(hh)) {
        case ddla::DdlaBackend::CPU:
#if DDLA_HAS_CPU
            return body(std::integral_constant<ddla::DdlaBackend, ddla::DdlaBackend::CPU>{});
#else
            return DDLA_C_STATUS_BACKEND_UNAVAILABLE;
#endif
        case ddla::DdlaBackend::GPU:
#if DDLA_HAS_GPU
            return body(std::integral_constant<ddla::DdlaBackend, ddla::DdlaBackend::GPU>{});
#else
            return DDLA_C_STATUS_BACKEND_UNAVAILABLE;
#endif
        default:
            return DDLA_C_STATUS_INVALID_VALUE;
        }
    } catch (const std::exception& e) {
        ddla_c_record_error(e.what());
        return DDLA_C_STATUS_EXCEPTION;
    } catch (...) {
        ddla_c_record_error("unknown C++ exception in a LibDDLA C API call");
        return DDLA_C_STATUS_EXCEPTION;
    }
}

} // namespace

extern "C" {

const char* ddla_c_error_string(int status)
{
    if (status == DDLA_C_STATUS_EXCEPTION && !g_ddla_c_last_error.empty())
        return g_ddla_c_last_error.c_str();
    if (status >= 0 && status <= 5)
        return ddla::ddlaGetErrorString(static_cast<ddla::ddlaStatus_t>(status));
    return "unknown LibDDLA C API status";
}

int ddla_c_init(void** handle)
{
    try {
        ddla::DdlaHandle_t h = nullptr;
        const ddla::ddlaStatus_t st = ddla::ddlaInit(h);
        if (st != ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS) return static_cast<int>(st);
        *handle = static_cast<void*>(h);
        return DDLA_C_STATUS_SUCCESS;
    } catch (const std::exception& e) { ddla_c_record_error(e.what()); return DDLA_C_STATUS_EXCEPTION; }
    catch (...) { ddla_c_record_error("unknown C++ exception in ddla_c_init"); return DDLA_C_STATUS_EXCEPTION; }
}

int ddla_c_init_backend(void** handle, int backend)
{
    try {
        if (backend < 0 || backend > 1) return DDLA_C_STATUS_INVALID_VALUE;
        const ddla::DdlaBackend be = (backend == 0) ? ddla::DdlaBackend::CPU : ddla::DdlaBackend::GPU;
        if (!ddla::ddlaBackendAvailable(be)) return DDLA_C_STATUS_BACKEND_UNAVAILABLE;
        ddla::DdlaHandle_t h = nullptr;
        const ddla::ddlaStatus_t st = ddla::ddlaInit(h, be);
        if (st != ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS) return static_cast<int>(st);
        *handle = static_cast<void*>(h);
        return DDLA_C_STATUS_SUCCESS;
    } catch (const std::exception& e) { ddla_c_record_error(e.what()); return DDLA_C_STATUS_EXCEPTION; }
    catch (...) { ddla_c_record_error("unknown C++ exception in ddla_c_init_backend"); return DDLA_C_STATUS_EXCEPTION; }
}

int ddla_c_set(void* handle, int f_comm, int nprows, int npcols, char major)
{
    try {
        if (handle == nullptr) return DDLA_C_STATUS_INVALID_HANDLE;
        ddla::DdlaHandle_t hh = static_cast<ddla::DdlaHandle_t>(handle);
        const MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(f_comm));
        ddla::ddlaStatus_t st;
        if (nprows > 0 && npcols > 0)
            st = ddla::ddlaSet(hh, comm, nprows, npcols, major);
        else
            st = ddla::ddlaSet(hh, comm, major);
        if (st != ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS) return static_cast<int>(st);
        return DDLA_C_STATUS_SUCCESS;
    } catch (const std::exception& e) { ddla_c_record_error(e.what()); return DDLA_C_STATUS_EXCEPTION; }
    catch (...) { ddla_c_record_error("unknown C++ exception in ddla_c_set"); return DDLA_C_STATUS_EXCEPTION; }
}

int ddla_c_destroy(void** handle)
{
    try {
        if (handle == nullptr || *handle == nullptr) return DDLA_C_STATUS_SUCCESS;
        ddla::DdlaHandle_t h = static_cast<ddla::DdlaHandle_t>(*handle);
        const ddla::ddlaStatus_t st = ddla::ddlaDestroy(h);
        *handle = static_cast<void*>(h);
        if (st != ddla::ddlaStatus_t::DDLA_STATUS_SUCCESS) return static_cast<int>(st);
        return DDLA_C_STATUS_SUCCESS;
    } catch (const std::exception& e) { ddla_c_record_error(e.what()); return DDLA_C_STATUS_EXCEPTION; }
    catch (...) { ddla_c_record_error("unknown C++ exception in ddla_c_destroy"); return DDLA_C_STATUS_EXCEPTION; }
}

int ddla_c_backend_available(int backend)
{
    if (backend < 0 || backend > 1) return 0;
    return ddla::ddlaBackendAvailable(backend == 0 ? ddla::DdlaBackend::CPU : ddla::DdlaBackend::GPU) ? 1 : 0;
}

int ddla_c_get_backend(void* handle)
{
    if (handle == nullptr) return -DDLA_C_STATUS_INVALID_HANDLE;
    return static_cast<int>(ddla::ddlaGetBackend(static_cast<ddla::DdlaHandle_t>(handle)));
}

int ddla_c_grid_dims(void* handle, int* nprows, int* npcols)
{
    if (handle == nullptr) return DDLA_C_STATUS_INVALID_HANDLE;
    ddla::ddlaGetGridDims(static_cast<ddla::DdlaHandle_t>(handle), *nprows, *npcols);
    return DDLA_C_STATUS_SUCCESS;
}

int ddla_c_grid_coords(void* handle, int* myprow, int* mypcol)
{
    if (handle == nullptr) return DDLA_C_STATUS_INVALID_HANDLE;
    ddla::ddlaGetGridCoords(static_cast<ddla::DdlaHandle_t>(handle), *myprow, *mypcol);
    return DDLA_C_STATUS_SUCCESS;
}

int ddla_c_rank(void* handle)
{
    if (handle == nullptr) return -1;
    return ddla::ddlaGetRank(static_cast<ddla::DdlaHandle_t>(handle));
}

int ddla_c_size(void* handle)
{
    if (handle == nullptr) return -1;
    return ddla::ddlaGetSize(static_cast<ddla::DdlaHandle_t>(handle));
}

int ddla_c_descinit(int* desc, void* handle, int m, int n, int mb, int nb,
                    int irsrc, int icsrc, int lld)
{
    if (handle == nullptr) return DDLA_C_STATUS_INVALID_HANDLE;
    return static_cast<int>(ddla::ddlaDescInit(desc, static_cast<ddla::DdlaHandle_t>(handle),
                                               m, n, mb, nb, irsrc, icsrc, lld));
}

int ddla_c_numloc(int n, int nb, int iproc, int srcproc, int nprocs)
{
    return ddla::num_loc(n, nb, iproc, srcproc, nprocs);
}

int ddla_c_malloc(void* handle, void** ptr, long long bytes)
{
    if (handle == nullptr) return DDLA_C_STATUS_INVALID_HANDLE;
    return static_cast<int>(ddla::ddlaMalloc(static_cast<ddla::DdlaHandle_t>(handle), ptr,
                                             static_cast<std::size_t>(bytes)));
}

int ddla_c_free(void* handle, void* ptr)
{
    if (handle == nullptr) return DDLA_C_STATUS_INVALID_HANDLE;
    return static_cast<int>(ddla::ddlaFree(static_cast<ddla::DdlaHandle_t>(handle), ptr));
}

int ddla_c_memcpy(void* handle, void* dst, const void* src, long long bytes)
{
    if (handle == nullptr) return DDLA_C_STATUS_INVALID_HANDLE;
    return static_cast<int>(ddla::ddlaMemcpy(static_cast<ddla::DdlaHandle_t>(handle), dst, src,
                                             static_cast<std::size_t>(bytes)));
}

int ddla_c_memcpy2d(void* handle, void* dst, long long dpitch, const void* src, long long spitch,
                    long long width, long long height)
{
    if (handle == nullptr) return DDLA_C_STATUS_INVALID_HANDLE;
    return static_cast<int>(ddla::ddlaMemcpy2D(static_cast<ddla::DdlaHandle_t>(handle), dst,
                                               static_cast<std::size_t>(dpitch), src,
                                               static_cast<std::size_t>(spitch),
                                               static_cast<std::size_t>(width),
                                               static_cast<std::size_t>(height)));
}

int ddla_c_memset(void* handle, void* ptr, int value, long long bytes)
{
    if (handle == nullptr) return DDLA_C_STATUS_INVALID_HANDLE;
    return static_cast<int>(ddla::ddlaMemset(static_cast<ddla::DdlaHandle_t>(handle), ptr, value,
                                             static_cast<std::size_t>(bytes)));
}

int ddla_c_synchronize(void* handle)
{
    if (handle == nullptr) return DDLA_C_STATUS_INVALID_HANDLE;
    return static_cast<int>(ddla::ddlaSynchronize(static_cast<ddla::DdlaHandle_t>(handle)));
}
static bool g_mpi_init_by_ddla = false;
static void ddla_c_finalize_on_exit()
{
    int is_init = 0, is_fin = 0;
    MPI_Initialized(&is_init);
    MPI_Finalized(&is_fin);
    if (is_init && !is_fin && g_mpi_init_by_ddla) {
        MPI_Finalize();
    }
}

static void ensure_mpi_init()
{
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (!is_init) {
        MPI_Init(nullptr, nullptr);
        g_mpi_init_by_ddla = true;
        std::atexit(ddla_c_finalize_on_exit);
    }
}

int ddla_c_default_comm()
{
    ensure_mpi_init();
    return static_cast<int>(MPI_Comm_c2f(MPI_COMM_WORLD));
}

int ddla_c_comm_rank(int f_comm)
{
    ensure_mpi_init();
    MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(f_comm));
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

int ddla_c_comm_size(int f_comm)
{
    ensure_mpi_init();
    MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(f_comm));
    int size = 1;
    MPI_Comm_size(comm, &size);
    return size;
}

void ddla_c_finalize()
{
    int is_init = 0, is_fin = 0;
    MPI_Initialized(&is_init);
    MPI_Finalized(&is_fin);
    if (is_init && !is_fin) {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
    }
}

int ddla_c_bcast_dims(int f_comm, int* m, int* n, int* type_char, int root)
{
    MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(f_comm));
    int buf[3] = {*m, *n, *type_char};
    MPI_Bcast(buf, 3, MPI_INT, root, comm);
    *m = buf[0];
    *n = buf[1];
    *type_char = buf[2];
    return 0;
}
"""


def cpp_scalar_type(t):
    return t[1]


def cpp_real_type(t):
    # C++ real type underlying the scalar type (float for s/c, double for d/z)
    return "float" if t[0] in ("s", "c") else "double"


def f_scalar_decl(kind, name, t):
    fk = t[2]
    if not t[3]:
        return "real(%s), intent(in) :: %s" % (fk, name)
    return "type(ddla_complex_%s), intent(in) :: %s" % ("float" if t[0] == "c" else "double", name)


def f_dummy(kind, name, t):
    fk = t[2]
    if kind == "h":
        return "type(c_ptr), value :: %s" % name
    if kind == "c":
        return "character(kind=c_char), value :: %s" % name
    if kind == "i":
        return "integer(c_int), value :: %s" % name
    if kind == "i64":
        return "integer(c_int64_t), value :: %s" % name
    if kind == "s":
        return f_scalar_decl(kind, name, t)
    if kind == "sr":
        return "real(%s), intent(in) :: %s" % (t[4], name)
    if kind in ("pm_io", "pm_i", "pi_io", "pi_i", "pwork", "pr_io"):
        return "type(c_ptr), value :: %s" % name
    if kind == "ph_in":
        return "integer(c_int), intent(in) :: %s(*)" % name
    if kind == "ph_io":
        return "integer(c_int), intent(inout) :: %s(*)" % name
    if kind == "pt":
        if t[3]:
            return "complex(%s), intent(out) :: %s(*)" % (fk, name)
        return "real(%s), intent(out) :: %s(*)" % (fk, name)
    if kind == "d":
        return "integer(c_int), intent(in) :: %s(9)" % name
    if kind == "ip":
        return "integer(c_int), intent(out) :: %s" % name
    if kind == "b":
        return "integer(c_int), value :: %s" % name
    raise ValueError(kind)


def cpp_param(kind, name, t):
    if kind == "h":
        return "void* %s" % name
    if kind == "c":
        return "char %s" % name
    if kind == "i":
        return "int %s" % name
    if kind == "i64":
        return "long long %s" % name
    if kind == "s":
        if t[3]:
            return "const ddla_c_%s_complex* %s" % ("float" if t[0] == "c" else "double", name)
        return "const %s* %s" % (t[1], name)
    if kind == "sr":
        return "const %s* %s" % ("float" if t[0] in ("s", "c") else "double", name)
    if kind in ("pm_io", "pwork", "pt", "pi_io", "pi_i", "pr_io"):
        return "void* %s" % name
    if kind == "pm_i":
        return "const void* %s" % name
    if kind in ("ph_in",):
        return "const int* %s" % name
    if kind == "ph_io":
        return "int* %s" % name
    if kind == "d":
        return "const int* %s" % name
    if kind == "ip":
        return "int* %s" % name
    if kind == "b":
        return "int %s" % name
    raise ValueError(kind)


def cpp_expr(kind, name, t):
    ST = t[1]
    if kind == "h":
        return "hh"
    if kind in ("c", "i", "i64", "b"):
        return name
    if kind == "s":
        return "*%s" % name if not t[3] else "%s_v" % name
    if kind == "sr":
        return "*%s" % name
    if kind == "pm_io":
        return "static_cast<%s*>(%s)" % (ST, name)
    if kind == "pm_i":
        return "static_cast<const %s*>(%s)" % (ST, name)
    if kind in ("pwork", "pi_io"):
        return "static_cast<int*>(%s)" % name
    if kind == "pi_i":
        return "static_cast<const int*>(%s)" % name
    if kind == "pt":
        return "static_cast<%s*>(%s)" % (ST, name)
    if kind == "pr_io":
        return "static_cast<%s*>(%s)" % ("float" if t[0] in ("s", "c") else "double", name)
    if kind in ("ph_in", "ph_io", "d"):
        return name
    if kind == "ip":
        return "*%s" % name
    raise ValueError(kind)


def gen_cpp():
    out = [PROLOGUE]
    for name, tspec, args, opts in FUNCS:
        types = TYPES if tspec == "all4" else TYPES[:2]
        for t in types:
            sfx, ST = t[0], t[1]
            cname = "ddla_%s_%s" % (sfx, name)
            sig_args = args
            if "drop_args" in opts:
                sig_args = [a for a in args if a[1] not in opts["drop_args"]]
            params = ", ".join(cpp_param(k, n, t) for k, n in sig_args)
            decls = []
            if any(k == "s" and t[3] for k, n in sig_args):
                RT = cpp_real_type(t)
                for k, n in sig_args:
                    if k == "s" and t[3]:
                        decls.append("        const std::complex<%s> %s_v(%s->re, %s->im);" % (RT, n, n, n))
            call_args_spec = sig_args
            if "cpp_order" in opts:
                call_args_spec = [sig_args[i] for i in opts["cpp_order"]]
            call_args = ", ".join(cpp_expr(k, n, t) for k, n in call_args_spec)
            lines = []
            lines.append("")
            lines.append("extern \"C\" int %s(%s)" % (cname, params))
            lines.append("{")
            lines.append("    return ddla_c_run(handle, [&](auto be) -> int {")
            lines.append("        using BE = decltype(be);")
            lines.append("        const ddla::DdlaHandle_t hh = static_cast<ddla::DdlaHandle_t>(handle);")
            lines.extend(decls)
            if "ret_bool" in opts:
                lines.append("        const bool nega = ddla::%s<BE::value, %s>(%s);" % (name, ST, call_args))
                rn = opts["ret_bool"]
                lines.append("        if (%s != nullptr) *%s = nega ? 1 : 0;" % (rn, rn))
            else:
                lines.append("        ddla::%s<BE::value, %s>(%s);" % (name, ST, call_args))
            lines.append("        return DDLA_C_STATUS_SUCCESS;")
            lines.append("    });")
            lines.append("}")
            out.append("\n".join(lines))
    out.append("\n} // extern \"C\"\n")
    return "\n".join(out)


def wrap_fortran_lines(prefix, names, cont_char="&"):
    """Emit 'prefix n1, n2, ...' wrapped at ~100 columns with trailing &."""
    lines = []
    cur = prefix
    for i, nm in enumerate(names):
        piece = nm + ("," if i < len(names) - 1 else "")
        if len(cur) + len(piece) + 1 > 100:
            lines.append(cur.rstrip() + " &")
            cur = "     "  # continuation indent
        cur += piece + (" " if piece != names[-1] else "")
    lines.append(cur.rstrip())
    return lines


def gen_f90():
    out = []
    out.append("""! ---------------------------------------------------------------------------
! ddla.f90 -- Fortran bindings for LibDDLA.  GENERATED by
! fortran/gen_bindings.py from the signature table there; edit that table, not
! this file.
!
! Layering: every binding is a bind(C) interface of the C-ABI shim in
! fortran/ddla_c_api.cpp.  All of them return integer(c_int) status, 0 =
! success; ddla_check(status) aborts with ddla_error_text(status) on failure.
!
! Naming: LAPACK-style type-prefixed names -- ddla_s_*/ddla_d_*/ddla_c_*/ddla_z_*
! take float / double / complex(c_float) / complex(c_double) data.  There are
! deliberately no generic names: most routines pass their matrices as
! type(c_ptr), which a Fortran generic cannot resolve by scalar type.
!
! Argument conventions:
!   * handle       type(c_ptr), value -- from ddla_init / ddla_init_backend.
!   * matrices     type(c_ptr), value -- host memory for a CPU handle, device
!                  memory for a GPU handle, exactly where the C++ API puts
!                  them.  Pass c_loc(fortran_array) on a CPU handle; on a GPU
!                  handle pass the type(c_ptr) ddla_malloc returned and move
!                  data with ddla_memcpy / ddla_memcpy2d.
!   * characters   by value ('N', 'T', 'U', 'L', ...).
!   * integers     by value; descriptors integer(c_int) :: desc(9) by
!                  reference, filled by ddla_descinit (ScaLAPACK DESCINIT
!                  layout; CTXT_ is ignored, the grid comes from the handle).
!   * scalars      by reference: real(c_float/c_double) for real types, the
!                  ddla_complex_float / ddla_complex_double derived types for
!                  complex ones (a structure constructor builds them).
!   * exceptions   a C++ exception from the library surfaces as status
!                  DDLA_C_STATUS_EXCEPTION; ddla_error_text returns its text.
! ---------------------------------------------------------------------------
module ddla
  use, intrinsic :: iso_c_binding, only: c_int, c_char, c_ptr, c_float, c_double, &
                                         c_int64_t, c_null_ptr, c_null_char, &
                                         c_f_pointer, c_associated
  implicit none
  private
""")
    # publics
    typeless = ["ddla_init", "ddla_init_backend", "ddla_set", "ddla_destroy",
                "ddla_backend_available", "ddla_get_backend", "ddla_grid_dims",
                "ddla_grid_coords", "ddla_rank", "ddla_size", "ddla_descinit",
                "ddla_numloc", "ddla_malloc", "ddla_free", "ddla_memcpy",
                "ddla_memcpy2d", "ddla_memset", "ddla_synchronize",
                "ddla_error_string", "ddla_error_text", "ddla_check",
                "ddla_complex_float", "ddla_complex_double",
                "DDLA_BACKEND_CPU", "DDLA_BACKEND_GPU",
                "DDLA_STATUS_SUCCESS", "DDLA_STATUS_INVALID_HANDLE",
                "DDLA_STATUS_INVALID_VALUE", "DDLA_STATUS_ALLOC_FAILED",
                "DDLA_STATUS_BACKEND_UNAVAILABLE", "DDLA_STATUS_INTERNAL_ERROR",
                "DDLA_STATUS_EXCEPTION"]
    bound = []
    for name, tspec, args, opts in FUNCS:
        types = ["s", "d", "c", "z"] if tspec == "all4" else ["s", "d"]
        bound.extend("ddla_%s_%s" % (s, name) for s in types)
    out.extend(wrap_fortran_lines("  public :: ", typeless))
    out.extend(wrap_fortran_lines("  public :: ", bound))
    out.append("""
  ! Backend ids for ddla_init_backend / ddla_backend_available / ddla_get_backend.
  integer(c_int), parameter :: DDLA_BACKEND_CPU = 0_c_int
  integer(c_int), parameter :: DDLA_BACKEND_GPU = 1_c_int

  ! Status codes returned by every binding (mirror ddlaStatus_t plus one).
  integer(c_int), parameter :: DDLA_STATUS_SUCCESS = 0_c_int
  integer(c_int), parameter :: DDLA_STATUS_INVALID_HANDLE = 1_c_int
  integer(c_int), parameter :: DDLA_STATUS_INVALID_VALUE = 2_c_int
  integer(c_int), parameter :: DDLA_STATUS_ALLOC_FAILED = 3_c_int
  integer(c_int), parameter :: DDLA_STATUS_BACKEND_UNAVAILABLE = 4_c_int
  integer(c_int), parameter :: DDLA_STATUS_INTERNAL_ERROR = 5_c_int
  integer(c_int), parameter :: DDLA_STATUS_EXCEPTION = 6_c_int

  ! Complex scalars, passed by reference; layout-identical to
  ! std::complex<float> / std::complex<double> and to Fortran
  ! complex(c_float) / complex(c_double) storage.
  type, bind(C) :: ddla_complex_float
    real(c_float) :: re, im
  end type ddla_complex_float

  type, bind(C) :: ddla_complex_double
    real(c_double) :: re, im
  end type ddla_complex_double

  interface
""")
    # interface bodies
    for name, tspec, args, opts in FUNCS:
        types = TYPES if tspec == "all4" else TYPES[:2]
        for t in types:
            sfx = t[0]
            cname = "ddla_%s_%s" % (sfx, name)
            sig_args = args
            if "drop_args" in opts:
                sig_args = [a for a in args if a[1] not in opts["drop_args"]]
            dummies = [n.lower() for k, n in sig_args]
            arglist = ", ".join(dummies)
            lines = []
            lines.append("    function %s(%s) &" % (cname, arglist))
            lines.append('        result(status) bind(C, name="%s")' % cname)
            lines.append("      import :: c_int, c_char, c_ptr, c_float, c_double, c_int64_t, ddla_complex_float, ddla_complex_double")
            lines.append("      implicit none")
            lines.append("      integer(c_int) :: status")
            for (k, n), dn in zip(sig_args, dummies):
                lines.append("      " + f_dummy(k, dn, t))
            lines.append("    end function %s" % cname)
            lines.append("")
            out.extend(lines)
    # type-less lifecycle/memory interfaces
    out.append("""    function ddla_init(handle) &
        result(status) bind(C, name="ddla_c_init")
      import :: c_int, c_ptr
      implicit none
      integer(c_int) :: status
      type(c_ptr), intent(out) :: handle
    end function ddla_init

    function ddla_init_backend(handle, backend) &
        result(status) bind(C, name="ddla_c_init_backend")
      import :: c_int, c_ptr
      implicit none
      integer(c_int) :: status
      type(c_ptr), intent(out) :: handle
      integer(c_int), value :: backend
    end function ddla_init_backend

    function ddla_set(handle, fcomm, nprows, npcols, major) &
        result(status) bind(C, name="ddla_c_set")
      import :: c_int, c_char, c_ptr
      implicit none
      integer(c_int) :: status
      type(c_ptr), value :: handle
      integer(c_int), value :: fcomm
      integer(c_int), value :: nprows, npcols
      character(kind=c_char), value :: major
    end function ddla_set

    function ddla_destroy(handle) &
        result(status) bind(C, name="ddla_c_destroy")
      import :: c_int, c_ptr
      implicit none
      integer(c_int) :: status
      type(c_ptr), intent(inout) :: handle
    end function ddla_destroy

    function ddla_backend_available(backend) &
        result(avail) bind(C, name="ddla_c_backend_available")
      import :: c_int
      implicit none
      integer(c_int) :: avail
      integer(c_int), value :: backend
    end function ddla_backend_available

    function ddla_get_backend(handle) &
        result(backend) bind(C, name="ddla_c_get_backend")
      import :: c_int, c_ptr
      implicit none
      integer(c_int) :: backend
      type(c_ptr), value :: handle
    end function ddla_get_backend

    function ddla_grid_dims(handle, nprows, npcols) &
        result(status) bind(C, name="ddla_c_grid_dims")
      import :: c_int, c_ptr
      implicit none
      integer(c_int) :: status
      type(c_ptr), value :: handle
      integer(c_int), intent(out) :: nprows, npcols
    end function ddla_grid_dims

    function ddla_grid_coords(handle, myprow, mypcol) &
        result(status) bind(C, name="ddla_c_grid_coords")
      import :: c_int, c_ptr
      implicit none
      integer(c_int) :: status
      type(c_ptr), value :: handle
      integer(c_int), intent(out) :: myprow, mypcol
    end function ddla_grid_coords

    function ddla_rank(handle) &
        result(rank) bind(C, name="ddla_c_rank")
      import :: c_int, c_ptr
      implicit none
      integer(c_int) :: rank
      type(c_ptr), value :: handle
    end function ddla_rank

    function ddla_size(handle) &
        result(nproc) bind(C, name="ddla_c_size")
      import :: c_int, c_ptr
      implicit none
      integer(c_int) :: nproc
      type(c_ptr), value :: handle
    end function ddla_size

    function ddla_descinit(desc, handle, m, n, mb, nb, irsrc, icsrc, lld) &
        result(status) bind(C, name="ddla_c_descinit")
      import :: c_int, c_ptr
      implicit none
      integer(c_int) :: status
      integer(c_int), intent(out) :: desc(9)
      type(c_ptr), value :: handle
      integer(c_int), value :: m, n, mb, nb, irsrc, icsrc, lld
    end function ddla_descinit

    function ddla_numloc(n, nb, iproc, srcproc, nprocs) &
        result(nloc) bind(C, name="ddla_c_numloc")
      import :: c_int
      implicit none
      integer(c_int) :: nloc
      integer(c_int), value :: n, nb, iproc, srcproc, nprocs
    end function ddla_numloc

    function ddla_malloc(handle, ptr, bytes) &
        result(status) bind(C, name="ddla_c_malloc")
      import :: c_int, c_ptr, c_int64_t
      implicit none
      integer(c_int) :: status
      type(c_ptr), value :: handle
      type(c_ptr), intent(out) :: ptr
      integer(c_int64_t), value :: bytes
    end function ddla_malloc

    function ddla_free(handle, ptr) &
        result(status) bind(C, name="ddla_c_free")
      import :: c_int, c_ptr
      implicit none
      integer(c_int) :: status
      type(c_ptr), value :: handle
      type(c_ptr), value :: ptr
    end function ddla_free

    function ddla_memcpy(handle, dst, src, bytes) &
        result(status) bind(C, name="ddla_c_memcpy")
      import :: c_int, c_ptr, c_int64_t
      implicit none
      integer(c_int) :: status
      type(c_ptr), value :: handle
      type(c_ptr), value :: dst, src
      integer(c_int64_t), value :: bytes
    end function ddla_memcpy

    function ddla_memcpy2d(handle, dst, dpitch, src, spitch, width, height) &
        result(status) bind(C, name="ddla_c_memcpy2d")
      import :: c_int, c_ptr, c_int64_t
      implicit none
      integer(c_int) :: status
      type(c_ptr), value :: handle
      type(c_ptr), value :: dst, src
      integer(c_int64_t), value :: dpitch, spitch, width, height
    end function ddla_memcpy2d

    function ddla_memset(handle, ptr, value, bytes) &
        result(status) bind(C, name="ddla_c_memset")
      import :: c_int, c_ptr, c_int64_t
      implicit none
      integer(c_int) :: status
      type(c_ptr), value :: handle
      type(c_ptr), value :: ptr
      integer(c_int), value :: value
      integer(c_int64_t), value :: bytes
    end function ddla_memset

    function ddla_synchronize(handle) &
        result(status) bind(C, name="ddla_c_synchronize")
      import :: c_int, c_ptr
      implicit none
      integer(c_int) :: status
      type(c_ptr), value :: handle
    end function ddla_synchronize

    function ddla_error_string(status) &
        result(p) bind(C, name="ddla_c_error_string")
      import :: c_int, c_ptr
      implicit none
      type(c_ptr) :: p
      integer(c_int), value :: status
    end function ddla_error_string

  end interface
""")
    out.append("""
contains

  ! Text of a status code; for DDLA_STATUS_EXCEPTION this is the what() of the
  ! C++ exception the library threw (kept until the next failing call on the
  ! thread).  The C string is copied out immediately, as its lifetime ends
  ! with the next call.
  function ddla_error_text(status) result(txt)
    integer(c_int), intent(in) :: status
    character(len=512) :: txt
    type(c_ptr) :: p
    character(kind=c_char), pointer :: s(:)
    integer :: i, nul
    p = ddla_error_string(status)
    if (.not. c_associated(p)) then
      txt = ""
      return
    end if
    call c_f_pointer(p, s, [512])
    nul = 0
    do i = 1, 512
      if (s(i) == c_null_char) then
        nul = i - 1
        exit
      end if
    end do
    if (nul == 0) nul = 512
    txt = ""
    do i = 1, nul
      txt(i:i) = s(i)
    end do
  end function ddla_error_text

  ! Abort with the library error text; call after every binding whose status
  ! matters, or handle the status yourself.
  subroutine ddla_check(status, where)
    integer(c_int), intent(in) :: status
    character(len=*), intent(in), optional :: where
    character(len=32) :: st
    if (status /= DDLA_STATUS_SUCCESS) then
      write (st, '(I0)') status
      if (present(where)) then
        write (*, '(A)') 'ddla_check: LibDDLA error ' // trim(st) // &
             ' after ' // trim(where) // ': ' // trim(ddla_error_text(status))
      else
        write (*, '(A)') 'ddla_check: LibDDLA error ' // trim(st) // &
             ': ' // trim(ddla_error_text(status))
      end if
      error stop 1
    end if
  end subroutine ddla_check

end module ddla
""")
    return "\n".join(out)



def gen_py():
    out = []
    out.append("""# ---------------------------------------------------------------------------
# _c_api.py -- ctypes binding layer over LibDDLA C ABI.
# GENERATED by fortran/gen_bindings.py from the FUNCS signature table.
# Edit that table, not this file.
# ---------------------------------------------------------------------------
import ctypes
import os
import sys
from typing import Optional, Tuple, List, Dict, Any

# Status codes (mirrors DDLA_C_STATUS_*)
STATUS_SUCCESS = 0
STATUS_INVALID_HANDLE = 1
STATUS_INVALID_VALUE = 2
STATUS_ALLOC_FAILED = 3
STATUS_BACKEND_UNAVAILABLE = 4
STATUS_INTERNAL_ERROR = 5
STATUS_EXCEPTION = 6

# Backend constants
BACKEND_CPU = 0
BACKEND_GPU = 1

# ScaLAPACK descriptor constants
DTYPE_ = 0
CTXT_ = 1
M_ = 2
N_ = 3
MB_ = 4
NB_ = 5
RSRC_ = 6
CSRC_ = 7
LLD_ = 8
DLEN_ = 9

class DdlaError(Exception):
    pass

class DdlaInvalidHandleError(DdlaError):
    pass

class DdlaInvalidValueError(DdlaError):
    pass

class DdlaAllocFailedError(DdlaError):
    pass

class DdlaBackendUnavailableError(DdlaError):
    pass

class DdlaInternalError(DdlaError):
    pass

class DdlaExceptionError(DdlaError):
    pass

class ddla_c_float_complex(ctypes.Structure):
    _fields_ = [("re", ctypes.c_float), ("im", ctypes.c_float)]

class ddla_c_double_complex(ctypes.Structure):
    _fields_ = [("re", ctypes.c_double), ("im", ctypes.c_double)]

def to_scalar(val, sfx: str):
    if sfx == 's':
        return ctypes.c_float(float(val))
    elif sfx == 'd':
        return ctypes.c_double(float(val))
    elif sfx == 'c':
        c = complex(val)
        return ddla_c_float_complex(float(c.real), float(c.imag))
    elif sfx == 'z':
        c = complex(val)
        return ddla_c_double_complex(float(c.real), float(c.imag))
    else:
        raise ValueError(f"Unknown type suffix: {sfx}")

def _load_library():
    candidates = []
    env_path = os.environ.get("DDLA_LIB_PATH")
    if env_path:
        candidates.append(env_path)
    env_dir = os.environ.get("DDLA_DIR")
    if env_dir:
        candidates.extend([
            os.path.join(env_dir, "lib", "libddla_c.so"),
            os.path.join(env_dir, "lib64", "libddla_c.so"),
            os.path.join(env_dir, "lib", "libddla_fortran.so"),
            os.path.join(env_dir, "lib64", "libddla_fortran.so"),
        ])
    ld_path = os.environ.get("LD_LIBRARY_PATH")
    if ld_path:
        for p in ld_path.split(":"):
            if p:
                candidates.extend([
                    os.path.join(p, "libddla_c.so"),
                    os.path.join(p, "libddla_fortran.so"),
                ])
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    candidates.extend([
        os.path.join(root, "build_src_reorg_cpu", "fortran", "libddla_c.so"),
        os.path.join(root, "build_src_reorg_dual", "fortran", "libddla_c.so"),
        os.path.join(root, "build_src_reorg_cuda", "fortran", "libddla_c.so"),
        os.path.join(root, "build_src_reorg_cpu", "src", "libddla_c.so"),
        os.path.join(root, "build_src_reorg_dual", "src", "libddla_c.so"),
        os.path.join(root, "build_src_reorg_cuda", "src", "libddla_c.so"),
        os.path.join(root, "build_cpu", "fortran", "libddla_c.so"),
        os.path.join(root, "build_cpu", "src", "libddla_c.so"),
        os.path.join(root, "build", "fortran", "libddla_c.so"),
        os.path.join(root, "build", "src", "libddla_c.so"),
        os.path.join(root, "build_src_reorg_cpu", "fortran", "libddla_fortran.so"),
        os.path.join(root, "build_src_reorg_dual", "fortran", "libddla_fortran.so"),
        os.path.join(root, "build", "fortran", "libddla_fortran.so"),
        os.path.join(root, "lib", "libddla_c.so"),
        os.path.join(root, "lib64", "libddla_c.so"),
        "libddla_c.so",
        "libddla_fortran.so",
    ])
    for cand in candidates:
        if os.path.exists(cand):
            try:
                return ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    for name in ("libddla_c.so", "libddla_fortran.so"):
        try:
            return ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
    raise OSError("Could not locate libddla_c.so or libddla_fortran.so. Please set DDLA_LIB_PATH.")

lib = _load_library()

lib.ddla_c_error_string.restype = ctypes.c_char_p
lib.ddla_c_error_string.argtypes = [ctypes.c_int]

def check_status(status: int, where: str = ""):
    if status == STATUS_SUCCESS:
        return
    msg = lib.ddla_c_error_string(status)
    msg_str = msg.decode("utf-8", errors="replace") if msg else ""
    ctx = f" after {where}" if where else ""
    if status == STATUS_INVALID_HANDLE:
        raise DdlaInvalidHandleError(f"LibDDLA invalid handle{ctx}: {msg_str}")
    elif status == STATUS_INVALID_VALUE:
        raise DdlaInvalidValueError(f"LibDDLA invalid value{ctx}: {msg_str}")
    elif status == STATUS_ALLOC_FAILED:
        raise DdlaAllocFailedError(f"LibDDLA memory allocation failed{ctx}: {msg_str}")
    elif status == STATUS_BACKEND_UNAVAILABLE:
        raise DdlaBackendUnavailableError(f"LibDDLA backend unavailable{ctx}: {msg_str}")
    elif status == STATUS_INTERNAL_ERROR:
        raise DdlaInternalError(f"LibDDLA internal/device error{ctx}: {msg_str}")
    elif status == STATUS_EXCEPTION:
        raise DdlaExceptionError(f"LibDDLA exception{ctx}: {msg_str}")
    else:
        raise DdlaError(f"LibDDLA error {status}{ctx}: {msg_str}")

ddla_c_init = lib.ddla_c_init
ddla_c_init.restype = ctypes.c_int
ddla_c_init.argtypes = [ctypes.c_void_p]

ddla_c_init_backend = lib.ddla_c_init_backend
ddla_c_init_backend.restype = ctypes.c_int
ddla_c_init_backend.argtypes = [ctypes.c_void_p, ctypes.c_int]

ddla_c_set = lib.ddla_c_set
ddla_c_set.restype = ctypes.c_int
ddla_c_set.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char]

ddla_c_destroy = lib.ddla_c_destroy
ddla_c_destroy.restype = ctypes.c_int
ddla_c_destroy.argtypes = [ctypes.c_void_p]

ddla_c_backend_available = lib.ddla_c_backend_available
ddla_c_backend_available.restype = ctypes.c_int
ddla_c_backend_available.argtypes = [ctypes.c_int]

ddla_c_get_backend = lib.ddla_c_get_backend
ddla_c_get_backend.restype = ctypes.c_int
ddla_c_get_backend.argtypes = [ctypes.c_void_p]

ddla_c_grid_dims = lib.ddla_c_grid_dims
ddla_c_grid_dims.restype = ctypes.c_int
ddla_c_grid_dims.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

ddla_c_grid_coords = lib.ddla_c_grid_coords
ddla_c_grid_coords.restype = ctypes.c_int
ddla_c_grid_coords.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

ddla_c_rank = lib.ddla_c_rank
ddla_c_rank.restype = ctypes.c_int
ddla_c_rank.argtypes = [ctypes.c_void_p]

ddla_c_size = lib.ddla_c_size
ddla_c_size.restype = ctypes.c_int
ddla_c_size.argtypes = [ctypes.c_void_p]

ddla_c_descinit = lib.ddla_c_descinit
ddla_c_descinit.restype = ctypes.c_int
ddla_c_descinit.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

ddla_c_numloc = lib.ddla_c_numloc
ddla_c_numloc.restype = ctypes.c_int
ddla_c_numloc.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

ddla_c_malloc = lib.ddla_c_malloc
ddla_c_malloc.restype = ctypes.c_int
ddla_c_malloc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]

ddla_c_free = lib.ddla_c_free
ddla_c_free.restype = ctypes.c_int
ddla_c_free.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

ddla_c_memcpy = lib.ddla_c_memcpy
ddla_c_memcpy.restype = ctypes.c_int
ddla_c_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]

ddla_c_memcpy2d = lib.ddla_c_memcpy2d
ddla_c_memcpy2d.restype = ctypes.c_int
ddla_c_memcpy2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]

ddla_c_memset = lib.ddla_c_memset
ddla_c_memset.restype = ctypes.c_int
ddla_c_memset.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int64]

ddla_c_synchronize = lib.ddla_c_synchronize
ddla_c_synchronize.restype = ctypes.c_int
ddla_c_synchronize.argtypes = [ctypes.c_void_p]

ddla_c_finalize = getattr(lib, 'ddla_c_finalize', None)
if ddla_c_finalize is not None:
    ddla_c_finalize.restype = None
    ddla_c_finalize.argtypes = []

ddla_c_default_comm = lib.ddla_c_default_comm
ddla_c_default_comm.restype = ctypes.c_int
ddla_c_default_comm.argtypes = []

ddla_c_comm_rank = lib.ddla_c_comm_rank
ddla_c_comm_rank.restype = ctypes.c_int
ddla_c_comm_rank.argtypes = [ctypes.c_int]

ddla_c_comm_size = lib.ddla_c_comm_size
ddla_c_comm_size.restype = ctypes.c_int
ddla_c_comm_size.argtypes = [ctypes.c_int]

ddla_c_bcast_dims = lib.ddla_c_bcast_dims
ddla_c_bcast_dims.restype = ctypes.c_int
ddla_c_bcast_dims.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]

_TYPED_ROUTINES = {}
_ROUTINE_ARGS = {}
""")
    for name, tspec, args, opts in FUNCS:
        types = TYPES if tspec == "all4" else TYPES[:2]
        for t in types:
            sfx = t[0]
            cname = "ddla_%s_%s" % (sfx, name)
            sig_args = args
            if "drop_args" in opts:
                sig_args = [a for a in args if a[1] not in opts["drop_args"]]

            arg_types = []
            for k, n in sig_args:
                if k == "c":
                    arg_types.append("ctypes.c_char")
                elif k in ("i", "b"):
                    arg_types.append("ctypes.c_int")
                elif k == "i64":
                    arg_types.append("ctypes.c_int64")
                else:
                    arg_types.append("ctypes.c_void_p")

            arg_list_str = "[" + ", ".join(arg_types) + "]"
            arg_names_str = repr([(k, n) for k, n in sig_args])
            out.append("""
try:
    _fn = getattr(lib, "%s")
    _fn.restype = ctypes.c_int
    _fn.argtypes = %s
    _TYPED_ROUTINES[("%s", "%s")] = _fn
    _ROUTINE_ARGS[("%s", "%s")] = %s
except AttributeError:
    pass
""" % (cname, arg_list_str, name, sfx, name, sfx, arg_names_str))

    out.append("""
def get_typed_func(name: str, sfx: str):
    key = (name, sfx)
    if key not in _TYPED_ROUTINES:
        raise AttributeError(f"Routine {name} with type prefix {sfx} not found in LibDDLA C library.")
    return _TYPED_ROUTINES[key]

def get_routine_args(name: str, sfx: str) -> List[Tuple[str, str]]:
    key = (name, sfx)
    return _ROUTINE_ARGS.get(key, [])
""")
    return "\n".join(out)


def main():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "ddla_c_api.cpp"), "w") as f:
        f.write(gen_cpp())
    with open(os.path.join(here, "ddla.f90"), "w") as f:
        f.write(gen_f90())
    py_dir = os.path.join(os.path.dirname(here), "python", "ddla")
    os.makedirs(py_dir, exist_ok=True)
    with open(os.path.join(py_dir, "_c_api.py"), "w") as f:
        f.write(gen_py())
    n_c = sum(len(TYPES if t == "all4" else TYPES[:2]) for _, t, _, _ in FUNCS)
    print("generated %d typed compute bindings + 20 type-less C functions" % n_c)


if __name__ == "__main__":
    main()
