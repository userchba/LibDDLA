#ifndef API_GRID_TEST_COMMON_H
#define API_GRID_TEST_COMMON_H

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

#include <ddla/ddla.h>
#include <ddla/ddla_connector.h>
#include <ddla/ddla_stream.h>
#include <ddla/ptran.h>
#include <ddla/transport_block.h>

namespace api_grid_test {

using Complex = std::complex<double>;
using namespace ddla;

struct Shape {
    int m;
    int n;
    int k;
    int nb;
};

struct TestOptions {
    std::vector<std::pair<int, int>> grids;
    std::vector<Shape> shapes;
    int irsrc = 0;
    int icsrc = 0;
};

// Populated by run_grid_test from --src IR IC (default 0,0). Only the
// pgemm nonzero-irsrc/icsrc probe currently reads these; every other
// api_grid test keeps using descriptor irsrc=icsrc=0 as before.
inline int g_test_irsrc = 0;
inline int g_test_icsrc = 0;

inline std::string grid_name(const ddla::DdlaHandle_t& handle)
{
    int nprows = 0, npcols = 0;
    ddla_get_grid_dims(handle, nprows, npcols);
    std::ostringstream os;
    os << nprows << "x" << npcols;
    return os.str();
}

inline size_t local_size(const ddla::DdlaDesc& desc)
{
    return static_cast<size_t>(desc.lld()) * desc.n_loc();
}

inline int round_up_for_grid(int value, int block, int procs)
{
    return std::max(value, block * procs + 1);
}

inline int square_size(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    int nprows = 0, npcols = 0;
    ddla_get_grid_dims(handle, nprows, npcols);
    return round_up_for_grid(base.m, base.nb, std::max(nprows, npcols));
}

inline int nrhs_size(const Shape& base, int upper = 4)
{
    return std::max(2, std::min(upper, base.n));
}

inline bool parse_positive_int(const std::string& text, int& value)
{
    if(text.empty()) return false;
    long long parsed = 0;
    for(char ch : text){
        if(!std::isdigit(static_cast<unsigned char>(ch))) return false;
        parsed = parsed * 10 + (ch - '0');
        if(parsed > std::numeric_limits<int>::max()) return false;
    }
    if(parsed <= 0) return false;
    value = static_cast<int>(parsed);
    return true;
}

inline bool parse_nonneg_int(const std::string& text, int& value)
{
    if(text.empty()) return false;
    long long parsed = 0;
    for(char ch : text){
        if(!std::isdigit(static_cast<unsigned char>(ch))) return false;
        parsed = parsed * 10 + (ch - '0');
        if(parsed > std::numeric_limits<int>::max()) return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

inline bool parse_grid_spec(const std::string& spec, int& nprows, int& npcols)
{
    const size_t xpos = spec.find_first_of("xX");
    if(xpos == std::string::npos || xpos == 0 || xpos + 1 >= spec.size()){
        return false;
    }
    if(spec.find_first_of("xX", xpos + 1) != std::string::npos){
        return false;
    }
    return parse_positive_int(spec.substr(0, xpos), nprows)
        && parse_positive_int(spec.substr(xpos + 1), npcols);
}

inline std::string usage(const char* program)
{
    std::ostringstream os;
    os << "Usage: " << program << " [--grid RxC] [--size N]\n"
       << "       " << program << " [N]";
    return os.str();
}

inline bool parse_options(int argc, char** argv, int nprocs, TestOptions& options, std::string& error)
{
    bool grid_set = false;
    bool size_set = false;
    bool src_set = false;
    int requested_grid_rows = 0;
    int requested_grid_cols = 0;
    int requested_size = 0;
    int requested_irsrc = 0;
    int requested_icsrc = 0;

    for(int i = 1; i < argc; ++i){
        std::string arg(argv[i]);
        if(arg == "--grid"){
            if(i + 1 >= argc){
                error = "--grid requires a value like 2x3";
                return false;
            }
            if(grid_set){
                error = "--grid was provided more than once";
                return false;
            }
            grid_set = true;
            if(!parse_grid_spec(argv[++i], requested_grid_rows, requested_grid_cols)){
                error = "invalid --grid value: " + std::string(argv[i]);
                return false;
            }
        }else if(arg.rfind("--grid=", 0) == 0){
            if(grid_set){
                error = "--grid was provided more than once";
                return false;
            }
            grid_set = true;
            const std::string spec = arg.substr(7);
            if(!parse_grid_spec(spec, requested_grid_rows, requested_grid_cols)){
                error = "invalid --grid value: " + spec;
                return false;
            }
        }else if(arg == "--size"){
            if(i + 1 >= argc){
                error = "--size requires a positive integer";
                return false;
            }
            if(size_set){
                error = "matrix size was provided more than once";
                return false;
            }
            size_set = true;
            if(!parse_positive_int(argv[++i], requested_size)){
                error = "invalid --size value: " + std::string(argv[i]);
                return false;
            }
        }else if(arg.rfind("--size=", 0) == 0){
            if(size_set){
                error = "matrix size was provided more than once";
                return false;
            }
            size_set = true;
            const std::string size_text = arg.substr(7);
            if(!parse_positive_int(size_text, requested_size)){
                error = "invalid --size value: " + size_text;
                return false;
            }
        }else if(arg == "--src"){
            if(i + 2 >= argc){
                error = "--src requires two non-negative integers: IR IC";
                return false;
            }
            if(src_set){
                error = "--src was provided more than once";
                return false;
            }
            src_set = true;
            const std::string ir_text(argv[++i]);
            const std::string ic_text(argv[++i]);
            if(!parse_nonneg_int(ir_text, requested_irsrc) ||
               !parse_nonneg_int(ic_text, requested_icsrc)){
                error = "invalid --src values: " + ir_text + " " + ic_text;
                return false;
            }
        }else if(!arg.empty() && arg[0] == '-'){
            error = "unknown option: " + arg;
            return false;
        }else{
            if(size_set){
                error = "matrix size was provided more than once";
                return false;
            }
            size_set = true;
            if(!parse_positive_int(arg, requested_size)){
                error = "invalid matrix size: " + arg;
                return false;
            }
        }
    }

    if(grid_set){
        const long long requested_ranks =
            static_cast<long long>(requested_grid_rows) * requested_grid_cols;
        if(requested_ranks != nprocs){
            std::ostringstream os;
            os << "--grid " << requested_grid_rows << "x" << requested_grid_cols
               << " requires " << requested_ranks
               << " MPI ranks, but this run has " << nprocs;
            error = os.str();
            return false;
        }
        options.grids = {{requested_grid_rows, requested_grid_cols}};
    }else{
        for(int r = 1; r <= nprocs; ++r){
            if(nprocs % r == 0){
                options.grids.emplace_back(r, nprocs / r);
            }
        }
    }

    if(size_set){
        const int n = std::max(6, requested_size);
        options.shapes = {{n, std::max(5, n - 2), std::max(5, n - 1), 3}};
    }else{
        options.shapes = {
            {13, 11, 9, 3},
            {17, 10, 14, 4},
        };
    }

    options.irsrc = requested_irsrc;
    options.icsrc = requested_icsrc;
    return true;
}

// ---------------------------------------------------------------------------
// Fail-fast wrappers for ddla_* memory/synchronization calls
// ---------------------------------------------------------------------------
inline void check_ddla_malloc(void** ptr, size_t bytes,
                              const ddla::DdlaHandle_t& handle)
{
    int rc = ddla_malloc(ptr, bytes, handle);
    if (rc != 0) {
        std::cerr << "ddla_malloc failed (rc=" << rc << ")" << std::endl;
        MPI_Abort(ddla_get_communicator(handle), 1);
    }
    if (bytes > 0 && *ptr == nullptr) {
        std::cerr << "ddla_malloc returned null for nonzero size" << std::endl;
        MPI_Abort(ddla_get_communicator(handle), 1);
    }
}

inline void check_ddla_free(void* ptr, const ddla::DdlaHandle_t& handle)
{
    if (ptr == nullptr) return;
    int rc = ddla_free(ptr, handle);
    if (rc != 0) {
        std::cerr << "ddla_free failed (rc=" << rc << ")" << std::endl;
        MPI_Abort(ddla_get_communicator(handle), 1);
    }
}

inline void check_ddla_memcpy(void* dst, const void* src, size_t bytes,
                              ddla::DdlaMemoryCopyKind kind,
                              const ddla::DdlaHandle_t& handle)
{
    int rc = ddla_memcpy(dst, src, bytes, kind, handle);
    if (rc != 0) {
        std::cerr << "ddla_memcpy failed (rc=" << rc << ")" << std::endl;
        MPI_Abort(ddla_get_communicator(handle), 1);
    }
}

inline void check_ddla_sync(const ddla::DdlaHandle_t& handle)
{
    int rc = ddla_synchronize(handle);
    if (rc != 0) {
        std::cerr << "ddla_synchronize failed (rc=" << rc << ")" << std::endl;
        MPI_Abort(ddla_get_communicator(handle), 1);
    }
}

template <typename T>
struct DeviceBuffer {
    ddla::DdlaHandle_t handle;
    T* ptr = nullptr;
    size_t count = 0;

    DeviceBuffer(const ddla::DdlaHandle_t& h, size_t n) : handle(h), count(std::max<size_t>(1, n))
    {
        check_ddla_malloc(reinterpret_cast<void**>(&ptr), count * sizeof(T), handle);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer()
    {
        if(ptr != nullptr){
            check_ddla_free(ptr, handle);
        }
    }
};

template <typename T>
inline void upload(const ddla::DdlaHandle_t& handle, T* dst, const std::vector<T>& src)
{
    if(!src.empty()){
        check_ddla_memcpy(dst, src.data(), src.size() * sizeof(T),
                          DdlaMemoryCopyKind::HostToDevice, handle);
    }
}

template <typename T>
inline std::vector<T> download(const ddla::DdlaHandle_t& handle, const T* src, size_t count)
{
    std::vector<T> host(count);
    if(count > 0){
        check_ddla_memcpy(host.data(), src, count * sizeof(T),
                          DdlaMemoryCopyKind::DeviceToHost, handle);
    }
    check_ddla_sync(handle);
    return host;
}

template <typename T, typename Fn>
inline std::vector<T> make_local(const ddla::DdlaDesc& desc, Fn value)
{
    std::vector<T> local(local_size(desc), T{});
    for(int jloc = 0; jloc < desc.n_loc(); ++jloc){
        const int j = desc.indx_l2g_c(jloc);
        if(j >= desc.n()) continue;
        for(int iloc = 0; iloc < desc.m_loc(); ++iloc){
            const int i = desc.indx_l2g_r(iloc);
            if(i >= desc.m()) continue;
            local[iloc + jloc * desc.lld()] = value(i, j);
        }
    }
    return local;
}

template <typename T, typename Fn>
inline double local_max_error(const ddla::DdlaDesc& desc, const std::vector<T>& local, Fn expected)
{
    double err = 0.0;
    for(int jloc = 0; jloc < desc.n_loc(); ++jloc){
        const int j = desc.indx_l2g_c(jloc);
        if(j >= desc.n()) continue;
        for(int iloc = 0; iloc < desc.m_loc(); ++iloc){
            const int i = desc.indx_l2g_r(iloc);
            if(i >= desc.m()) continue;
            err = std::max(err, static_cast<double>(std::abs(local[iloc + jloc * desc.lld()] - expected(i, j))));
        }
    }
    return err;
}

inline void require_close(const ddla::DdlaHandle_t& handle, const std::string& name,
                          double local_err, double tol)
{
    double global_err = 0.0;
    MPI_Comm comm = ddla_get_communicator(handle);
    MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, comm);
    int rank = ddla_get_rank(handle);
    if(rank == 0){
        std::cout << "[grid " << grid_name(handle) << "] " << name
                  << " max_err=" << global_err << std::endl;
    }
    if(global_err > tol){
        if(rank == 0){
            std::cerr << "FAIL: " << name << " exceeded tolerance " << tol << std::endl;
        }
        MPI_Abort(comm, 1);
    }
}

inline Complex general_value(int i, int j, int tag)
{
    const double seed = static_cast<double>((i + 1) * (tag + 3) - (j + 2) * (tag + 5));
    const double re = 0.03 * std::sin(seed) + 0.002 * (i - j);
    const double im = 0.02 * std::cos(seed * 0.7) + 0.001 * (i + j + tag);
    return Complex(re, im);
}

inline Complex dominant_value(int i, int j, int n)
{
    if(i == j){
        return Complex(4.0 + 0.1 * i, 0.0);
    }
    return Complex(0.015 * ((i + 2 * j) % 5 - 2), 0.01 * ((2 * i + j) % 7 - 3));
}

// Row-cycled variant of dominant_value: global row i carries the strong
// diagonal element of row (i+1)%n, so diagonal blocks factored with partial
// pivoting must actually swap rows (non-identity ipiv).  Exercises the pivot
// path that a strictly diagonally dominant matrix never triggers.
inline Complex cycled_value(int i, int j, int n)
{
    return dominant_value((i + 1) % n, j, n);
}

inline Complex hpd_value(int i, int j, int n)
{
    if(i == j){
        return Complex(5.0 + 0.2 * n + 0.05 * i, 0.0);
    }
    const int lo = std::min(i, j);
    const int hi = std::max(i, j);
    const Complex val(0.01 * ((lo + 2 * hi) % 5 - 2),
                      0.006 * ((3 * lo + hi) % 7 - 3));
    return i < j ? val : std::conj(val);
}

inline Complex triangular_l_value(int i, int j)
{
    if(i < j) return Complex(0.0, 0.0);
    if(i == j) return Complex(2.0 + 0.05 * i, 0.0);
    return Complex(0.04 * ((i + j) % 4 + 1), 0.01 * ((i - j) % 3));
}

inline Complex x_value(int i, int j)
{
    return Complex(0.2 + 0.03 * i - 0.02 * j, 0.04 * (i + 1) + 0.01 * j);
}

inline Complex op_value(char trans, int rows, int cols, int i, int j,
                        Complex (*value)(int, int, int), int tag)
{
    (void)rows;
    (void)cols;
    if(trans == 'N') return value(i, j, tag);
    const Complex raw = value(j, i, tag);
    return trans == 'T' ? raw : std::conj(raw);
}

inline std::vector<Complex> build_rhs(const ddla::DdlaDesc& descB, int n,
                                      Complex (*matrix)(int, int, int),
                                      int tag)
{
    return make_local<Complex>(descB, [&](int i, int j) -> Complex {
        Complex sum(0.0, 0.0);
        for(int l = 0; l < n; ++l){
            sum += matrix(i, l, tag) * x_value(l, j);
        }
        return sum;
    });
}

// B := op(A)*X (side='L', B is n x nrhs) or B := X*op(A) (side='R',
// B is nrhs x n), with op(A) chosen by trans via op_value.
inline std::vector<Complex> build_rhs_side(const ddla::DdlaDesc& descB, int n,
                                           char side, char trans,
                                           Complex (*matrix)(int, int, int),
                                           int tag)
{
    return make_local<Complex>(descB, [&](int i, int j) -> Complex {
        Complex sum(0.0, 0.0);
        for(int l = 0; l < n; ++l){
            if(side == 'L')
                sum += op_value(trans, n, n, i, l, matrix, tag) * x_value(l, j);
            else
                sum += x_value(i, l) * op_value(trans, n, n, l, j, matrix, tag);
        }
        return sum;
    });
}

inline void check_solution(const ddla::DdlaHandle_t& handle, const ddla::DdlaDesc& descB,
                           const Complex* d_B, size_t count,
                           const std::string& name, double tol)
{
    auto out = download(handle, d_B, count);
    const double err = local_max_error<Complex>(descB, out, [](int i, int j){
        return x_value(i, j);
    });
    require_close(handle, name, err, tol);
}

inline bool skip_non_square_grid(const ddla::DdlaHandle_t& handle, const std::string& name)
{
    int nprows = 0, npcols = 0;
    ddla_get_grid_dims(handle, nprows, npcols);
    if(nprows == npcols) return false;
    int rank = ddla_get_rank(handle);
    if(rank == 0){
        std::cout << "[grid " << grid_name(handle) << "] skip " << name
                  << " on non-square process grid" << std::endl;
    }
    return true;
}

template <typename Body>
int run_grid_test(int argc, char** argv, const std::string& test_name, Body body)
{
    MPI_Init(&argc, &argv);

    int nprocs = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    TestOptions options;
    std::string error;
    if(!parse_options(argc, argv, nprocs, options, error)){
        if(rank == 0){
            std::cerr << "Error: " << error << std::endl;
            std::cerr << usage(argv[0]) << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    g_test_irsrc = options.irsrc;
    g_test_icsrc = options.icsrc;

    for (const auto& grid : options.grids){
        const int nprows = grid.first;
        const int npcols = grid.second;
        ddla::DdlaHandle_t handle = nullptr;
        ddla::ddla_init(handle);
        ddla::ddla_set(handle, MPI_COMM_WORLD, nprows, npcols);
        int myid = ddla_get_rank(handle);
        if(myid == 0){
            std::cout << "=== grid " << grid_name(handle) << " ===" << std::endl;
        }

        for(const auto& shape : options.shapes){
            body(handle, shape);
        }

        check_ddla_sync(handle);
        ddla::ddla_destroy(handle);
    }

    if(rank == 0){
        std::cout << test_name << " passed" << std::endl;
    }

    MPI_Finalize();
    return 0;
}

} // namespace api_grid_test

#endif
