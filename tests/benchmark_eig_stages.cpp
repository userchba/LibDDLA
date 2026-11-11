#include "Eigensolver/pstedc_impl.h"
// ---------------------------------------------------------------------------
// Benchmark: LibDDLA two-stage eigensolver -- per-stage timings.
//
// This is the LibDDLA side of the ELPA2 comparison.  All five stages of the
// two-stage pipeline are implemented and timed:
//
//   stage 1  phe2hb        ~ ELPA bandred               (full -> band)
//   stages 2-4             ~ ELPA tridiag + trans_ev_tridiag
//            phb2st (band -> tridiagonal, recording rotations),
//            pstedc (distributed divide-and-conquer), and
//            punmtr_hb2st (tridiagonal -> band replay) are timed as one
//            group because that is how the driver consumes them.
//   stage 5  punmtr_he2hb  ~ ELPA trans_ev_band_to_full (band -> full)
//
// All are measured with all eigenvalues requested (the back-transform runs
// on n columns), matching the ELPA benchmark invocation
// `validate_{real,complex}_double_eigenvectors_2stage_default_kernel_gpu_random`
// with na = nev = n.
//
// This private row-baseline tridiagonal group uses distributed eigenvectors and replay at every
// size. The small correctness gate exchanges one column at a time.
//
// Heavy: registered with add_bench_executable, so ctest never runs it.
//
// Usage:
//   mpirun -n 4 ./benchmark_eig_stages --grid 2x2 500 5000 10000 15000 20000
//
// Environment overrides:
//   DDLA_BENCH_NB      block size nb (default 32, matching the ELPA nblk)
//   DDLA_BENCH_ITERS   timed iterations per size (default 3 for n <= 1000, else 1)
//   DDLA_BENCH_SIZES   comma-separated sizes, used when no sizes are on argv
// ---------------------------------------------------------------------------

#include "row_share.h"
#include "mpi_datatype.h"
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "api_grid_test_common.h"
#include "Eigensolver/eig_params.h"
#include "Eigensolver/two_stage/phb2st_impl.h"
#include "Eigensolver/two_stage/hb2st_cpu_replay_workspace.h"
#if DDLA_HAS_GPU
#include "Eigensolver/two_stage/hb2st_replay_gpu.h"
#endif
#include "eig_test_common.h"

using namespace api_grid_test;

namespace {

// Sizes at or below this get the host-side correctness gate (O(n^3) dense
// comparison, so it is only affordable at the smallest benchmarked size).
constexpr int kMaxCheckSize = 1000;

template <typename U> U bench_conj(U v) { return v; }
template <typename U> std::complex<U> bench_conj(std::complex<U> v) { return std::conj(v); }

template <typename T>
T herm_input(int i, int j)
{
    using rt = typename ddla::real_type<T>::type;
    const double re = 0.3 * ((i * 7 + j * 13) % 11) - 1.4 + 0.05 * ((i + j) % 3);
    const double im = 0.15 * ((i * 5 + j * 3) % 7) - 0.4;
    if (i == j) return T(static_cast<rt>(3.0 + 0.1 * i));   // real, positive-ish diagonal
    if constexpr (std::is_same<T, rt>::value) {
        (void)im;
        if (i > j) return T(static_cast<rt>(re));
        return herm_input<T>(j, i);   // mirror: a real input must be symmetric
    } else if (i > j) {
        return T(static_cast<rt>(re), static_cast<rt>(im));
    } else {
        return bench_conj(herm_input<T>(j, i));   // mirror: A(i,j) = conj(A(j,i))
    }
}

struct BenchRow {
    int n;
    std::string type;
    int nbw;
    double bandred;
    double tridiag;
    double b2f;
    int iters;
};

template <typename T>
void bench_type(const ddla::DdlaHandle_t& handle, int n, int nb, int niter,
                std::vector<BenchRow>& rows, bool show_progress)
{
    using rt = typename ddla::real_type<T>::type;
    const bool is_complex = !std::is_same<T, rt>::value;
    MPI_Comm comm = ddlaGetCommunicator(handle);

    int desc[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(desc, handle, n, n, nb, nb, 0, 0));
    int descZ[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descZ, handle, n, n, nb, nb, 0, 0));

    const ddla::EigParams params = ddla::eig_select_params(is_complex, nb, true);
    const int nbw = ddla::eig_distributed_bandwidth(n,nb,params.nbw,ddlaGetSize(handle));
    const int nsteps = std::max(1, (n - 1) / nbw);

    // Host sources.  phe2hb destroys its input, so A is restored from the host
    // copy between timed iterations (outside the timed region).
    auto h_A = make_local<T>(handle, desc, [](int i, int j) { return herm_input<T>(i, j); });
    auto h_I = make_local<T>(handle, descZ, [](int i, int j) { return T(i == j ? 1 : 0); });

    DeviceBuffer<T> d_A(handle, local_size(handle, desc));
    DeviceBuffer<T> d_Z(handle, local_size(handle, descZ));
    DeviceBuffer<T> d_tmat(handle, static_cast<size_t>(nbw) * nbw * nsteps);

    int info = 0;

    // ---- stage 1: full -> band -------------------------------------------
    upload(handle, d_A.ptr, h_A);
    check_ddla_sync(handle);
    ddla::phe2hb(handle, n, d_A.ptr, desc, nbw, d_tmat.ptr, info);   // warm-up
    check_ddla_sync(handle);
    if (info != 0) {
        if (ddlaGetRank(handle) == 0) std::cerr << "phe2hb info=" << info << std::endl;
        MPI_Abort(comm, 1);
    }

    double bandred = 0.0;
    for (int iter = 0; iter < niter; ++iter) {
        upload(handle, d_A.ptr, h_A);          // restore input, not timed
        check_ddla_sync(handle);
        MPI_Barrier(comm);
        const double t0 = MPI_Wtime();
        ddla::phe2hb(handle, n, d_A.ptr, desc, nbw, d_tmat.ptr, info);
        check_ddla_sync(handle);
        MPI_Barrier(comm);
        bandred += MPI_Wtime() - t0;
    }
    bandred /= niter;
    // d_A now holds the stage-1 output (band + reflectors).

    // ---- stages 2-4: the tridiagonal round trip --------------------------
    // phb2st (band -> tridiagonal with recorded rotations), pstedc (the
    // distributed divide-and-conquer solve) and punmtr_hb2st (the rotation
    // replay back to band form).  Timed as one group because that is how the
    // driver uses them: these are ELPA's "tridiag" plus "trans_ev_tridiag"
    // equivalents, and separating them would report the same host work twice.
    std::vector<rt> d_eigs(n), e_off(n), teigs(n);
    ddla::detail::DistributedHistory<T> history;
    std::vector<T> phases;
    int ranks=1,rank=0;MPI_Comm_size(comm,&ranks);MPI_Comm_rank(comm,&rank);
    const ddla::detail::RowShare row_share(n,ranks);
    const int nr=row_share.count(rank),ldy=std::max(1,nr);

    double tridiag = 0.0;
    for (int iter = 0; iter < niter; ++iter) {
        DeviceBuffer<rt> d_Y(handle,static_cast<std::size_t>(ldy)*n);
        upload(handle, d_A.ptr, h_A);
        check_ddla_sync(handle);
        ddla::phe2hb(handle, n, d_A.ptr, desc, nbw, d_tmat.ptr, info);
        check_ddla_sync(handle);
        MPI_Barrier(comm);
        const double t0 = MPI_Wtime();
        ddla::phb2st_node<ddla::default_backend_v,T>(handle, n, nbw, d_A.ptr, desc, d_eigs.data(), e_off.data(),
                     history, phases, true, info);
        if (info != 0) { MPI_Abort(comm, 1); }
        ddla::pstedc<rt>(handle,'V',n,d_eigs.data(),e_off.data(),teigs.data(),d_Y.ptr,ldy,info);
        if (info != 0) MPI_Abort(comm,1);
        if(ddlaGetBackend(handle)==ddla::DdlaBackend::CPU) {
            ddla::detail::hb2st_replay_rows_cpu<T>(handle,n,d_Y.ptr,ldy,history,phases,d_Z.ptr,descZ,info);
        } else
        {
#if DDLA_HAS_GPU
            ddla::detail::hb2st_replay_rows_gpu<T>(handle,n,d_Y.ptr,ldy,history,phases,d_Z.ptr,descZ,info);
#else
            throw std::runtime_error("benchmark: unsupported backend");
#endif
        }
        if(info!=0)MPI_Abort(comm,1);
        check_ddla_sync(handle);
        MPI_Barrier(comm);
        tridiag += MPI_Wtime() - t0;
    }
    tridiag /= niter;

    // ---- stage 5: band -> full -------------------------------------------
    upload(handle, d_Z.ptr, h_I);
    check_ddla_sync(handle);
    ddla::punmtr_he2hb(handle, 'N', n, nbw, d_A.ptr, desc, d_tmat.ptr, d_Z.ptr, descZ, info);
    check_ddla_sync(handle);
    if (info != 0) {
        if (ddlaGetRank(handle) == 0) std::cerr << "punmtr_he2hb info=" << info << std::endl;
        MPI_Abort(comm, 1);
    }

    double b2f = 0.0;
    for (int iter = 0; iter < niter; ++iter) {
        upload(handle, d_Z.ptr, h_I);          // restore identity, not timed
        check_ddla_sync(handle);
        MPI_Barrier(comm);
        const double t0 = MPI_Wtime();
        ddla::punmtr_he2hb(handle, 'N', n, nbw, d_A.ptr, desc, d_tmat.ptr, d_Z.ptr, descZ, info);
        check_ddla_sync(handle);
        MPI_Barrier(comm);
        b2f += MPI_Wtime() - t0;
    }
    b2f /= niter;

    // ---- cheap correctness gate ------------------------------------------
    // The benchmarked path should not be entirely unverified.  At the smallest
    // size only (a dense O(n^3) host comparison is affordable there), check
    // that the driver's back-transform direction gives Q^H.  d_Z currently
    // holds Q from the trans='N' iterations above; re-running with 'C' on the
    // identity must produce its adjoint.
    if (n <= kMaxCheckSize) {
        // Save only the local block-cyclic part of Q. Compare one global
        // column at a time against the corresponding row of Q^H.
        const auto Qlocal = download<T>(handle,d_Z.ptr,local_size(handle,descZ));
        upload(handle, d_Z.ptr, h_I);
        check_ddla_sync(handle);
        ddla::punmtr_he2hb(handle,'C',n,nbw,d_A.ptr,desc,d_tmat.ptr,d_Z.ptr,descZ,info);
        check_ddla_sync(handle);
        const auto Zlocal = download<T>(handle,d_Z.ptr,local_size(handle,descZ));
        const int nloc=ddla_test::n_loc(handle,descZ),mloc=ddla_test::m_loc(handle,descZ);
        const int ld=descZ[ddla::DDLA_LLD_];
        std::vector<T> column(n);
        double cerr=0.0;
        for(int j=0;j<n;++j) {
            std::fill(column.begin(),column.end(),T(0));
            for(int c=0;c<nloc;++c)if(ddla::indx_l2g_c(descZ,handle,c)==j)
                for(int r=0;r<mloc;++r) {
                    const int global=ddla::indx_l2g_r(descZ,handle,r);
                    if(global<n)column[global]=Qlocal[r+std::size_t(c)*ld];
                }
            MPI_Allreduce(MPI_IN_PLACE,column.data(),n,ddla::detail::mpi_datatype<T>(),MPI_SUM,comm);
            for(int r=0;r<mloc;++r)if(ddla::indx_l2g_r(descZ,handle,r)==j)
                for(int c=0;c<nloc;++c) {
                    const int global=ddla::indx_l2g_c(descZ,handle,c);
                    if(global<n)cerr=std::max(cerr,static_cast<double>(
                        std::abs(Zlocal[r+std::size_t(c)*ld]-bench_conj(column[global]))));
                }
        }
        MPI_Allreduce(MPI_IN_PLACE,&cerr,1,MPI_DOUBLE,MPI_MAX,comm);
        const double metric = cerr / (static_cast<double>(n) * eig_test::eps_trait<rt>::value);
        if (metric > 50.0) {
            if (ddlaGetRank(handle) == 0)
                std::cerr << "benchmark correctness gate FAILED at n=" << n
                          << ": |punmtr_he2hb('C', I) - Q^H|_max / (n eps) = " << metric << std::endl;
            MPI_Abort(comm, 1);
        }
        if (ddlaGetRank(handle) == 0 && show_progress)
            std::cout << " (correctness gate n=" << n << " ok, metric=" << metric << ")" << std::flush;
    }

    double rmax[3] = {bandred, tridiag, b2f};
    double gmax[3] = {0.0, 0.0, 0.0};
    MPI_Reduce(rmax, gmax, 3, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (ddlaGetRank(handle) == 0) {
        rows.push_back({n, is_complex ? "complex<double>" : "double", nbw,
                        gmax[0], gmax[1], gmax[2], niter});
    }
}

int pick_block_size(int n, int nb)
{
    const char* e = std::getenv("DDLA_BENCH_NB");
    if (e != nullptr && *e != '\0') {
        const int v = std::atoi(e);
        if (v > 0 && v <= n) return v;
    }
    return std::min(nb, n);
}

int pick_iters(int n)
{
    const char* e = std::getenv("DDLA_BENCH_ITERS");
    if (e != nullptr && *e != '\0') {
        const int v = std::atoi(e);
        if (v > 0) return v;
    }
    return (n <= 1000) ? 3 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    ddla_test_mpi_init(&argc, &argv);

    int nprocs = 0;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // One grid only (unlike the api_grid harness, which sweeps every divisor).
    int nprows = 0, npcols = 0;
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--grid" && i + 1 < argc) {
            const std::string spec(argv[++i]);
            if (!parse_grid_spec(spec, nprows, npcols)) {
                if (rank == 0) std::cerr << "invalid --grid value: " << spec << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        } else if (arg.rfind("--grid=", 0) == 0) {
            const std::string spec = arg.substr(7);
            if (!parse_grid_spec(spec, nprows, npcols)) {
                if (rank == 0) std::cerr << "invalid --grid value: " << spec << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        } else {
            sizes.push_back(std::atoi(arg.c_str()));
        }
    }

    if (sizes.empty()) {
        const char* e = std::getenv("DDLA_BENCH_SIZES");
        if (e != nullptr && *e != '\0') {
            std::istringstream iss(e);
            std::string tok;
            while (std::getline(iss, tok, ',')) {
                const int v = std::atoi(tok.c_str());
                if (v > 0) sizes.push_back(v);
            }
        }
    }
    if (sizes.empty()) sizes = {500, 5000, 10000, 15000, 20000};

    if (nprows == 0) {   // most square divisor pair of nprocs
        nprows = 1;
        npcols = nprocs;
        for (int r = 1; r <= nprocs; ++r) {
            if (nprocs % r == 0 && r * r <= nprocs) {
                nprows = r;
                npcols = nprocs / r;
            }
        }
    }
    if (nprows * npcols != nprocs) {
        if (rank == 0) {
            std::cerr << "--grid " << nprows << "x" << npcols << " needs "
                      << nprows * npcols << " ranks, this run has " << nprocs << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int nb = 32;
    const int nbo = pick_block_size(sizes.front(), nb);

    if (rank == 0) {
        std::cout << "=== LibDDLA two-stage eigensolver stage benchmark ===" << std::endl;
        std::cout << "ranks=" << nprocs << " grid=" << nprows << "x" << npcols
                  << " nb=" << nbo << " (all eigenpairs)" << std::endl;
        std::cout << "sizes:";
        for (int n : sizes) std::cout << " " << n;
        std::cout << std::endl << std::flush;
    }

    ddla::DdlaHandle_t handle = nullptr;
    ddla::ddlaInit(handle);
    ddla::ddlaSet(handle, MPI_COMM_WORLD, nprows, npcols);

    std::vector<BenchRow> rows;
    for (int n : sizes) {
        const int iters = pick_iters(n);
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) std::cout << "  running n=" << n << " ..." << std::flush;
        bench_type<double>(handle, n, nbo, iters, rows, rank == 0);
        bench_type<Complex>(handle, n, nbo, iters, rows, rank == 0);
        if (rank == 0) std::cout << " done" << std::endl;
    }

    if (rank == 0) {
        std::cout << std::endl << "=== stage timings (seconds, max over ranks) ===" << std::endl;
        std::cout << "  tridiag includes distributed phb2st, pstedc and replay at every size." << std::endl;
        std::cout << std::setw(8) << "n"
                  << std::setw(18) << "type"
                  << std::setw(6) << "nbw"
                  << std::setw(14) << "bandred"
                  << std::setw(14) << "tridiag"
                  << std::setw(14) << "b2f"
                  << std::setw(14) << "total"
                  << std::setw(8) << "iters" << std::endl;
        std::cout << std::string(96, '-') << std::endl;
        for (const auto& r : rows) {
            const char* mark = "";
            std::cout << std::setw(8) << r.n
                      << std::setw(18) << r.type
                      << std::setw(6) << r.nbw
                      << std::setw(14) << std::fixed << std::setprecision(4) << r.bandred
                      << std::setw(13) << r.tridiag << mark
                      << std::setw(14) << r.b2f
                      << std::setw(14) << (r.bandred + r.tridiag + r.b2f)
                      << std::setw(8) << r.iters << std::endl;
        }
    }

    check_ddla_sync(handle);
    ddla::ddlaDestroy(handle);
    MPI_Finalize();
    return 0;
}
