#include "matrix_transfer.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "api_grid_test_common.h"
#include <ddla/ddla.h>
#include "row_share.h"

// ---------------------------------------------------------------------------
// Isolation test of the row-distributed bidiagonal divide and conquer
// pbdsdc (src/SVD/pbdsdc.cpp): B = U diag(s) V^T for a REPLICATED real
// bidiagonal (d, e), upper or lower, in float and double, with
// DDLA_SVD_DC_LEAF forced small so the tree is deep (a leaf of 1024 would
// never merge at these sizes).
//
// Cases: generic spectra, exact zeros in d and e (independent trees of tiny
// size), a clustered spectrum, a graded one, e = 0 at fixed positions (several
// LARGE independent trees), and a near-diagonal matrix with four distinct
// diagonal values (heavy deflation by tiny z and by equal d: the Givens path).
//
// Checks per case (all bounds in units of k*eps):
//   - bitwise identical singular values on every rank;
//   - gather the row slabs only in this small verification harness;
//   - s descending and >= 0; s against host ?BDSDC('N');
//   - U^T U = I, V^T V = I and B = U diag(s) V^T (a pure residual test cannot
//     see a sign error on a near-null row, the reconstruction does);
//   - one large k with sampled checks (B v_j = s_j u_j, orthogonality of
//     random column pairs) that would be too slow in O(k^3).
// ---------------------------------------------------------------------------

using namespace api_grid_test;

namespace {

extern "C" {
void dbdsdc_(const char*, const char*, const int*, double*, double*, double*, const int*,
             double*, const int*, double*, int*, double*, int*, int*);
void sbdsdc_(const char*, const char*, const int*, float*, float*, float*, const int*,
             float*, const int*, float*, int*, float*, int*, int*);
}

void bdsdc_n(char uplo, int n, double* d, double* e, std::vector<double>& work, std::vector<int>& iw,
             int& info)
{
    const char compq = 'N';
    dbdsdc_(&uplo, &compq, &n, d, e, nullptr, &n, nullptr, &n, nullptr, nullptr, work.data(),
            iw.data(), &info);
}
void bdsdc_n(char uplo, int n, float* d, float* e, std::vector<float>& work, std::vector<int>& iw,
             int& info)
{
    const char compq = 'N';
    sbdsdc_(&uplo, &compq, &n, d, e, nullptr, &n, nullptr, &n, nullptr, nullptr, work.data(),
            iw.data(), &info);
}

template <typename rt> MPI_Datatype mpi_rt();
template <> MPI_Datatype mpi_rt<float>() { return MPI_FLOAT; }
template <> MPI_Datatype mpi_rt<double>() { return MPI_DOUBLE; }
template <typename rt> const char* tag_rt() { return std::is_same<rt, float>::value ? "s" : "d"; }

// splitmix64 of (i, tag) -> uniform in [-0.5, 0.5).
inline double hash01(int i, int tag)
{
    std::uint64_t x = static_cast<std::uint64_t>(i + 1) * 0x9E3779B97F4A7C15ull
                    ^ static_cast<std::uint64_t>(tag) * 0xC2B2AE3D27D4EB4Full;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27; x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return static_cast<double>(x >> 11) / 9007199254740992.0 - 0.5;
}

enum Cls { C_RAND, C_RANKDEF, C_CLUSTER, C_GRADED, C_SPLIT, C_DEFL, C_DIAG, C_COUNT };
const char* cls_name(int c)
{
    static const char* names[] = {"rand", "rankdef", "cluster", "graded", "split", "defl", "diag"};
    return names[c];
}

template <typename rt>
void make_bidiag(int cls, int k, int tag, std::vector<rt>& d, std::vector<rt>& e)
{
    d.assign(static_cast<std::size_t>(k), rt(0));
    e.assign(static_cast<std::size_t>(std::max(1, k - 1)), rt(0));
    for (int i = 0; i < k; ++i) {
        double di = 0.0, ei = 0.0;
        switch (cls) {
        case C_RAND:    di = hash01(i, tag); ei = hash01(i, tag + 1); break;
        case C_RANKDEF: di = (i % 3 == 0) ? 0.0 : hash01(i, tag);
                        ei = (i % 3 == 0) ? 0.0 : hash01(i, tag + 1); break;
        case C_CLUSTER: di = 1.0 + 1e-6 * hash01(i, tag); ei = 0.5 * hash01(i, tag + 1); break;
        case C_GRADED:  di = std::pow(10.0, -8.0 * i / std::max(1, k)) * (1.0 + hash01(i, tag));
                        ei = std::pow(10.0, -8.0 * i / std::max(1, k)) * 0.5 * hash01(i, tag + 1); break;
        case C_SPLIT:   di = hash01(i, tag); ei = hash01(i, tag + 1);
                        if (k >= 30 && (i == k / 3 || i == (2 * k) / 3)) ei = 0.0; break;
        case C_DEFL:    di = 1.0 + (hash01(i, tag) > 0 ? 1.0 : 0.0) + 2.0 * (hash01(i, tag + 2) > 0.25 ? 1.0 : 0.0);
                        ei = 1e-9 * hash01(i, tag + 1); break;
        case C_DIAG:    di = hash01(i, tag); ei = 0.0; break;      // every e splits: ~k one-element leaf jobs
        }
        d[static_cast<std::size_t>(i)] = static_cast<rt>(di);
        if (i + 1 < k) e[static_cast<std::size_t>(i)] = static_cast<rt>(ei);
    }
}

// B(i, j) of the bidiagonal.
template <typename rt>
rt bval(char uplo, int i, int j, const std::vector<rt>& d, const std::vector<rt>& e)
{
    if (i == j) return d[static_cast<std::size_t>(i)];
    if (uplo == 'U' && j == i + 1) return e[static_cast<std::size_t>(i)];
    if (uplo == 'L' && i == j + 1) return e[static_cast<std::size_t>(j)];
    return rt(0);
}

// Max |x - y| over ranks of a replicated vector; 0 means bitwise equal.
template <typename rt>
double replication_diff(const ddla::DdlaHandle_t& handle, std::vector<rt> v)
{
    MPI_Comm comm = ddlaGetCommunicator(handle);
    std::vector<rt> r0 = v;
    MPI_Bcast(r0.data(), static_cast<int>(r0.size()), mpi_rt<rt>(), 0, comm);
    int bad = (std::memcmp(r0.data(), v.data(), v.size() * sizeof(rt)) != 0) ? 1 : 0;
    int all = 0;
    MPI_Allreduce(&bad, &all, 1, MPI_INT, MPI_MAX, comm);
    return static_cast<double>(all);
}

template <typename rt>
void run_case(const ddla::DdlaHandle_t& handle, int cls, int k, char uplo, int leaf, bool sampled)
{
    const double eps = static_cast<double>(std::numeric_limits<rt>::epsilon());
    const std::string tag = std::string(tag_rt<rt>()) + " " + cls_name(cls) + " k=" + std::to_string(k) +
                            " " + uplo + " leaf=" + std::to_string(leaf);
    const int tg = 100 + 7 * k + 13 * cls;
    std::vector<rt> d0, e0;
    make_bidiag<rt>(cls, k, tg, d0, e0);

    setenv("DDLA_SVD_DC_LEAF", std::to_string(leaf).c_str(), 1);
    std::vector<rt> d = d0, e = e0, s(static_cast<std::size_t>(k), rt(0));
    int ranks=1; MPI_Comm_size(ddlaGetCommunicator(handle), &ranks);
    const ddla::detail::RowShare share(k,ranks);
    const int rank=ddlaGetRank(handle), rows=share.count(rank), ld=std::max(1,rows);
    DeviceBuffer<rt> dU(handle, std::size_t(ld)*k), dV(handle, std::size_t(ld)*k);
    int info = -1;
#if DDLA_HAS_GPU
    ddla::pbdsdc<ddla::DdlaBackend::GPU,rt>(handle,uplo,k,d.data(),e.data(),s.data(),dU.ptr,dV.ptr,ld,info);
#else
    ddla::pbdsdc<ddla::DdlaBackend::CPU,rt>(handle,uplo,k,d.data(),e.data(),s.data(),dU.ptr,dV.ptr,ld,info);
#endif
    check_ddla_sync(handle);
    require_close(handle, tag + " info", static_cast<double>(info), 0.0);
    auto uloc=download(handle,dU.ptr,std::size_t(ld)*k);
    auto vloc=download(handle,dV.ptr,std::size_t(ld)*k);
    // Replication belongs only to this small verification harness, after solve.
    std::vector<rt> U(std::size_t(k)*k,rt(0)), V(U.size(),rt(0));
    for(int j=0;j<k;++j)for(int r=0;r<rows;++r) {
        U[share.lo(rank)+r+std::size_t(j)*k]=uloc[r+std::size_t(j)*ld];
        V[share.lo(rank)+r+std::size_t(j)*k]=vloc[r+std::size_t(j)*ld];
    }
    MPI_Allreduce(MPI_IN_PLACE,U.data(),int(U.size()),mpi_rt<rt>(),MPI_SUM,ddlaGetCommunicator(handle));
    MPI_Allreduce(MPI_IN_PLACE,V.data(),int(V.size()),mpi_rt<rt>(),MPI_SUM,ddlaGetCommunicator(handle));

    // ---- replicated output ---------------------------------------------------
    require_close(handle, tag + " s replication", replication_diff(handle, s), 0.0);
    require_close(handle, tag + " U replication", replication_diff(handle, U), 0.0);
    require_close(handle, tag + " V replication", replication_diff(handle, V), 0.0);

    // Dense verification against replicated data (U, V, s): root-only.  The
    // identical O(k^3) reconstruction on every rank dominated this test's
    // check phase (k up to 601); the replication checks above need all ranks
    // and stay collective.
    RootSection dense(handle);
    if (dense.root) {
    // ---- s: descending, >= 0, against host ?BDSDC('N') -------------------------
    double smax = 0.0, sorder = 0.0;
    for (int i = 0; i < k; ++i) {
        smax = std::max(smax, std::fabs(static_cast<double>(s[static_cast<std::size_t>(i)])));
        if (s[static_cast<std::size_t>(i)] < rt(0)) sorder = 1.0;
        if (i > 0 && s[static_cast<std::size_t>(i)] > s[static_cast<std::size_t>(i) - 1]) sorder = 1.0;
    }
    dense.check(tag + " s descending/nonnegative", sorder, 0.0);
    const double smax_g = std::max(smax, 1e-300);
    {
        std::vector<rt> dr = d0, er = e0;
        std::vector<rt> work(static_cast<std::size_t>(4 * k + 64));
        std::vector<int> iw(static_cast<std::size_t>(8 * k + 64));
        int ri = 0;
        bdsdc_n(uplo, k, dr.data(), er.data(), work, iw, ri);
        double serr = 0.0;
        for (int i = 0; i < k; ++i)
            serr = std::max(serr, std::fabs(static_cast<double>(s[static_cast<std::size_t>(i)]) -
                                            static_cast<double>(dr[static_cast<std::size_t>(i)])));
        dense.check(tag + " s-vs-?BDSDC", serr / (smax_g * std::max(k, 1) * eps), 50.0);
    }

    // ---- factors -----------------------------------------------------------------
    auto Ue = [&](int i, int j) { return static_cast<double>(U[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * k]); };
    auto Ve = [&](int i, int j) { return static_cast<double>(V[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * k]); };
    double rec = 0.0, uorth = 0.0, vorth = 0.0;
    if (!sampled) {
        for (int j = 0; j < k; ++j)
            for (int i = 0; i < k; ++i) {
                double acc = 0.0;
                for (int r = 0; r < k; ++r)
                    acc += Ue(i, r) * static_cast<double>(s[static_cast<std::size_t>(r)]) * Ve(j, r);
                rec = std::max(rec, std::fabs(acc - static_cast<double>(bval<rt>(uplo, i, j, d0, e0))));
            }
        for (int j = 0; j < k; ++j)
            for (int i = j; i < k; ++i) {
                double gu = 0.0, gv = 0.0;
                for (int r = 0; r < k; ++r) {
                    gu += Ue(r, i) * Ue(r, j);
                    gv += Ve(r, i) * Ve(r, j);
                }
                uorth = std::max(uorth, std::fabs(gu - (i == j ? 1.0 : 0.0)));
                vorth = std::max(vorth, std::fabs(gv - (i == j ? 1.0 : 0.0)));
            }
    } else {
        // B v_j = s_j u_j and orthogonality on a deterministic column sample.
        const int ns = 16;
        for (int t = 0; t < ns; ++t) {
            const int j = static_cast<int>(std::min<double>(k - 1, std::floor((hash01(t, 900) + 0.5) * k)));
            const int j2 = static_cast<int>(std::min<double>(k - 1, std::floor((hash01(t, 901) + 0.5) * k)));
            for (int i = 0; i < k; ++i) {
                double bv = 0.0;
                for (int c = std::max(0, i - 1); c <= std::min(k - 1, i + 1); ++c)
                    bv += static_cast<double>(bval<rt>(uplo, i, c, d0, e0)) * Ve(c, j);
                rec = std::max(rec, std::fabs(bv - static_cast<double>(s[static_cast<std::size_t>(j)]) * Ue(i, j)));
            }
            double gu = 0.0, gv = 0.0, nu = 0.0, nv = 0.0;
            for (int r = 0; r < k; ++r) {
                gu += Ue(r, j) * Ue(r, j2);
                gv += Ve(r, j) * Ve(r, j2);
                nu += Ue(r, j) * Ue(r, j);
                nv += Ve(r, j) * Ve(r, j);
            }
            uorth = std::max(uorth, std::fabs(gu - (j == j2 ? 1.0 : 0.0)));
            vorth = std::max(vorth, std::fabs(gv - (j == j2 ? 1.0 : 0.0)));
            uorth = std::max(uorth, std::fabs(nu - 1.0));
            vorth = std::max(vorth, std::fabs(nv - 1.0));
        }
    }
    dense.check(tag + " B-USV^T", rec / (smax_g * std::max(k, 1) * eps), 50.0);
    dense.check(tag + " U orth", uorth / (std::max(k, 1) * eps), 50.0);
    dense.check(tag + " V orth", vorth / (std::max(k, 1) * eps), 50.0);
    }
    dense.finish();

    if (ddlaGetRank(handle) == 0)
        std::cout << "[grid " << grid_name(handle) << "] pbdsdc " << tag << " ok" << std::endl;
}

// Exercise redistribution independently of the eigensolve: a slab exceeding
// 512 rows must cross a conversion-panel boundary, including its short tail.
// Exact dyadic values make the layout/conversion checks exact for every type.
template <typename rt>
rt scatter_value(int row, int col)
{
    return rt(row % 509 - 254) + rt(col % 503 - 251) / rt(1024);
}

template <typename T>
void check_rows_scatter_case(const ddla::DdlaHandle_t& handle, int k, bool transpose)
{
    using rt = typename ddla::real_type<T>::type;
    int ranks = 1, nprows = 0, npcols = 0;
    MPI_Comm_size(ddlaGetCommunicator(handle), &ranks);
    ddlaGetGridDims(handle, nprows, npcols);
    const int rank = ddlaGetRank(handle);
    const ddla::detail::RowShare share(k, ranks);
    const int rows = share.count(rank), lds = std::max(1, rows) + 3;
    const std::size_t source_size = std::size_t(lds) * std::max(1, k);
    std::vector<rt> source(source_size, std::numeric_limits<rt>::quiet_NaN());
    for (int col = 0; col < k; ++col)
        for (int row = 0; row < rows; ++row)
            source[row + std::size_t(col) * lds] = scatter_value<rt>(share.lo(rank) + row, col);

    // Both extensions are intentional: the helper zeros the whole logical
    // rectangle outside the factor, but must preserve descriptor/LLD padding.
    const int m = k + 5, n = k + 7, pad = 3;
    int desc[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(desc, handle, m + pad, n + pad, 7, 11,
                            nprows - 1, npcols - 1));
    desc[ddla::DDLA_LLD_] += 4;
    T sentinel;
    if constexpr (std::is_same<T, rt>::value) sentinel = T(-19);
    else sentinel = T(-19, 7);
    const std::size_t dest_size = std::max<std::size_t>(1, local_size(handle, desc));
    std::vector<T> destination(dest_size, sentinel);
    DeviceBuffer<rt> d_source(handle, source_size);
    DeviceBuffer<T> d_destination(handle, dest_size);
    upload(handle, d_source.ptr, source);
    upload(handle, d_destination.ptr, destination);
    check_ddla_sync(handle);

#if DDLA_HAS_GPU
    if (ddlaGetBackend(handle) == DdlaBackend::CPU)
        ddla::matrix_impl::scatter_rows<DdlaBackend::CPU, T>(handle, k, d_source.ptr, lds,
            transpose, m, n, d_destination.ptr, desc);
    else
        ddla::matrix_impl::scatter_rows<DdlaBackend::GPU, T>(handle, k, d_source.ptr, lds,
            transpose, m, n, d_destination.ptr, desc);
#elif DDLA_HAS_GPU
    ddla::matrix_impl::scatter_rows<DdlaBackend::GPU, T>(handle, k, d_source.ptr, lds,
        transpose, m, n, d_destination.ptr, desc);
#else
    ddla::matrix_impl::scatter_rows<DdlaBackend::CPU, T>(handle, k, d_source.ptr, lds,
        transpose, m, n, d_destination.ptr, desc);
#endif

    const auto actual = download(handle, d_destination.ptr, dest_size);
    auto expected = destination;
    const int local_rows = ddla_test::m_loc(handle, desc);
    const int local_cols = ddla_test::n_loc(handle, desc);
    for (int col = 0; col < local_cols; ++col) {
        const int gc = ddla::indx_l2g_c(desc, handle, col);
        for (int row = 0; row < local_rows; ++row) {
            const int gr = ddla::indx_l2g_r(desc, handle, row);
            if (gr < m && gc < n)
                expected[row + std::size_t(col) * desc[ddla::DDLA_LLD_]] =
                    gr < k && gc < k ? T(transpose ? scatter_value<rt>(gc, gr)
                                                   : scatter_value<rt>(gr, gc)) : T(0);
        }
    }
    double error = 0.0;
    for (std::size_t i = 0; i < dest_size; ++i) {
        const double e = static_cast<double>(std::abs(actual[i] - expected[i]));
        if (!std::isfinite(e)) { error = std::numeric_limits<double>::infinity(); break; }
        error = std::max(error, e);
    }
    const std::string tag = std::string("rows_scatter ") +
        (ddlaGetBackend(handle) == DdlaBackend::CPU ? "cpu " : "gpu ") +
        (std::is_same<T, rt>::value ? "real " : "complex ") + tag_rt<rt>() +
        " k=" + std::to_string(k) + (transpose ? " transpose" : " direct");
    require_close(handle, tag + " values and padding", error, 0.0);
    const auto after = download(handle, d_source.ptr, source_size);
    require_close(handle, tag + " source unchanged",
                  std::memcmp(after.data(), source.data(), source_size * sizeof(rt)) == 0 ? 0.0 : 1.0, 0.0);
}

// Rejection must be collective before any rank enters a panel transfer.
void check_rows_scatter_failures(const ddla::DdlaHandle_t& h)
{
    constexpr int k=17;int ranks=0;MPI_Comm_size(ddlaGetCommunicator(h),&ranks);
    const int rank=ddlaGetRank(h);const ddla::detail::RowShare share(k,ranks);
    const int ld=std::max(1,share.count(rank));
    int desc[ddla::DDLA_DLEN_];ddlaDescInit(desc,h,k,k,7,7,0,0);
    DeviceBuffer<double> src(h,std::size_t(ld)*k),dst(h,std::max<std::size_t>(1,local_size(h,desc)));
    upload(h,src.ptr,std::vector<double>(std::size_t(ld)*k,1));
    for(int mode=0;mode<4;++mode) {
        const bool root=rank==0;
        const double* input=mode==0&&root?nullptr:src.ptr;
        const int lds=mode==1&&root?ld-1:ld;
        const int m=mode==2&&root?k+1:k;
        const int order=mode==3&&root&&ranks>1?k-1:k;
        if(mode==3&&ranks==1)continue;
        bool caught=false;
        try {
#if DDLA_HAS_GPU
            if(ddlaGetBackend(h)==DdlaBackend::CPU)
                ddla::matrix_impl::scatter_rows<DdlaBackend::CPU,double>(h,order,input,lds,false,m,k,dst.ptr,desc);
            else ddla::matrix_impl::scatter_rows<DdlaBackend::GPU,double>(h,order,input,lds,false,m,k,dst.ptr,desc);
#else
            ddla::matrix_impl::scatter_rows<DdlaBackend::CPU,double>(h,order,input,lds,false,m,k,dst.ptr,desc);
#endif
        }catch(const std::exception&){caught=true;}
        require_close(h,"row scatter invalid input rejected collectively",caught?0:1,0);
    }
}

void check_rows_scatter(const ddla::DdlaHandle_t& handle)
{
    int ranks = 1;
    MPI_Comm_size(ddlaGetCommunicator(handle), &ranks);
    check_rows_scatter_failures(handle);
    std::vector<int> sizes{0, 1, std::max(1, ranks - 1), 512 * ranks + 3};
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    for (int k : sizes)
        for (bool transpose : {false, true}) {
            check_rows_scatter_case<float>(handle, k, transpose);
            check_rows_scatter_case<double>(handle, k, transpose);
            check_rows_scatter_case<std::complex<float>>(handle, k, transpose);
            check_rows_scatter_case<std::complex<double>>(handle, k, transpose);
        }
}

void check_body(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    (void)base;
    check_rows_scatter(handle);
#if DDLA_HAS_GPU
    if (ddlaGetBackend(handle) == DdlaBackend::GPU) {
        int nprows = 0, npcols = 0;
        ddlaGetGridDims(handle, nprows, npcols);
        ddla::DdlaHandle_t cpu = nullptr;
        ddla::ddlaInit(cpu, DdlaBackend::CPU);
        ddla::ddlaSet(cpu, ddlaGetCommunicator(handle), nprows, npcols);
        check_rows_scatter(cpu);
        ddla::ddlaDestroy(cpu);
    }
#endif
    // k <= leaf goes through host LASDQ; k > leaf through distributed D&C
    // (k = 17 with leaf 16 is the smallest real merge).
    const int ks[] = {1, 2, 3, 5, 17, 40, 63, 120, 257, 601};
    const int leaves[] = {16, 25, 64};
    for (int cls = 0; cls < C_COUNT; ++cls)
        for (const int k : ks)
            for (const char uplo : {'U', 'L'})
                for (const int leaf : leaves) {
                    if (leaf >= k && leaf != leaves[0]) continue;      // one host-path leg per k
                    if (k == 601 && leaf == 25) continue;
                    run_case<double>(handle, cls, k, uplo, leaf, false);
                }
    // float
    for (int cls = 0; cls < C_COUNT; ++cls)
        for (const int k : {5, 63, 257})
            for (const char uplo : {'U', 'L'})
                run_case<float>(handle, cls, k, uplo, 25, false);
    // one large problem, sampled checks (leaf 64: ~6 merge levels)
    for (const char uplo : {'U', 'L'}) {
        run_case<double>(handle, C_RAND, 2000, uplo, 64, true);
        run_case<double>(handle, C_SPLIT, 2000, uplo, 64, true);
        run_case<double>(handle, C_DEFL, 2000, uplo, 64, true);
        run_case<double>(handle, C_DIAG, 2000, uplo, 64, true);    // ~2000 bounded leaf jobs
    }
    unsetenv("DDLA_SVD_DC_LEAF");
}

} // namespace

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pbdsdc", check_body);
}
