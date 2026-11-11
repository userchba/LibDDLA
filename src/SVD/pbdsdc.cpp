// Bidiagonal SVD by row-distributed divide and conquer on CPU and GPU.
// U and V have O(k^2/P) storage per rank. Only scalar vectors and bounded
// coefficient panels/leaf factors are replicated; see pbdsdc_impl.h.
// LAPACK-style LASD2 deflation and LASD3 secular arithmetic retain 1-based
// host indices, converted only at the backend boundary.
#include <ddla/ddla.h>
#include "pbdsdc_impl.h"
#include "check_agree.h"
#include "BLAS/blas_impl.h"
#include <ddla/ddla_vector.h>
#include <optional>

#include "ddla_connector.h"
#include "comm_traits.h"
#include "dc_common.h"
#include "dc_threads.h"
#include "ddla_stream.h"
#include "mpi_datatype.h"
#include "plan_util.h"
#include "row_share.h"
#include "matrix_transfer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// Host LAPACK/BLAS through the Fortran ABI (hidden CHARACTER lengths omitted,
// as in src/pstedc.cpp).  OpenBLAS's LAPACK: the same library code the
// reference ?BDSDC runs.
extern "C" {
double dlamch_(const char*);
float slamch_(const char*);
double dlanst_(const char*, const int*, const double*, const double*);
float slanst_(const char*, const int*, const float*, const float*);
void dlasd4_(const int*, const int*, const double*, const double*, double*,
             const double*, double*, double*, int*);
void slasd4_(const int*, const int*, const float*, const float*, float*,
             const float*, float*, float*, int*);
double dlapy2_(const double*, const double*);
float slapy2_(const float*, const float*);
void dlascl_(const char*, const int*, const int*, const double*, const double*,
             const int*, const int*, double*, const int*, int*);
void slascl_(const char*, const int*, const int*, const float*, const float*,
             const int*, const int*, float*, const int*, int*);
double dnrm2_(const int*, const double*, const int*);
float snrm2_(const int*, const float*, const int*);
void dlasdq_(const char*, const int*, const int*, const int*, const int*, const int*,
             double*, double*, double*, const int*, double*, const int*, double*,
             const int*, double*, int*);
void slasdq_(const char*, const int*, const int*, const int*, const int*, const int*,
             float*, float*, float*, const int*, float*, const int*, float*,
             const int*, float*, int*);
}

namespace ddla {

namespace {

// ---- typed Fortran ABI dispatch --------------------------------------------

void lasdq_host(char uplo, int sqre, int n, int ncvt, int nru, int ncc,
                float* d, float* e, float* vt, int ldvt, float* u, int ldu,
                float* c, int ldc, float* work, int& info)
{
    int i2 = 0;
    slasdq_(&uplo, &sqre, &n, &ncvt, &nru, &ncc, d, e, vt, &ldvt, u, &ldu, c, &ldc, work, &i2);
    info = i2;
}
void lasdq_host(char uplo, int sqre, int n, int ncvt, int nru, int ncc,
                double* d, double* e, double* vt, int ldvt, double* u, int ldu,
                double* c, int ldc, double* work, int& info)
{
    int i2 = 0;
    dlasdq_(&uplo, &sqre, &n, &ncvt, &nru, &ncc, d, e, vt, &ldvt, u, &ldu, c, &ldc, work, &i2);
    info = i2;
}
void lasd4(int n, int i, const double* d, const double* z, double* delta,
           const double& rho, double& dlm, double* work, int& info)
{
    dlasd4_(&n, &i, d, z, delta, &rho, &dlm, work, &info);
}
void lasd4(int n, int i, const float* d, const float* z, float* delta,
           const float& rho, float& dlm, float* work, int& info)
{
    slasd4_(&n, &i, d, z, delta, &rho, &dlm, work, &info);
}
double lapst(double a, double b) { return dlapy2_(&a, &b); }
float lapst(float a, float b) { return slapy2_(&a, &b); }
template <typename rt> rt eps_machine();
template <> double eps_machine<double>() { return dlamch_("Epsilon"); }
template <> float eps_machine<float>() { return slamch_("Epsilon"); }
// DLASCL('G'): some BLAS builds dereference KL/KU/M/N/LDA unconditionally, so
// every argument is a real object.
void lascl_g(const double& cfrom, const double& cto, int m, int n,
             double* a, int lda, int& info)
{
    int kl = 0, ku = 0, info2 = 0;
    dlascl_("G", &kl, &ku, &cfrom, &cto, &m, &n, a, &lda, &info2);
    info = info2;
}
void lascl_g(const float& cfrom, const float& cto, int m, int n,
             float* a, int lda, int& info)
{
    int kl = 0, ku = 0, info2 = 0;
    slascl_("G", &kl, &ku, &cfrom, &cto, &m, &n, a, &lda, &info2);
    info = info2;
}
template <typename rt>
rt nrm2(int n, const rt* x)
{
    const int inc = 1;
    if constexpr (std::is_same<rt, double>::value) return static_cast<rt>(dnrm2_(&n, x, &inc));
    else return snrm2_(&n, x, &inc);
}
template <typename rt>
rt lanst_m(int n, const rt* d, const rt* e)
{
    if constexpr (std::is_same<rt, double>::value) return dlanst_("M", &n, d, e);
    else return slanst_("M", &n, d, e);
}

// DLASDT's exact index-math port; node values stay 1-based as dlasdt.f writes
// them, the arrays are 0-based.
void pbdc_dlasdt(int n, int& lvl, int& nd, int* inode, int* ndiml, int* ndimr, int msub)
{
    const int maxn = std::max(1, n);
    const double temp = std::log(static_cast<double>(maxn) / static_cast<double>(msub + 1))
                      / std::log(2.0);
    lvl = static_cast<int>(temp) + 1;
    const int i = n / 2;
    inode[0] = i + 1;
    ndiml[0] = i;
    ndimr[0] = n - i - 1;
    int il = -1, ir = 0, llst = 1;
    for (int nlvl = 1; nlvl < lvl; ++nlvl) {
        for (int ii = 0; ii < llst; ++ii) {
            il += 2;
            ir += 2;
            const int ncrnt = llst + ii - 1;
            ndiml[il] = ndiml[ncrnt] / 2;
            ndimr[il] = ndiml[ncrnt] - ndiml[il] - 1;
            inode[il] = inode[ncrnt] - ndimr[il] - 1;
            ndiml[ir] = ndimr[ncrnt] / 2;
            ndimr[ir] = ndimr[ncrnt] - ndiml[ir] - 1;
            inode[ir] = inode[ncrnt] + ndiml[ir] + 1;
        }
        llst *= 2;
    }
    nd = llst * 2 - 1;
}

// Largest tree (and leaf) size solved by ONE host call: dc_leaf_size(
// "DDLA_SVD_DC_LEAF", 128).  Host DLASDQ costs O(L^3) at BLAS-1 speed, so the
// default is small; the tests use 16..64 to force deep trees.
// ---------------------------------------------------------------------------
// pbdc_lasd2: DLASD2's decisions, ported 1-based (element 0 of every array is
// unused).  d: 1-based working slice (N = NL+NR+1; mutated by the shift, the
// merge sort and the deflated-tail copy, as dlasd2 does); z: 1-based, M =
// N+SQRE, PRE-BUILT by the caller from the device (z(1) = ALPHA*VT(NLP1,
// NLP1), Z(I+1) = ALPHA*VT(I,NLP1) for I = 1..NL, Z(I) = BETA*VT(I,NLP2) for
// I = NLP2..M).  idxq: the children's sort permutation, 1-based.
// Outputs: k; dsigma (1-based); z(1..K) = dlasd3's input vector; idx/idxc/
// idxp/coltyp (1-based); gathsrc(J) = the 1-based LOCAL source column of
// U2/VT2 column J (J = 2..N); the deflation Givens appended to givcol/givnum
// (1-based LOCAL column pairs, (c, s)); c2/s2 = the VT2-row-1 / VT-row-M
// special factors (with m_gt_n).
// ---------------------------------------------------------------------------
template <typename rt>
void pbdc_lasd2_run(int nl, int nr, int sqre, int& k, rt* d, rt* z, rt& alpha, rt& beta,
                rt* dsigma, int* idx, int* idxc, int* idxp, int* coltyp,
                int* idxq, int* gathsrc,
                std::array<int,2>* givcol,std::array<rt,2>* givnum,int& rotations,rt* zc1,
                rt& c2,rt& s2,bool& m_gt_n)
{
    rotations=0;idxc[1]=0;gathsrc[1]=0;
    const int n = nl + nr + 1;
    const int m = n + sqre;
    const int nlp1 = nl + 1;
    const int nlp2 = nl + 2;
    m_gt_n = (m > n);
    c2 = rt(1);
    s2 = rt(0);

    std::fill_n(zc1,static_cast<std::size_t>(n)+2,rt(0)); // U2(:,1) of dlasd2
    const rt z1 = z[1];

    // loop 10: D(I+1) = D(I), IDXQ(I+1) = IDXQ(I) + 1
    for (int i = nl; i >= 1; --i) {
        d[i + 1] = d[i];
        idxq[i + 1] = idxq[i] + 1;
    }
    // loops 30..50: column types and the right child's IDXQ shift
    for (int i = 2; i <= nlp1; ++i) coltyp[i] = 1;
    for (int i = nlp2; i <= n; ++i) coltyp[i] = 2;
    for (int i = nlp2; i <= n; ++i) idxq[i] += nlp1;

    // loop 60, DLAMRG(NL, NR, DSIGMA(2), 1, 1, IDX(2)), loop 70
    for (int i = 2; i <= n; ++i) {
        dsigma[i] = d[idxq[i]];
        zc1[i] = z[idxq[i]];
        idxc[i] = coltyp[idxq[i]];
    }
    lamrg(nl, nr, dsigma + 2, 1, 1, idx + 2);
    for (int i = 2; i <= n; ++i) {
        const int idxi = 1 + idx[i];
        d[i] = dsigma[idxi];
        z[i] = zc1[idxi];
        coltyp[i] = idxc[idxi];
    }

    // tolerance: 8*8*eps*max(|D(N)|, |ALPHA|, |BETA|)
    const rt tol0 = std::max(std::fabs(alpha), std::fabs(beta));
    const rt tol = static_cast<rt>(64) * eps_machine<rt>() * std::max(std::fabs(d[n]), tol0);

    // deflation scan (labels 80..120)
    k = 1;
    int k2 = n + 1;
    int jprev = 0;
    bool exited120 = false;
    for (int j = 2; j <= n; ++j) {
        if (std::fabs(z[j]) <= tol) {
            --k2;
            idxp[k2] = j;
            coltyp[j] = 4;
            if (j == n) { exited120 = true; break; }
        } else {
            jprev = j;
            break;
        }
    }
    if (!exited120) {
        int j = jprev;
        while (true) {
            ++j;
            if (j > n) {                       // label 110: the last singular value
                ++k;
                zc1[k] = z[jprev];
                dsigma[k] = d[jprev];
                idxp[k] = jprev;
                break;
            }
            if (std::fabs(z[j]) <= tol) {
                --k2;
                idxp[k2] = j;
                coltyp[j] = 4;
            } else if (d[j] - d[jprev] <= tol) {
                rt s_ = z[jprev];
                rt c_ = z[j];
                const rt tau = lapst(c_, s_);
                c_ = c_ / tau;
                s_ = -s_ / tau;
                z[j] = tau;
                z[jprev] = rt(0);
                int idxjp = idxq[idx[jprev] + 1];
                int idxj = idxq[idx[j] + 1];
                if (idxjp <= nlp1) --idxjp;
                if (idxj <= nlp1) --idxj;
                givcol[rotations]={idxjp,idxj};givnum[rotations++]={c_,s_};
                if (coltyp[j] != coltyp[jprev]) coltyp[j] = 3;
                coltyp[jprev] = 4;
                // DEVIATION from dlasd2 (which stores only IDXP(K2) = JPREV): the columns jprev..j-1
                // (jprev and the small-z columns deflated since) are written as the contiguous
                // descending-index block k2 .. k2+j-1-jprev.  That keeps the deflated tail in
                // decreasing D order, which is DLASD1's DLAMRG(K, N-K, D, 1, -1, IDXQ) precondition;
                // the reference order differs from this one only among columns whose D values are
                // within TOL of each other, i.e. the results differ by at most the deflation
                // tolerance (tests/test_api_grid_pbdsdc.cpp, class "defl", pins the accuracy).
                --k2;
                for (int jp = jprev; jp < j; ++jp) idxp[k2 + j - 1 - jp] = jp;
                jprev = j;
            } else {
                ++k;
                zc1[k] = z[jprev];
                dsigma[k] = d[jprev];
                idxp[k] = jprev;
                jprev = j;
            }
        }
    }

    // loops 130..150: CTOT / PSM / IDXC
    int ctot[5] = {0, 0, 0, 0, 0};
    for (int j = 2; j <= n; ++j) ++ctot[coltyp[j]];
    int psm[5];
    psm[1] = 2;
    psm[2] = 2 + ctot[1];
    psm[3] = psm[2] + ctot[2];
    psm[4] = psm[3] + ctot[3];
    for (int j = 2; j <= n; ++j) {
        const int jp = idxp[j];
        const int ct = coltyp[jp];
        idxc[psm[ct]] = j;
        ++psm[ct];
    }

    // loop 160: DSIGMA rebuild + the U2/VT2 gather plan
    for (int j = 2; j <= n; ++j) {
        const int jp = idxp[j];
        dsigma[j] = d[jp];
        int idxj = idxq[idx[idxp[idxc[j]]] + 1];
        if (idxj <= nlp1) --idxj;
        gathsrc[j] = idxj;
    }

    // DSIGMA(1), Z(1) and the VT2-row-1 / VT-row-M special factors
    dsigma[1] = rt(0);
    const rt hlftol = tol / rt(2);
    if (std::fabs(dsigma[2]) <= hlftol) dsigma[2] = hlftol;
    if (m > n) {
        z[1] = lapst(z1, z[m]);
        if (z[1] <= tol) {
            c2 = rt(1);
            s2 = rt(0);
            z[1] = tol;
        } else {
            c2 = z1 / z[1];
            s2 = z[m] / z[1];
        }
    } else {
        if (std::fabs(z1) <= tol) z[1] = tol;
        else z[1] = z1;
    }
    for (int i = 2; i <= k; ++i) z[i] = zc1[i];

    // D(K+1..N) = the deflated DSIGMA tail
    for (int j = k + 1; j <= n; ++j) d[j] = dsigma[j];
}

// ---------------------------------------------------------------------------
// DLASD3's secular part, split by singular-value column.  Columns [a, b)
// (0-based) <-> Fortran J = a+1 .. b.  delta/work: K x cw, ld K.
// ---------------------------------------------------------------------------

// DLASD4 for each owned column; dsol[c] = the root.  Returns the first
// nonzero DLASD4 info (0 = ok).  z: 1-based, ALREADY normalized (|z| = 1).
template <typename rt>
int pbdc_roots(int K, const rt* dsigma, const rt* z, rt rho, int a, int b,
               rt* delta, rt* work, rt* dsol, int nth,detail::DcHostTeam* team=nullptr)
{
    // columns are independent (DLASD4 touches only its own delta/work column): dynamic chunks, the
    // first nonzero info of any column is kept
    std::atomic<int> err(0);
    detail::dc_parallel_for(team,nth, b - a, 8, [&](int c0, int c1) {
        for (int c = c0; c < c1; ++c) {
            if (err.load(std::memory_order_relaxed) != 0) return;
            int jinfo = 0;
            lasd4(K, a + 1 + c, dsigma + 1, z + 1, delta + static_cast<std::size_t>(c) * K,
                  rho, dsol[c], work + static_cast<std::size_t>(c) * K, jinfo);
            if (jinfo != 0) {
                int none = 0;
                err.compare_exchange_strong(none, jinfo);
                return;
            }
        }
    });
    return err.load();
}

// Partial product of the Loewner z-hat over the owned columns: dlasd3's loops
// 60/40/50 with the j-loop restricted to the chunk (column K contributes the
// unscaled base term, the others one ratio each).  zp: 1-based, K entries.
template <typename rt>
void pbdc_zhat_partial(int K, const rt* dsigma, int a, int b, const rt* delta,
                       const rt* work, rt* zp, int nth,detail::DcHostTeam* team=nullptr)
{
    // split over the ROWS i: every zp[i] keeps its serial column order, so the result is bitwise
    // identical to the single-threaded loop
    detail::dc_parallel_for(team,nth, K, std::max(512, (K + 2 * nth - 1) / (2 * nth)), [&](int i0, int i1) {
        for (int i = i0 + 1; i <= i1; ++i) zp[i] = rt(1);
        for (int c = 0; c < b - a; ++c) {
            const int j = a + 1 + c;
            const rt* dc = delta + static_cast<std::size_t>(c) * K;
            const rt* wc = work + static_cast<std::size_t>(c) * K;
            if (j == K) {
                for (int i = i0 + 1; i <= i1; ++i) zp[i] = zp[i] * (dc[i - 1] * wc[i - 1]);
            } else {
                for (int i = i0 + 1; i <= i1; ++i) {
                    const rt prod = dc[i - 1] * wc[i - 1];
                    if (j < i)
                        zp[i] = zp[i] * (prod / (dsigma[i] - dsigma[j]) / (dsigma[i] + dsigma[j]));
                    else
                        zp[i] = zp[i] * (prod / (dsigma[i] - dsigma[j + 1]) / (dsigma[i] + dsigma[j + 1]));
                }
            }
        }
    });
}

// Singular-vector blocks of the secular problem for the owned columns
// (dlasd3 loops 90 and 120 merged): ql(:, c) = the left vector (DLASD3's
// Q(:, I)), and the right vector W(:, c) = Qr(I, :)^T overwrites delta's
// column c.  zh: the final z-hat (signs applied), 1-based.  When ql aliases
// work, left_scratch supplies K * min(nth,cw) entries: keep one normalized
// left vector there until the raw right vector in work has been consumed.
template <typename rt>
void pbdc_q_chunk(int K, const rt* dsigma, const rt* zh, const int* idxc, int cw,
                  rt* delta, rt* work, rt* ql, int nth, rt* left_scratch = nullptr,detail::DcHostTeam* team=nullptr)
{
    auto column = [&](int c, rt* qc) {
        rt* const uci = delta + static_cast<std::size_t>(c) * K;
        rt* const vci = work + static_cast<std::size_t>(c) * K;
        vci[0] = zh[1] / uci[0] / vci[0];
        uci[0] = rt(-1);
        for (int j = 2; j <= K; ++j) {
            const rt ui = uci[j - 1], vi = vci[j - 1];
            vci[j - 1] = zh[j] / ui / vi;
            uci[j - 1] = dsigma[j] * vci[j - 1];
        }
        const rt tu = nrm2<rt>(K, uci);
        qc[0] = uci[0] / tu;
        for (int j = 2; j <= K; ++j) qc[j - 1] = uci[idxc[j] - 1] / tu;
        const rt tv = nrm2<rt>(K, vci);
        uci[0] = vci[0] / tv;
        for (int j = 2; j <= K; ++j) uci[j - 1] = vci[idxc[j] - 1] / tv;
    };
    if (left_scratch != nullptr) {
        const int tasks = std::min(nth, cw);
        // Logical tasks, rather than physical worker ids, own scratch. Each
        // task processes a disjoint contiguous column range serially.
        detail::dc_parallel_for(team,nth, tasks, 1, [&](int t0, int t1) {
            for (int t = t0; t < t1; ++t) {
                rt* const qc = left_scratch + static_cast<std::size_t>(t) * K;
                const int c0 = static_cast<int>(static_cast<long long>(t) * cw / tasks);
                const int c1 = static_cast<int>(static_cast<long long>(t + 1) * cw / tasks);
                for (int c = c0; c < c1; ++c) {
                    column(c, qc);
                    std::copy_n(qc, K, ql + static_cast<std::size_t>(c) * K);
                }
            }
        });
    } else {
        detail::dc_parallel_for(team,nth, cw, 4, [&](int c0, int c1) {
            for (int c = c0; c < c1; ++c)
                column(c, ql + static_cast<std::size_t>(c) * K);
        });
    }
}

} // namespace

#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
namespace {
constexpr unsigned kThreads = 256;

template <typename rt>
void launch_givens(const DdlaHandle_t& handle, rt* q, int ldq, int m, const int* givcol, const rt* givnum, int ng)
{
    const gpuStream_t st = static_cast<gpuStream_t>(handle->stream);
    if (ng <= 0 || m <= 0) return;
    dc_givens_kernel<rt><<<(static_cast<unsigned>(m) + kThreads - 1) / kThreads, kThreads, 0, st>>>(
        q, ldq, m, givcol, givnum, ng);
    DDLA_CHECK(ddlaGetLastError(handle));
}
} // namespace
#endif

namespace {
// Only length-k vectors and bounded panels are replicated. Matrix storage
// is balanced over the entire communicator.
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
template <typename rt>
__global__ void pbdc_rows_identity_kernel(rt* u, rt* v, int ld, int rows, int lo)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) u[r + std::size_t(lo+r)*ld] = v[r + std::size_t(lo+r)*ld] = rt(1);
}
template <typename rt>
__global__ void pbdc_rows_special_kernel(rt* u2, rt* v2, int ld,
                                        const rt* vb, int ldv, int rows,
                                        int row0, int middle, int last, rt c, rt s,
                                        bool rectangular)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int gr = row0 + r;
    u2[r] = (gr == middle) ? rt(1) : rt(0);
    const rt x = vb[r + std::size_t(middle) * ldv];
    if (!rectangular) v2[r] = x;
    else {
        const rt y = vb[r + std::size_t(last) * ldv];
        v2[r] = gr <= middle ? c * x : s * y;
        v2[r + std::size_t(last) * ld] = gr <= middle ? -s * x : c * y;
    }
}

#endif

template <DdlaBackend B, typename rt>
void pbdc_rows_gather(const DdlaHandle_t& h, rt* dst, int ldd, const rt* src,
                      int lds, int rows, const int* perm, int count,
                      int* device_perm)
{
    if (!rows || count <= 0) return;
    if constexpr (B == DdlaBackend::CPU) {
        for (std::size_t j = 0; j < std::size_t(count); ++j)
            std::copy_n(src + std::size_t(perm[j]-1)*lds, rows, dst+j*ldd);
    } else {
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
        launch_gather(h, dst, ldd, src, lds, rows, device_perm, count);
#endif
    }
}

// Width of the broadcast coefficient panels: fixed, independent of n.
constexpr int kPbdcRowsPanel = 512;

// A root broadcasts only its owned secular columns. The panel capacity need
// not cover more than the largest rank share, even when n <= the panel cap.
int pbdc_panel_width(int n, int ranks)
{
    return std::min(kPbdcRowsPanel, detail::RowShare(n, ranks).count(0));
}

// Host records of pbdsdc_run: an independent tree (its index arrays at
// `storage` in the tree index) and a leaf (its retained factors at `storage`).
struct PbdsdcTree {int off=0,n=0,level=0,nodes=0;std::size_t storage=0;};
struct PbdsdcLeaf {int off=0,n=0,m=0,sqre=0;std::size_t storage=0;};

// Reals of the leaf phase's per-leaf buffers, L = min(n, leaf): the receive buffer
// [values | U | V] of one leaf (n <= L, m <= L + 1) and the DLASDQ scratch, u and vt
// of order L + 2, e (L + 2) and the 4 (L + 2) + 512 work.
inline std::size_t pbdc_receive_count(int L) { return 2 * std::size_t(L + 1) * (L + 1) + L; }
// The tree construction bounds the total leaf count by n, including
// empty children. Exact zeros can split it into n independent leaves.
inline int pbdc_leaf_workers(const spectral_impl::PbdsdcPlan& p)
{
    const int owned = p.rank < p.n ? 1 + (p.n - 1 - p.rank) / p.ranks : 0;
    return std::min(p.threads, owned);
}

inline std::size_t pbdc_leaf_temp_count(int L)
{
    const std::size_t ldl = std::size_t(L) + 2;
    return 2 * ldl * ldl + 5 * ldl + 512;
}

// Host bytes of PbdcRows' merge arrays, in the order its constructor takes them:
// d1, z, zsave, dsigma, zp, zc1, row1, row2 (n + 3 each), the secular columns
// delta / work (n x max(1, rows) each), left_scratch (n x min(threads, rows)),
// roots (max(1, rows)), idx, idxc, idxp, coltyp, idxq, gath (n + 3 each),
// decisions / root_decisions (2n + 3 each), permutation (n + 3), counts / displs
// (P each) and the deflation rotations (n column pairs, n (c, s) pairs).
template <typename rt>
std::size_t pbdc_rows_host_bytes(int n, int P, int rows, int threads)
{
    using detail::padded_bytes;
    const std::size_t vec = std::size_t(n) + 3, share = std::max(1, rows);
    return 8 * padded_bytes<rt>(vec) + 2 * padded_bytes<rt>(std::size_t(n) * share)
         + padded_bytes<rt>(std::size_t(n) * std::min(threads, rows)) + padded_bytes<rt>(share)
         + 6 * padded_bytes<int>(vec) + 2 * padded_bytes<int>(2 * std::size_t(n) + 3)
         + padded_bytes<int>(vec) + 2 * padded_bytes<int>(std::size_t(P))
         + padded_bytes<std::array<int, 2>>(std::size_t(n)) + padded_bytes<std::array<rt, 2>>(std::size_t(n));
}

// Backend bytes of PbdcRows, in the same order: the slabs u2 / v2 (max(1, rows) x n
// each), the coefficient panels qd (2n x min(n, kPbdcRowsPanel)) and, on a GPU handle,
// the device permutation (n), rotation columns (2n) and rotation numbers (2n).
template <typename rt>
std::size_t pbdc_rows_backend_bytes(bool gpu, int n, int rows, int ranks)
{
    using detail::padded_bytes;
    std::size_t bytes = 2 * padded_bytes<rt>(std::size_t(std::max(1, rows)) * n)
                      + padded_bytes<rt>(2 * std::size_t(n) * pbdc_panel_width(n, ranks));
    if (gpu)
        bytes += padded_bytes<int>(std::size_t(n)) + padded_bytes<int>(2 * std::size_t(n))
               + padded_bytes<rt>(2 * std::size_t(n));
    return bytes;
}

template <DdlaBackend B, typename rt>
struct PbdcRows {
    const DdlaHandle_t& h;
    const int n, P, me, lo, rows, ld;
    const detail::RowShare share;
    rt *U, *V;
    rt *u2=nullptr,*v2=nullptr,*qd=nullptr,*gndev=nullptr;
    int *permdev=nullptr,*gcdev=nullptr;
    detail::DcHostTeam& team;
    int nth=1,rotations=0;
    const int panel;
    rt *d1,*z,*zsave,*dsigma,*zp,*zc1,*delta,*work,*left_scratch,*roots,*row1,*row2;
    int *idx,*idxc,*idxp,*coltyp,*idxq,*gath,*decisions,*root_decisions,*permutation,*counts,*displs;
    std::array<int,2>* gc;
    std::array<rt,2>* gn;
    // Carves the merge arrays (pbdc_rows_host_bytes / pbdc_rows_backend_bytes).
    PbdcRows(const DdlaHandle_t& handle,const spectral_impl::PbdsdcPlan& p,detail::DcHostTeam& prepared,
        rt* u,rt* v,void* backend_work,[[maybe_unused]] std::size_t backend_bytes,
        void* host_work,[[maybe_unused]] std::size_t host_bytes)
      : h(handle),n(p.n),P(p.ranks),me(p.rank),lo(detail::RowShare(n,P).lo(me)),
        rows(detail::RowShare(n,P).count(me)),ld(p.ld),share(n,P),U(u),V(v),team(prepared),nth(p.threads),panel(pbdc_panel_width(n,P))
    {
        using detail::take;
        const std::size_t vec=std::size_t(n)+3,slab=std::size_t(std::max(1,rows))*n;
        char* const hbase=static_cast<char*>(host_work);
        char* hw=hbase;
        d1=take<rt>(hw,vec);z=take<rt>(hw,vec);zsave=take<rt>(hw,vec);dsigma=take<rt>(hw,vec);
        zp=take<rt>(hw,vec);zc1=take<rt>(hw,vec);row1=take<rt>(hw,vec);row2=take<rt>(hw,vec);
        delta=take<rt>(hw,slab);work=take<rt>(hw,slab);left_scratch=take<rt>(hw,std::size_t(n)*std::min(nth,rows));
        roots=take<rt>(hw,std::max(1,rows));
        idx=take<int>(hw,vec);idxc=take<int>(hw,vec);idxp=take<int>(hw,vec);
        coltyp=take<int>(hw,vec);idxq=take<int>(hw,vec);gath=take<int>(hw,vec);
        decisions=take<int>(hw,2*std::size_t(n)+3);root_decisions=take<int>(hw,2*std::size_t(n)+3);
        permutation=take<int>(hw,vec);counts=take<int>(hw,P);displs=take<int>(hw,P);
        gc=take<std::array<int,2>>(hw,n);gn=take<std::array<rt,2>>(hw,n);
        assert(std::size_t(hw-hbase)<=host_bytes);
        char* const bbase=static_cast<char*>(backend_work);
        char* bw=bbase;
        u2=take<rt>(bw,slab);v2=take<rt>(bw,slab);qd=take<rt>(bw,2*std::size_t(n)*panel);
        if constexpr(B==DdlaBackend::GPU) {
            permdev=take<int>(bw,n);gcdev=take<int>(bw,2*std::size_t(n));gndev=take<rt>(bw,2*std::size_t(n));
        }
        assert(std::size_t(bw-bbase)<=backend_bytes);
    }
    void copy(rt* dst,int ldd,const rt* src,int lds,int nr,int nc) {
        if (nr && nc) DDLA_CHECK(ddlaMemcpy2D(h,dst,std::size_t(ldd)*sizeof(rt),src,
                    std::size_t(lds)*sizeof(rt),std::size_t(nr)*sizeof(rt),nc));
    }
    // Only the owner reads a connecting row; scalar decisions see identical data.
    const rt* read_row(rt* out,int row,int col,int count) {
        const int owner=share.owner(row);
        if (me==owner) {
            DDLA_CHECK(ddlaMemcpy2D(h,out,sizeof(rt),V+row-lo+std::size_t(col)*ld,
                        std::size_t(ld)*sizeof(rt),sizeof(rt),count));
            DDLA_CHECK(ddlaSynchronize(h));
        }
        MPI_CHECK(MPI_Bcast(out,count,detail::mpi_datatype<rt>(),owner,h->comm));
        return out;
    }
    void merge(int g0,int s0,int nl,int nr,int sqre,int* treeq,rt* d,rt* e) {
        const int nn=nl+nr+1, mm=nn+sqre, middle=nl;
        const int begin=std::max(lo,g0), end=std::min(lo+rows,g0+mm);
        const int rv=std::max(0,end-begin), ru=std::max(0,std::min(end,g0+nn)-begin);
        const int ld2=std::max(1,rv), row0=begin-g0;
        // Use valid base pointers even for a rank with no rows in this node.
        rt* ub=U+std::size_t(g0)*ld+(ru ? begin-lo : 0);
        rt* vb=V+std::size_t(g0)*ld+(rv ? begin-lo : 0);
        auto r1=read_row(row1,g0+nl,g0,nl+1);
        auto r2=read_row(row2,g0+nl+1,g0+nl+1,mm-nl-1);
        for(int i=1;i<=nn;++i){d1[i]=d[g0+i-1];idxq[i]=treeq[s0+i-1];}
        rt alpha=d[g0+nl],beta=e[g0+nl],org=std::max(std::fabs(alpha),std::fabs(beta));
        d1[nl+1]=rt(0);
        for(int i=1;i<=nn;++i)org=std::max(org,std::fabs(d1[i]));
        int ierr=0;
        lascl_g(org,rt(1),nn,1,d1+1,nn,ierr);
        alpha/=org; beta/=org;
        z[1]=alpha*r1[nl];
        for(int i=1;i<=nl;++i)z[i+1]=alpha*r1[i-1];
        for(int i=nl+2;i<=mm;++i)z[i]=beta*r2[i-nl-2];
        int K=0;rt c=1,s=0;bool rectangular=false;
        pbdc_lasd2_run(nl,nr,sqre,K,d1,z,alpha,beta,dsigma,idx,
                    idxc,idxp,coltyp,idxq,gath,gc,gn,rotations,zc1,c,s,rectangular);
        // Verify the complete decisions before different counts enter collectives.
        const auto decision_count=2*std::size_t(nn)+3;
        std::fill_n(decisions,decision_count,0);
        decisions[0]=K; decisions[1]=rotations;
        for(int j=1;j<=nn;++j){decisions[2*j]=gath[j];decisions[2*j+1]=idxc[j];}
        std::copy_n(decisions,decision_count,root_decisions);
        MPI_CHECK(MPI_Bcast(root_decisions,int(decision_count),MPI_INT,0,h->comm));
        agree_or_throw(h->comm,!std::equal(decisions,decisions+decision_count,root_decisions),"pbdsdc","merge agreement");
        if(rotations) {
            if constexpr(B==DdlaBackend::CPU) {
                for(std::size_t g=0;g<static_cast<std::size_t>(rotations);++g) {
                    auto rotate=[&](rt* a,int rr){
                        for(int r=0;r<rr;++r){rt& x=a[r+std::size_t(gc[g][0]-1)*ld];
                            rt& y=a[r+std::size_t(gc[g][1]-1)*ld];
                            const rt old=x;x=gn[g][0]*old+gn[g][1]*y;y=gn[g][0]*y-gn[g][1]*old;}
                    }; rotate(ub,ru);rotate(vb,rv);
                }
            } else {
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
                // Copies and rotations share the handle stream. Host metadata
                // stays unchanged until the completion fence at the merge end.
                DDLA_CHECK(ddlaMemcpy(h, gcdev, gc, std::size_t(rotations)*2*sizeof(int)));
                DDLA_CHECK(ddlaMemcpy(h, gndev, gn, std::size_t(rotations)*2*sizeof(rt)));
                if(ru)launch_givens(h,ub,ld,ru,gcdev,gndev,rotations);
                if(rv)launch_givens(h,vb,ld,rv,gcdev,gndev,rotations);
#endif
            }
        }
        int* const perm=permutation;
        for(int j=2;j<=nn;++j)perm[j-2]=gath[j];
        if constexpr (B == DdlaBackend::GPU) {
            if ((ru || rv) && nn > 1) {
                DDLA_CHECK(ddlaMemcpy(h, permdev, perm, std::size_t(nn - 1) * sizeof(int)));
            }
        }
        pbdc_rows_gather<B>(h,u2+ld2,ld2,ub,ld,ru,perm,nn-1,permdev);
        pbdc_rows_gather<B>(h,v2+ld2,ld2,vb,ld,rv,perm,nn-1,permdev);
        if constexpr(B==DdlaBackend::CPU) {
            for(int r=0;r<rv;++r) {
                const int gr=row0+r;
                u2[r]=gr==middle?rt(1):rt(0);
                const rt x=vb[r+std::size_t(middle)*ld];
                if(!rectangular)v2[r]=x;
                else {const rt y=vb[r+std::size_t(mm-1)*ld];
                    v2[r]=gr<=middle?c*x:s*y;
                    v2[r+std::size_t(mm-1)*ld2]=gr<=middle?-s*x:c*y;}
            }
        } else {
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
            if(rv)pbdc_rows_special_kernel<<<(rv+255)/256,256,0,static_cast<gpuStream_t>(h->stream)>>>(u2,v2,ld2,
                          vb,ld,rv,row0,middle,mm-1,c,s,rectangular);
            DDLA_CHECK(ddlaGetLastError(h));
#endif
        }
        if(K==1) {
            d1[1]=std::fabs(z[1]);
            copy(ub,ld,u2,ld2,ru,1);copy(vb,ld,v2,ld2,rv,1);
            if(ru && z[1]<=rt(0))blas_impl::scal_run<B,rt>(h,ru,rt(-1),ub,1);
        } else {
            for(int i=1;i<=K;++i)zsave[i]=z[i];
            rt rho=nrm2(K,z+1);
            lascl_g(rho,rt(1),K,1,z+1,K,ierr);rho*=rho;
            const detail::RowShare columns(K, P);
            const int a=columns.lo(me),b=columns.hi(me),cw=columns.count(me);
            const int threads=static_cast<long long>(K)*cw>=32768?nth:1;
            int err=pbdc_roots(K,dsigma,z,rho,a,b,delta,work,roots,threads,&team);
            agree_or_throw(h->comm,err,"pbdsdc","secular roots");
            for(int r=0;r<P;++r){counts[r]=columns.count(r);displs[r]=columns.lo(r);}
            MPI_CHECK(MPI_Allgatherv(roots,cw,detail::mpi_datatype<rt>(),d1+1,
                                      counts,displs,detail::mpi_datatype<rt>(),h->comm));
            pbdc_zhat_partial(K,dsigma,a,b,delta,work,zp,threads,&team);
            MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,zp+1,K,detail::mpi_datatype<rt>(),MPI_PROD,h->comm));
            for(int i=1;i<=K;++i)zp[i]=std::copysign(std::sqrt(std::fabs(zp[i])),zsave[i]);
            pbdc_q_chunk(K,dsigma,zp,idxc,cw,delta,work,work,
                          threads,left_scratch,&team);
            // Broadcast coefficient panels, never factor matrices. Both factors
            // update only their owned rows; all ranks enter every broadcast.
            for(int root=0;root<P;++root)for(int j=displs[root];j<displs[root]+counts[root];j+=panel) {
                const int width=std::min(panel,displs[root]+counts[root]-j);
                const std::size_t size=std::size_t(K)*width;
                if(me==root) {
                    // These column blocks stay alive and unchanged until the
                    // single sync below; queue uploads with the broadcast/GEMMs.
                    DDLA_CHECK(ddlaMemcpy(h, qd, work+std::size_t(j-a)*K, size*sizeof(rt)));
                    DDLA_CHECK(ddlaMemcpy(h, qd+size, delta+std::size_t(j-a)*K, size*sizeof(rt)));
                }
                commBcast<B>(h,CommScope::Grid,qd,2*size,root);
                if(ru)blas_impl::gemm_run<B,rt>(h,'N','N',ru,width,K,rt(1),u2,ld2,qd,K,
                                  rt(0),ub+std::size_t(j)*ld,ld);
                if(rv)blas_impl::gemm_run<B,rt>(h,'N','N',rv,width,K,rt(1),v2,ld2,qd+size,K,
                                  rt(0),vb+std::size_t(j)*ld,ld);
            }
        }
        copy(ub+std::size_t(K)*ld,ld,u2+std::size_t(K)*ld2,ld2,ru,nn-K);
        copy(vb+std::size_t(K)*ld,ld,v2+std::size_t(K)*ld2,ld2,rv,mm-K);
        lascl_g(rt(1),org,nn,1,d1+1,nn,ierr);
        lamrg(K,nn-K,d1+1,1,-1,idxq+1);
        for(int i=1;i<=nn;++i){d[g0+i-1]=d1[i];treeq[s0+i-1]=idxq[i];}
        // Retire all uploads before their host arrays are reused, including
        // the K==1 path. Later operations never need an intermediate fence.
        if constexpr(B==DdlaBackend::GPU) DDLA_CHECK(ddlaSynchronize(h));
    }
};
} // namespace

// Internal entry: real U,V with all columns but only RowShare(k,P)'s rows.
// Lower B is the transpose of upper B with the same d,e: swap the factors,
// avoiding a dependent inter-rank row-rotation sweep.
namespace spectral_impl {
void pbdsdc_freeze_policy(const DdlaHandle_t& h,PbdsdcPlan& p)
{
    if(!p.n)return;
    if(!p.rank)p.leaf=std::min(256,dc_leaf_size("DDLA_SVD_DC_LEAF",128));
    MPI_CHECK(MPI_Bcast(&p.leaf,1,MPI_INT,0,h->comm));
    int onnode=1;MPI_Comm node=MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(h->comm,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&node));
    MPI_CHECK(MPI_Comm_size(node,&onnode));MPI_CHECK(MPI_Comm_free(&node));
    p.threads=p.n>p.leaf?detail::dc_host_threads(onnode,p.backend==DdlaBackend::CPU?h->host_threads:0):1;
}

PbdsdcPlan pbdsdc_check(const DdlaHandle_t& h,DdlaBackend backend,char uplo,int n,
    const void* d,const void* e,const void* s,const void* U,const void* V,int ld)
{
    auto p=detail::agree_check(h,"pbdsdc",[&](detail::Agree& agreement) {
        agreement.mode("spectral operation",3112);PbdsdcPlan result;
        if(!h||ddlaGetBackend(h)!=backend)throw std::invalid_argument("pbdsdc: invalid handle/backend");
        result.backend=backend;result.n=n;result.ld=ld;result.ranks=h->nprocs;result.rank=h->myid;
        result.uplo=char(std::toupper(static_cast<unsigned char>(uplo)));
        if(n<0||(result.uplo!='U'&&result.uplo!='L'))throw std::invalid_argument("pbdsdc: invalid n/uplo");
        if(result.ranks<1||result.rank<0||result.rank>=result.ranks)
            throw std::invalid_argument("pbdsdc: invalid communicator rank");
        const detail::RowShare share(n,result.ranks);
        if(n&&(!d||!s||!U||!V||(n>1&&!e)||ld<std::max(1,share.count(result.rank))))
            throw std::invalid_argument("pbdsdc: invalid input/output pointer or row stride");
        return result;
    });
    const int signature[]={3112,int(backend),int(p.uplo),n,p.ranks};
    detail::agree_signature(h,"pbdsdc",signature);
    pbdsdc_freeze_policy(h,p);
    return p;
}

// Component 2 (pbdsdc_impl.h): host [ trees | tree index | max(leaf phase, merge
// phase) ], backend the merge arrays of PbdcRows.
template <typename rt>
void pbdsdc_workspace_size(const DdlaHandle_t&,const PbdsdcPlan& p,
    std::size_t* backend_bytes,std::size_t* host_bytes)
{
    using detail::padded_bytes;
    *backend_bytes=0;*host_bytes=0;
    if(!p.n)return;
    const std::size_t n=p.n;const int L=std::min(p.n,p.leaf);
    const int rows=detail::RowShare(p.n,p.ranks).count(p.rank);
    const std::size_t leaf_phase=padded_bytes<PbdsdcLeaf>(n)+padded_bytes<rt>(pbdsdc_retained_bound(p))
        +padded_bytes<rt>(pbdc_receive_count(L))
        +padded_bytes<int>(pbdc_leaf_workers(p))
        +padded_bytes<rt>(std::size_t(pbdc_leaf_workers(p))*pbdc_leaf_temp_count(L));
    *host_bytes=padded_bytes<PbdsdcTree>(n)+padded_bytes<int>(4*n)
        +std::max(leaf_phase,pbdc_rows_host_bytes<rt>(p.n,p.ranks,rows,p.threads));
    *backend_bytes=pbdc_rows_backend_bytes<rt>(p.backend==DdlaBackend::GPU,p.n,rows,p.ranks);
}

// Component 3.
template <DdlaBackend B, typename rt>
void pbdsdc_run(const DdlaHandle_t& h,const PbdsdcPlan& p,detail::DcHostTeam& team,
    rt* d,rt* e,rt* s,rt* U,rt* V,int& info,
    void* backend_work,std::size_t backend_bytes,void* host_work,std::size_t host_bytes)
{
    using detail::take;
    info=0;const int k=p.n;if(!k)return;
    try {
    if(p.uplo=='L')std::swap(U,V);
    const int P=p.ranks,me=p.rank,ld=p.ld,leaf=p.leaf;
    const detail::RowShare share(k,P);const int lo=share.lo(me),rows=share.count(me);
    // Preserve padding when a private caller supplies a noncompact slab.
    if(rows) {
        if(ld==rows) {DDLA_CHECK(ddlaMemset(h, U, 0, std::size_t(ld)*k*sizeof(rt)));DDLA_CHECK(ddlaMemset(h, V, 0, std::size_t(ld)*k*sizeof(rt)));}
        else for(int j=0;j<k;++j) {DDLA_CHECK(ddlaMemset(h, U+std::size_t(j)*ld, 0, rows*sizeof(rt)));DDLA_CHECK(ddlaMemset(h, V+std::size_t(j)*ld, 0, rows*sizeof(rt)));}
    }
    if constexpr (B == DdlaBackend::CPU) {
        for (int r=0;r<rows;++r) U[r+std::size_t(lo+r)*ld]=V[r+std::size_t(lo+r)*ld]=rt(1);
    } else {
#if defined(DDLA_USE_CUDA) || defined(DDLA_USE_HIP)
        if(rows)pbdc_rows_identity_kernel<<<(rows+255)/256,256,0,static_cast<gpuStream_t>(h->stream)>>>(U,V,ld,rows,lo);
        DDLA_CHECK(ddlaGetLastError(h));
#endif
    }
    const rt org=lanst_m(k,d,e);
    if(org==rt(0)){std::fill_n(s,k,rt(0));DDLA_CHECK(ddlaSynchronize(h));return;}
    int ierr=0;lascl_g(org,rt(1),k,1,d,k,ierr);
    if(k>1)lascl_g(org,rt(1),k-1,1,e,k-1,ierr);
    const rt eps=rt(0.9)*eps_machine<rt>();
    if(k>leaf)for(int i=0;i<k;++i)if(std::fabs(d[i])<eps)d[i]=std::copysign(eps,d[i]);
    char* const hbase=static_cast<char*>(host_work);
    char* hw=hbase;
    PbdsdcTree* const trees=take<PbdsdcTree>(hw,k);int* const tree_index=take<int>(hw,4*std::size_t(k));
    // The leaf phase and the merge phase share the rest.
    char* const phase=hw;
    const std::size_t phase_bytes=host_bytes-std::size_t(phase-hbase);
    int tree_count=0,start=0;std::size_t storage=0;
    for(int i=0;i<k;++i)if(i==k-1||(k>leaf&&std::fabs(e[i])<eps)) {
        auto& t=trees[tree_count++];t={start,i-start+1,0,0,storage};storage+=4*std::size_t(t.n);
        if(t.n>leaf) {
            int* inode=tree_index+t.storage;int* nl=inode+t.n;int* nr=nl+t.n;
            pbdc_dlasdt(t.n,t.level,t.nodes,inode,nl,nr,leaf);
            std::fill_n(nr+t.n,t.n,0);
        }
        start=i+1;
    }
    {
        char* lw=phase;
        PbdsdcLeaf* const leaves=take<PbdsdcLeaf>(lw,k);int leaf_count=0;
        for(int j=0;j<tree_count;++j) {
            auto& t=trees[j];
            if(t.n<=leaf){leaves[leaf_count++]={t.off,t.n,t.n,0,0};continue;}
            int* inode=tree_index+t.storage;int* nl=inode+t.n;int* nr=nl+t.n;int* order=nr+t.n;
            for(int node=(t.nodes+1)/2;node<=t.nodes;++node) {
                const int ic=inode[node-1]-1,left=nl[node-1],right=nr[node-1],sqre=node==t.nodes?0:1;
                leaves[leaf_count++]={t.off+ic-left,left,left+1,1,0};
                leaves[leaf_count++]={t.off+ic+1,right,right+sqre,sqre,0};
                for(int i=0;i<left;++i)order[ic-left+i]=i+1;
                for(int i=0;i<right;++i)order[ic+1+i]=i+1;
            }
        }
        assert(leaf_count<=k);
        const int L=std::min(k,leaf);const std::size_t retained_count=pbdsdc_retained_bound(p);
        rt* const retained=take<rt>(lw,retained_count);
        rt* const recv = take<rt>(lw, pbdc_receive_count(L));
        const int owned = me < leaf_count ? 1 + (leaf_count - 1 - me) / P : 0;
        const int workers = std::min(p.threads, owned);
        int* const errors = take<int>(lw, pbdc_leaf_workers(p));
        const auto work_count = pbdc_leaf_temp_count(L);
        rt* const temporary = take<rt>(lw, std::size_t(pbdc_leaf_workers(p)) * work_count);
        assert(std::size_t(lw - hbase) <= host_bytes);
        const int largest = L + 2;
        const auto square = std::size_t(largest) * largest;
        std::size_t used = 0;
        for (int j = me; j < leaf_count; j += P) {
            auto& l = leaves[j];
            l.storage = used;
            used += l.n + std::size_t(l.n) * l.n + std::size_t(l.m) * l.m;
            assert(used <= retained_count);
        }
        std::atomic<std::size_t> next{0};
        {
            std::optional<detail::ScopedBlasThreads> serial_blas;
            if (workers > 1)
                serial_blas.emplace(1);
            team.run_roles(workers, [&](int worker) {
                rt* u = temporary + std::size_t(worker) * work_count;
                rt* vt = u + square;
                rt* ee = vt + square;
                rt* work = ee + largest;
                int error = 0;
                for (;;) {
                    const auto task = next.fetch_add(1, std::memory_order_relaxed);
                    if (task >= static_cast<std::size_t>(owned))
                        break;
                    const auto& l = leaves[me + static_cast<int>(task) * P];
                    const int ldl = l.m + 1;
                    rt* values = retained + l.storage;
                    rt* lu = values + l.n;
                    rt* lv = lu + std::size_t(l.n) * l.n;
                    std::copy_n(d + l.off, l.n, values);
                    std::fill_n(u, std::size_t(ldl) * ldl, rt(0));
                    std::fill_n(vt, std::size_t(ldl) * ldl, rt(0));
                    std::fill_n(ee, ldl, rt(0));
                    for (int i = 0; i < l.n; ++i)
                        u[i + std::size_t(i) * ldl] = rt(1);
                    for (int i = 0; i < l.m; ++i)
                        vt[i + std::size_t(i) * ldl] = rt(1);
                    for (int i = 0; i + 1 < l.m; ++i)
                        ee[i] = e[l.off + i];
                    int ei = 0;
                    lasdq_host('U', l.sqre, l.n, l.m, l.n, 0, values, ee, vt, ldl, u, ldl, u, ldl, work,
                               ei);
                    if (ei)
                        error = ei;
                    for (int c = 0; c < l.n; ++c)
                        for (int r = 0; r < l.n; ++r)
                            lu[r + std::size_t(c) * l.n] = u[r + std::size_t(c) * ldl];
                    for (int c = 0; c < l.m; ++c)
                        for (int r = 0; r < l.m; ++r)
                            lv[r + std::size_t(c) * l.m] = vt[c + std::size_t(r) * ldl];
                }
                errors[worker] = error;
            });
        }
        int error = 0;
        for (int worker = 0; worker < workers; ++worker)
            if (errors[worker])
                error = errors[worker];
        agree_or_throw(h->comm, error, "pbdsdc", "leaf DLASDQ");
        for (int j = 0; j < leaf_count; ++j) {
            auto& l = leaves[j];
            const int owner = j % P;
            rt* values = owner == me ? retained + l.storage : recv;
            rt* lu = values + l.n;
            rt* lv = lu + std::size_t(l.n) * l.n;
            const int payload = l.n + l.n * l.n + l.m * l.m;
            MPI_CHECK(MPI_Bcast(values, payload, detail::mpi_datatype<rt>(), owner, h->comm));
            std::copy_n(values, l.n, d + l.off);
            const int begin = std::max(lo, l.off),
                      ru = std::max(0, std::min(lo + rows, l.off + l.n) - begin),
                      rv = std::max(0, std::min(lo + rows, l.off + l.m) - begin);
            if (ru)
                DDLA_CHECK(ddlaMemcpy2D(h, U + begin - lo + std::size_t(l.off) * ld,
                                        std::size_t(ld) * sizeof(rt), lu + begin - l.off,
                                        std::size_t(l.n) * sizeof(rt), std::size_t(ru) * sizeof(rt),
                                        l.n));
            if (rv)
                DDLA_CHECK(ddlaMemcpy2D(h, V + begin - lo + std::size_t(l.off) * ld,
                                        std::size_t(ld) * sizeof(rt), lv + begin - l.off,
                                        std::size_t(l.m) * sizeof(rt), std::size_t(rv) * sizeof(rt),
                                        l.m));
            // The receive buffer is reused by the next broadcast. Both uploads
            // must retire before that reuse, with one fence for the pair.
            if (ru || rv)
                DDLA_CHECK(ddlaSynchronize(h));
        }
    }
    // Leaf metadata/factors have expired; the merge arrays overlay them.
    PbdcRows<B,rt> C(h,p,team,U,V,backend_work,backend_bytes,phase,phase_bytes);
    for(int j=0;j<tree_count;++j) {
        auto& t=trees[j];int* inode=tree_index+t.storage;int* nl=inode+t.n;int* nr=nl+t.n;
        for(int level=t.level;level>=1;--level) {
            const int first=1<<(level-1),last=2*first-1;
            for(int node=first;node<=last;++node) {
                const int left=nl[node-1],right=nr[node-1],s0=inode[node-1]-left-1;
                C.merge(t.off+s0,s0,left,right,node==last?0:1,nr+t.n,d,e);
            }
        }
    }
    lascl_g(rt(1),org,k,1,d,k,ierr);
    // Bottom-up stable merge uses retired decision scratch. Equal singular
    // values retain exactly the original column order, without stable_sort's
    // implementation-dependent temporary allocation.
    int* permutation=C.permutation;int* temporary=C.decisions;
    for(int j=0;j<k;++j)permutation[j]=j+1;
    for(std::size_t width=1;width<static_cast<std::size_t>(k);width*=2) {
        for(std::size_t begin=0;begin<static_cast<std::size_t>(k);begin+=2*width) {
            const auto middle=std::min(std::size_t(k),begin+width),end=std::min(std::size_t(k),begin+2*width);
            auto a=begin,b=middle;
            for(auto out=begin;out<end;++out) {
                const bool left=b==end||(a<middle&&!(d[permutation[b]-1]>d[permutation[a]-1]));
                temporary[out]=left?permutation[a++]:permutation[b++];
            }
        }
        std::copy_n(temporary,k,permutation);
    }
    for(int j=0;j<k;++j)s[j]=d[permutation[j]-1];std::copy_n(s,k,d);
    if constexpr (B == DdlaBackend::GPU) {
        if (rows) {
            DDLA_CHECK(ddlaMemcpy(h, C.permdev, permutation, std::size_t(k) * sizeof(int)));
        }
    }
    pbdc_rows_gather<B>(h,C.u2,std::max(1,rows),U,ld,rows,permutation,k,C.permdev);
    pbdc_rows_gather<B>(h,C.v2,std::max(1,rows),V,ld,rows,permutation,k,C.permdev);
    C.copy(U,ld,C.u2,std::max(1,rows),rows,k);C.copy(V,ld,C.v2,std::max(1,rows),rows,k);
    DDLA_CHECK(ddlaSynchronize(h));
    } catch (...) {
        // Uploaded host metadata and caller-owned scratch must outlive queued work.
        if constexpr(B==DdlaBackend::GPU) {
            try { (void)ddlaSynchronize(h); } catch (...) {}
        }
        throw;
    }
}
} // namespace spectral_impl

template<DdlaBackend B,typename R>
void pbdsdc(const DdlaHandle_t& h,char uplo,int n,R* d,R* e,R* s,R* U,R* V,int ld,int& info)
{
    info=0;const auto p=spectral_impl::pbdsdc_check(h,B,uplo,n,d,e,s,U,V,ld);
    if(!n)return;
    std::size_t backend_bytes=0,host_bytes=0;
    vector<char> backend_work(h);std::vector<char> host_work;std::optional<detail::DcHostTeam> team;
    std::exception_ptr failure;
    try{spectral_impl::pbdsdc_workspace_size<R>(h,p,&backend_bytes,&host_bytes);
        backend_work.resize(backend_bytes);host_work.resize(host_bytes);team.emplace(p.threads);}
    catch(...){failure=std::current_exception();}
    detail::agree_failure(h->comm,failure,"pbdsdc: workspace preparation");
    spectral_impl::pbdsdc_run<B,R>(h,p,*team,d,e,s,U,V,info,
        backend_work.data(),backend_bytes,host_work.data(),host_bytes);
}

#define INSTANTIATE_ROWS(B,R) \
 template void pbdsdc<B,R>(const DdlaHandle_t&,char,int,R*,R*,R*,R*,R*,int,int&); \
 template void spectral_impl::pbdsdc_run<B,R>(const DdlaHandle_t&,const spectral_impl::PbdsdcPlan&, \
     detail::DcHostTeam&,R*,R*,R*,R*,R*,int&,void*,std::size_t,void*,std::size_t)
#define INSTANTIATE_ROWS_SIZE(R) \
 template void spectral_impl::pbdsdc_workspace_size<R>(const DdlaHandle_t&,const spectral_impl::PbdsdcPlan&, \
     std::size_t*,std::size_t*)
INSTANTIATE_ROWS(DdlaBackend::CPU,float);
INSTANTIATE_ROWS(DdlaBackend::CPU,double);
#if DDLA_HAS_GPU
INSTANTIATE_ROWS(DdlaBackend::GPU,float);
INSTANTIATE_ROWS(DdlaBackend::GPU,double);
#endif
INSTANTIATE_ROWS_SIZE(float);
INSTANTIATE_ROWS_SIZE(double);
#undef INSTANTIATE_ROWS_SIZE
#undef INSTANTIATE_ROWS

} // namespace ddla
