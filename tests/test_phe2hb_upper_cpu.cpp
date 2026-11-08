#include <cstring>
#include <type_traits>

#include "api_grid_test_common.h"
#include "Eigensolver/eig_params.h"
#include "eig_test_common.h"
#include "Eigensolver/two_stage/phe2hb_impl.h"
#include "upper_prepared_oracle.h"

// ---------------------------------------------------------------------------
// Isolation test for stage 1 (phe2hb) and stage 5 (punmtr_he2hb).
//
// Stage 1: reduce a full Hermitian matrix to banded form, reconstruct
// A = Q^H B Q on the host from the banded part and the stored reflectors, and
// compare against the input.
//
// Stage 5: run both transform directions on the identity and check them
// against the host-built Q and Q^H, then check P B P^H = A for each.
//
// Output contract exercised:
//   - upper triangle: band entries at rows [j-nbw, j] plus the reflector
//     vector at rows < j-nbw (anchor 1 at row j-nbw implicit);
//   - d_tmat: replicated nbw x nbw x nsteps slots, diagonal = raw tau.
//
// Tolerances follow the `50 * n * eps` convention documented in
// eig_test_common.h.  The previously fixed 1e-11 (5e-4 for float) sat ~10^5
// above the observed error, so a five-orders-of-magnitude regression would
// have stayed green; the observed errors are ~2-3 n eps.
// ---------------------------------------------------------------------------

using namespace api_grid_test;

namespace {

template <typename U> U conj_v2(U v) { return v; }
template <typename U> std::complex<U> conj_v2(std::complex<U> v) { return std::conj(v); }

template <typename T>
T herm_input(int i,int j,bool structured=false){return structured?upper_prepared_oracle::value<T>(i,j):upper_prepared_oracle::full_value<T>(i,j);}

// Gather a distributed matrix's leading n x n block: every rank ends up with
// the full block (zero-fill + MPI_Allreduce).  The reduction datatype is
// selected from the scalar type rather than derived from sizeof(T), which
// integer-divided to 0 for single precision.
template <typename T>
std::vector<T> gather_full(const ddla::DdlaHandle_t& handle, const T* d_A,
                           const int* desc, int n)
{
    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);
    const int m_loc = ddla::num_loc(n, desc[ddla::DDLA_MB_], myprow, desc[ddla::DDLA_RSRC_], nprows);
    const int n_loc = ddla::num_loc(n, desc[ddla::DDLA_NB_], mypcol, desc[ddla::DDLA_CSRC_], npcols);
    const int lld = desc[ddla::DDLA_LLD_];
    const auto local = download<T>(handle, d_A, local_size(handle, desc));
    std::vector<T> full(static_cast<std::size_t>(n) * n, T(0));
    for (int j = 0; j < n_loc; ++j) {
        const int gj = ddla::indx_l2g_c(desc, handle, j);
        if (gj >= n) continue;
        for (int i = 0; i < m_loc; ++i) {
            const int gi = ddla::indx_l2g_r(desc, handle, i);
            if (gi >= n) continue;
            full[gi + static_cast<std::size_t>(gj) * n] =
                local[i + static_cast<std::size_t>(j) * lld];
        }
    }
    MPI_Comm comm = ddlaGetCommunicator(handle);
    if constexpr (std::is_same<T, float>::value) {
        MPI_Allreduce(MPI_IN_PLACE, full.data(), static_cast<int>(full.size()),
                      MPI_FLOAT, MPI_SUM, comm);
    } else if constexpr (std::is_same<T, double>::value) {
        MPI_Allreduce(MPI_IN_PLACE, full.data(), static_cast<int>(full.size()),
                      MPI_DOUBLE, MPI_SUM, comm);
    } else if constexpr (std::is_same<T, std::complex<float>>::value) {
        MPI_Allreduce(MPI_IN_PLACE, full.data(), static_cast<int>(full.size()),
                      MPI_C_FLOAT_COMPLEX, MPI_SUM, comm);
    } else {
        MPI_Allreduce(MPI_IN_PLACE, full.data(), static_cast<int>(full.size()),
                      MPI_C_DOUBLE_COMPLEX, MPI_SUM, comm);
    }
    return full;
}

// Host reconstruction of Q from the stored reflectors: apply them to the
// identity, innermost transform first.
template <typename T>
std::vector<T> build_host_Q(const std::vector<T>& Aout, const std::vector<T>& tmat,
                            int n, int nbw, int nsteps)
{
    auto idx = [n](int i, int j) { return i + static_cast<std::size_t>(j) * n; };
    std::vector<T> Q(static_cast<std::size_t>(n) * n, T(0)), x(n);
    for (int jcol = 0; jcol < n; ++jcol) {
        std::fill(x.begin(), x.end(), T(0));
        x[jcol] = T(1);
        for (int istep = nsteps; istep >= 1; --istep) {
            const int panel_start = istep * nbw;
            const int n_cols = std::min(n, panel_start + nbw) - panel_start;
            for (int lc = n_cols - 1; lc >= 0; --lc) {
                const int j = panel_start + lc;
                const int nrow = j - nbw;
                const T tau = tmat[static_cast<std::size_t>(lc) + lc * nbw
                                   + static_cast<std::size_t>(istep - 1) * nbw * nbw];
                if (tau == T(0)) continue;
                // v = stored vector + implicit anchor 1 at nrow
                T dot{};
                for (int i = 0; i < nrow; ++i) dot += conj_v2(Aout[idx(i, j)]) * x[i];
                dot += x[nrow];   // anchor
                for (int i = 0; i < nrow; ++i) x[i] -= tau * dot * Aout[idx(i, j)];
                x[nrow] -= tau * dot;
            }
        }
        for (int i = 0; i < n; ++i) Q[idx(i, jcol)] = x[i];
    }
    return Q;
}

// Dense host products for the small test sizes.
template <typename T>
std::vector<T> host_mul(const std::vector<T>& A, const std::vector<T>& B, int n)
{
    auto idx = [n](int i, int j) { return i + static_cast<std::size_t>(j) * n; };
    std::vector<T> C(static_cast<std::size_t>(n) * n, T(0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            T acc{};
            for (int k = 0; k < n; ++k) acc += A[idx(i, k)] * B[idx(k, j)];
            C[idx(i, j)] = acc;
        }
    return C;
}

/// C = A^H * B.
template <typename T>
std::vector<T> host_mul_conjA(const std::vector<T>& A, const std::vector<T>& B, int n)
{
    auto idx = [n](int i, int j) { return i + static_cast<std::size_t>(j) * n; };
    std::vector<T> C(static_cast<std::size_t>(n) * n, T(0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            T acc{};
            for (int k = 0; k < n; ++k) acc += conj_v2(A[idx(k, i)]) * B[idx(k, j)];
            C[idx(i, j)] = acc;
        }
    return C;
}

/// C = A * B^H.
template <typename T>
std::vector<T> host_mul_conjB(const std::vector<T>& A, const std::vector<T>& B, int n)
{
    auto idx = [n](int i, int j) { return i + static_cast<std::size_t>(j) * n; };
    std::vector<T> C(static_cast<std::size_t>(n) * n, T(0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            T acc{};
            for (int k = 0; k < n; ++k) acc += A[idx(i, k)] * conj_v2(B[idx(j, k)]);
            C[idx(i, j)] = acc;
        }
    return C;
}

/// Conjugate transpose of a dense host matrix.
template <typename T>
std::vector<T> host_adjoint(const std::vector<T>& A, int n)
{
    auto idx = [n](int i, int j) { return i + static_cast<std::size_t>(j) * n; };
    std::vector<T> AH(static_cast<std::size_t>(n) * n, T(0));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) AH[idx(i, j)] = conj_v2(A[idx(j, i)]);
    return AH;
}

template <typename T>
double max_abs_diff(const std::vector<T>& X, const std::vector<T>& Y)
{
    double err = 0.0;
    for (std::size_t k = 0; k < X.size(); ++k)
        {const double d=std::abs(X[k]-Y[k]);if(!std::isfinite(d)){return std::numeric_limits<double>::infinity();}err=std::max(err,d);}
    return err;
}

// One full stage-1 + stage-5 case at the given geometry.
template <typename T>
void run_case(const ddla::DdlaHandle_t& handle, int n, int nb, int nbw, const std::string& tag,int tile,int parent=0,int pad=0,bool structured=false)
{
    using rt = typename ddla::real_type<T>::type;
    // Backward-error metric in the units eig_test_common.h documents:
    //   ||X - A||_max / (||A||_max * n * eps)   accept < 50
    // The previous fixed 1e-11 was compared against ||X - A||/(||A||_max * n)
    // only, i.e. it was ~1/(n*eps) times looser than this; a regression of
    // five orders of magnitude would have passed unnoticed.
    const double tol = 50.0;
    const double eps = eig_test::eps_trait<rt>::value;

    int desc[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(desc, handle, n+parent, n+parent, nb, nb, 0, 0));
    desc[ddla::DDLA_LLD_]+=pad;

    auto h_A = make_local<T>(handle, desc, [&](int i, int j) { return herm_input<T>(i, j,structured); });
    DeviceBuffer<T> d_A(handle, h_A.size());
    upload(handle, d_A.ptr, h_A);
    check_ddla_sync(handle);

    const int nsteps = (n - 1) / nbw;
    DeviceBuffer<T> d_tmat(handle, static_cast<std::size_t>(nbw) * nbw * std::max(nsteps, 1));
    {   // poison the T slots: phe2hb must overwrite every entry (zeros included)
        const std::vector<T> poison(static_cast<std::size_t>(nbw) * nbw * std::max(nsteps, 1), T(7));
        upload(handle, d_tmat.ptr, poison);
        check_ddla_sync(handle);
    }
    DeviceBuffer<T> d_ref(handle,h_A.size());upload(handle,d_ref.ptr,h_A);
    DeviceBuffer<T> d_ref_t(handle,static_cast<std::size_t>(nbw)*nbw*std::max(nsteps,1));
    const std::vector<T> ref_poison(static_cast<std::size_t>(nbw)*nbw*std::max(nsteps,1),T(7));
    upload(handle,d_ref_t.ptr,ref_poison);
    int ref_info=0;ddla::phe2hb<DdlaBackend::CPU,T>(handle,n,d_ref.ptr,desc,nbw,d_ref_t.ptr,ref_info);
    const auto Aref=gather_full<T>(handle,d_ref.ptr,desc,n);
    const auto Tref=download<T>(handle,d_ref_t.ptr,ref_poison.size());
    const bool expected=(nsteps*nbw)>tile;
    auto candidate=make_local<T>(handle,desc,[&](int i,int j){
        if(i>=n||j>=n)return herm_input<T>(i,j,structured);
        return expected&&i/tile>j/tile?T(std::numeric_limits<rt>::quiet_NaN()):herm_input<T>(i,j,structured);
    });
    upload(handle,d_A.ptr,candidate);
    int info = 0;
    const bool used=ddla::detail::phe2hb_upper_cpu<T>(handle,n,d_A.ptr,desc,nbw,d_tmat.ptr,info,tile);
    require_close(handle,"private upper policy "+tag,used!=expected,0);
    const auto local_after=download<T>(handle,d_A.ptr,candidate.size());
    int mr=0,mc=0,pr=0,pc=0;ddlaGetGridDims(handle,pr,pc);ddlaGetGridCoords(handle,mr,mc);
    const int am=ddla::num_loc(n+parent,nb,mr,0,pr),an=ddla::num_loc(n+parent,nb,mc,0,pc),lld=desc[ddla::DDLA_LLD_];
    double padding_bad=0;
    for(int j=0;j<an;++j)for(int i=0;i<lld;++i){const int gi=(i/nb*pr+mr)*nb+i%nb,gj=(j/nb*pc+mc)*nb+j%nb;const auto at=i+std::size_t(j)*lld;if(i>=am||gi>=n||gj>=n)padding_bad+=std::memcmp(&local_after[at],&candidate[at],sizeof(T))!=0;}
    require_close(handle,"private upper parent/padding "+tag,padding_bad,0);
    if(!used){double mismatch=0;const auto ref_local=download<T>(handle,d_ref.ptr,candidate.size());for(std::size_t q=0;q<candidate.size();++q){mismatch+=std::memcmp(&local_after[q],&ref_local[q],sizeof(T))!=0;}require_close(handle,"private full fallback bitwise "+tag,mismatch,0);}

    check_ddla_sync(handle);
    if (info != 0) {
        if (ddlaGetRank(handle) == 0) std::cerr << "phe2hb info=" << info << std::endl;
        MPI_Abort(ddlaGetCommunicator(handle), 1);
    }

    const auto Aout = gather_full<T>(handle, d_A.ptr, desc, n);
    const auto tmat = download<T>(handle, d_tmat.ptr,
                                  static_cast<std::size_t>(nbw) * nbw * std::max(nsteps, 1));

    {
        RootSection comparison(handle);
        if(comparison.root&&!structured){double upper_error=0,scale=1;for(int j=0;j<n;++j)for(int i=0;i<=j;++i){const auto q=i+std::size_t(j)*n;const double d=std::abs(Aout[q]-Aref[q]);upper_error=std::isfinite(d)?std::max(upper_error,d):std::numeric_limits<double>::infinity();scale=std::max(scale,double(std::abs(Aref[q])));}comparison.check("private upper vs full A "+tag,upper_error/(scale*std::max(n,1)*eps),tol);comparison.check("private upper vs full T "+tag,max_abs_diff(tmat,Tref)/(std::max(n,1)*eps),tol);}
        comparison.finish();
    }

    // ---- the T slots: upper triangular, zero below the diagonal and outside the panel's
    // n_cols x n_cols block (the panel step stores T_L^H into the zeroed slot), real diagonal
    // magnitude <= 2 (tau = (beta - alpha)/beta has |tau - 1| <= 1, so |tau| <= 2) ----------
    {
        double stray = 0.0, diag_bad = 0.0;
        for (int is = 1; is <= nsteps; ++is) {
            const int ps = is * nbw;
            const int ncols = std::min(n, ps + nbw) - ps;
            const T* slot = tmat.data() + static_cast<std::size_t>(is - 1) * nbw * nbw;
            for (int c = 0; c < nbw; ++c)
                for (int r = 0; r < nbw; ++r) {
                    const double v = std::abs(slot[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * nbw]);
                    if (r > c || r >= ncols || c >= ncols) stray = std::max(stray, v);
                    else if (r == c) diag_bad = std::max(diag_bad, v > 2.0 + 1e-6 ? 1.0 : 0.0);
                }
        }
        require_close(handle, "phe2hb T slots zero outside the upper triangle " + tag, stray, 0.0);
        require_close(handle, "phe2hb T diagonal is a tau " + tag, diag_bad, 0.0);
    }

    // ---- host reconstruction: A_pre = Q^H B Q -----------------------------
    // Root-only: B, Q, A_in and the O(n^3) host products below read only the
    // replicated gather_full output, so computing them on every rank repeats
    // identical dense work P times -- the dominant share of this test's check
    // phase.  The stage-5 punmtr/gather calls stay collective on all ranks;
    // only their dense comparisons move inside the root-only verdicts.
    auto idx = [n](int i, int j) { return i + static_cast<std::size_t>(j) * n; };
    // Dense check inputs; filled on the root rank only (RootSection below):
    // every input reads only the replicated gather_full output, so computing
    // them on every rank repeats identical dense work P times -- the dominant
    // share of this test's check phase.
    std::vector<T> B, Q, A_in;
    double denom = 0.0;
    RootSection dense(handle);
    if (dense.root) {
    // B = banded part of A_out (upper band mirrored to a full Hermitian matrix).
    B.assign(static_cast<std::size_t>(n) * n, T(0));
    for (int j = 0; j < n; ++j)
        for (int i = std::max(0, j - nbw); i <= j; ++i)
            B[idx(i, j)] = Aout[idx(i, j)];
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < j; ++i)
            if (j - i <= nbw) B[idx(j, i)] = conj_v2(B[idx(i, j)]);

    Q = build_host_Q(Aout, tmat, n, nbw, nsteps);

    // The input matrix and the norm scale for all three checks below.
    A_in.assign(static_cast<std::size_t>(n) * n, T(0));
    double scale = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            A_in[idx(i, j)] = herm_input<T>(i, j,structured);
            scale = std::max(scale, static_cast<double>(std::abs(A_in[idx(i, j)])));
        }
    denom = std::max(scale, 1.0) * n * eps;

    // A_rec = Q^H * B * Q
    {
        const auto A_rec = host_mul<T>(host_mul_conjA<T>(Q, B, n), Q, n);
        dense.check("phe2hb reconstruction " + tag,
                      max_abs_diff(A_rec, A_in) / denom, tol);
    }
    }

    // ---- stage 5: transform the identity in both directions ---------------
    int descZ[ddla::DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(descZ, handle, n, n, nb, nb, 0, 0));
    auto h_I = make_local<T>(handle, descZ, [](int i, int j) { return T(i == j ? 1 : 0); });

    // trans='N' must give Q;  trans='C' must give Q^H.
    for (int pass = 0; pass < 2; ++pass) {
        const char trans = (pass == 0) ? 'N' : 'C';
        const bool adjoint = (trans != 'N');
        DeviceBuffer<T> d_Z(handle, h_I.size());
        upload(handle, d_Z.ptr, h_I);
        check_ddla_sync(handle);
        int info5 = 0;
        ddla::punmtr_he2hb(handle, trans, n, nbw, d_A.ptr, desc, d_tmat.ptr,
                           d_Z.ptr, descZ, info5);
        check_ddla_sync(handle);
        if (info5 != 0) {
            if (ddlaGetRank(handle) == 0)
                std::cerr << "punmtr_he2hb('" << trans << "') info=" << info5 << std::endl;
            MPI_Abort(ddlaGetCommunicator(handle), 1);
        }
        const auto P = gather_full<T>(handle, d_Z.ptr, descZ, n);

        // P should equal the host-built Q, or its adjoint (dense work is
        // root-only; Q and B exist only there).
        if (dense.root) {
        const std::vector<T> Qref = adjoint ? host_adjoint<T>(Q, n) : Q;
        dense.check(std::string("stage5 '") + trans + "' P = " + (adjoint ? "Q^H" : "Q"),
                      max_abs_diff(P, Qref) / denom, tol);

        // The factorization is A = Q^H B Q, so applying the stage-5 transform
        // to B recovers A in both directions, just from opposite sides: with
        // P = Q (trans='N') the identity is P^H B P = A, and with P = Q^H
        // (trans='C') it is P B P^H = A.  This is exactly the reduction the
        // eigensolver driver relies on, so the two cases are checked
        // separately rather than collapsed into one symmetric-looking form.
        const auto A_chk = adjoint ? host_mul_conjB<T>(host_mul<T>(P, B, n), P, n)
                                   : host_mul<T>(host_mul_conjA<T>(P, B, n), P, n);
        const char* expr = adjoint ? "P B P^H = A" : "P^H B P = A";
        dense.check(std::string("stage5 '") + trans + "' " + expr,
                      max_abs_diff(A_chk, A_in) / denom, tol);
        }
    }
    dense.finish();
}

template<class T>
void source_rejection(const DdlaHandle_t& h) {
    int pr=0,pc=0;
    ddlaGetGridDims(h,pr,pc);
    if(pr==1&&pc==1) {return;}
    const int n=17,nb=4;
    int desc[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(desc,h,n,n,nb,nb,pr>1?1:0,pc>1?1:0));
    auto input=make_local<T>(h,desc,[](int i,int j){return herm_input<T>(i,j);});
    DeviceBuffer<T> A(h,input.size()),tm(h,64);
    upload(h,A.ptr,input);
    std::vector<T> seven(64,T(7));
    upload(h,tm.ptr,seven);
    bool caught=false;
    int info=0;
    try {detail::phe2hb_upper_cpu<T>(h,n,A.ptr,desc,nb,tm.ptr,info,nb);}
    catch(const std::exception&) {caught=true;}
    require_close(h,"upper source0 rejection",!caught,0);
    const auto after=download<T>(h,A.ptr,input.size());
    const auto after_t=download<T>(h,tm.ptr,seven.size());
    double changed=0;
    for(std::size_t q=0;q<input.size();++q) {
        changed+=std::memcmp(&after[q],&input[q],sizeof(T))!=0;
    }
    for(std::size_t q=0;q<seven.size();++q) {
        changed+=std::memcmp(&after_t[q],&seven[q],sizeof(T))!=0;
    }
    require_close(h,"upper rejection leaves A/T",changed,0);
}
template<class T>
void tile_rejection(const DdlaHandle_t& h) {
    const int n=17,nb=4;
    int desc[DDLA_DLEN_];
    DDLA_CHECK(ddlaDescInit(desc,h,n,n,nb,nb,0,0));
    auto input=make_local<T>(h,desc,[](int i,int j){return herm_input<T>(i,j);});
    DeviceBuffer<T> A(h,input.size()),tm(h,64);
    upload(h,A.ptr,input);
    std::vector<T> seven(64,T(7));upload(h,tm.ptr,seven);
    const int tile=ddlaGetSize(h)==1?0:(ddlaGetRank(h)==0?nb:2*nb);
    bool caught=false;int info=0;
    try {detail::phe2hb_upper_cpu<T>(h,n,A.ptr,desc,nb,tm.ptr,info,tile);}
    catch(const std::exception&) {caught=true;}
    require_close(h,"upper tile metadata rejection",!caught,0);
    const auto after=download<T>(h,A.ptr,input.size());
    const auto after_t=download<T>(h,tm.ptr,seven.size());
    double changed=0;
    for(std::size_t q=0;q<input.size();++q) {changed+=std::memcmp(&after[q],&input[q],sizeof(T))!=0;}
    for(std::size_t q=0;q<seven.size();++q) {changed+=std::memcmp(&after_t[q],&seven[q],sizeof(T))!=0;}
    require_close(h,"upper tile rejection leaves A/T",changed,0);
}
template<class T>void run_type(const DdlaHandle_t&h){
 for(int n:{0,1,2,9,17,41})for(int tile:{4,8}){const auto r=upper_prepared_oracle::run<T>(h,n,4,tile,5,3);require_close(h,"upper prepared AV",r.av,50);require_close(h,"upper prepared update",r.update,50);require_close(h,"upper prepared padding/poison",r.changed_padding,0);}
 run_case<T>(h,9,4,4,"empty-ranks",4);
 run_case<T>(h,17,4,4,"tile4",4);
 run_case<T>(h,41,4,2,"partial",8);
 run_case<T>(h,73,8,4,"tile8",8);
 run_case<T>(h,73,8,4,"tile16",16);
 run_case<T>(h,33,8,2,"parent-padding",16,5,3);
 // Near-null targets may choose different reflector phases; validate the
 // reduction and both stage5 directions, not equality of those coefficients.
 run_case<T>(h,17,4,4,"structured-phase",8,0,0,true);
 run_case<T>(h,17,4,4,"default-small-fallback",2560);
 run_case<T>(h,1,4,4,"n1-fallback",4);
 run_case<T>(h,4,4,4,"no-panel-fallback",4);
 source_rejection<T>(h);tile_rejection<T>(h);
}
void check_body(const DdlaHandle_t&h,const Shape&){if(ddlaGetBackend(h)!=DdlaBackend::CPU)throw std::runtime_error("CPU upper test needs CPU");run_type<float>(h);run_type<double>(h);run_type<std::complex<float>>(h);run_type<std::complex<double>>(h);}
} // namespace
int main(int argc,char**argv){return run_grid_test(argc,argv,"test_phe2hb_upper_cpu",check_body);}
