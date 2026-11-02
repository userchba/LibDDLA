#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include "api_grid_test_common.h"
#include "Eigensolver/pvector_redist.h"
#include "ddla_stream.h"

// The node-local BLAS entries checked below (gemv/trmv/nrm2/dotc/hemv/her2/
// hemm/her2k/herk) are the backend-templated public API declared in
// <ddla/ddla.h>; the per-family wrapper headers under src/ no longer exist.

// ---------------------------------------------------------------------------
// Isolation test for the eigensolver vector-redistribution helpers:
//   - ptranspose_vectors: row-distributed -> column-distributed layout,
//     checked against the analytic global value, plus the accumulate mode.
//   - preduce_add_vectors: per-column partials summed across process rows
//     and transposed into the column layout, checked against the analytic
//     column sum.
// Plus a node-local check of the backend-templated public entries (gemv/trmv/
// nrm2/dotc/hemv/her2/hemm/her2k/herk) against plain host reference loops.
// ---------------------------------------------------------------------------

using namespace api_grid_test;

namespace {

template <typename T>
T redist_value(int i, int j)
{
    using rt = typename ddla::real_type<T>::type;
    const rt re = static_cast<rt>(0.25 * ((i * 7 + j * 13) % 11) - 1.0);
    const rt im = static_cast<rt>(0.125 * ((i * 5 + j * 3) % 7) - 0.25);
    if constexpr (std::is_same_v<T, rt>) {
        return T(re);   // real scalar: T(re, im) is not a valid functional cast
    } else {
        return T(re, im);
    }
}

// std::conj of a real argument returns std::complex<U> (and would poison the
// real-type instantiations), so conjugation goes through this overload set.
template <typename U> U conj_v(U v) { return v; }
template <typename U> std::complex<U> conj_v(std::complex<U> v) { return std::conj(v); }

// Full Hermitian matrix from the lower triangle produced by redist_value.
// The diagonal is forced real: BLAS Hermitian routines (xHEMV/xHEMM on both
// OpenBLAS and cuBLAS) use only the real part of the stored diagonal, so a
// complex diagonal would make the reference disagree with the wrappers.
template <typename T>
T herm_value(int i, int j)
{
    using rt = typename ddla::real_type<T>::type;
    if (i == j) {
        const rt re = std::real(redist_value<T>(i, j));
        return T(re);
    }
    if (i > j) return redist_value<T>(i, j);
    return conj_v(redist_value<T>(j, i));
}

// The node-local entries are backend-templated public API: the template
// argument has to match the handle's backend (every entry validates that), and
// a build only provides the arms it was configured for.  @p Backend is chosen
// once, in check_redist(), as the backend the handle was created with.
template <ddla::DdlaBackend Backend, typename T>
void check_node_local_arms(const ddla::DdlaHandle_t& handle)
{
    using rt = typename ddla::real_type<T>::type;
    static constexpr const char* tn = std::is_same_v<T, float> ? "f"
                                    : std::is_same_v<T, double> ? "d"
                                    : std::is_same_v<T, std::complex<float>> ? "cf" : "cd";
    auto tagged = [&](const char* name) { return std::string(name) + "<" + tn + ">"; };
    const int n = 13;
    const int k2 = 2;  // narrow second factor for her2k

    // Host reference data.
    std::vector<T> A_full(n * n), x(n), y(n), L_full(n * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            A_full[i + j * n] = herm_value<T>(i, j);
            L_full[i + j * n] = (i >= j) ? redist_value<T>(i, j) : T{};
        }
    }
    for (int i = 0; i < n; ++i) x[i] = redist_value<T>(i + 3, 1);
    for (int i = 0; i < n; ++i) y[i] = redist_value<T>(2, i + 4);
    std::vector<T> ref_ax(n), ref_trx(n), ref_hemm(n * k2);
    for (int i = 0; i < n; ++i) {
        T acc{};
        for (int j = 0; j < n; ++j) acc += A_full[i + j * n] * x[j];
        ref_ax[i] = acc;
        acc = T{};
        for (int j = 0; j <= i; ++j) acc += L_full[i + j * n] * x[j];
        ref_trx[i] = acc;
    }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < k2; ++j) {
            T acc{};
            for (int l = 0; l < n; ++l) acc += A_full[i + l * n] * redist_value<T>(l, j + 9);
            ref_hemm[i + j * n] = acc;
        }
    rt ref_nrm{};
    for (int i = 0; i < n; ++i) { const double a = std::abs(x[i]); ref_nrm += static_cast<rt>(a * a); }
    ref_nrm = std::sqrt(ref_nrm);
    T ref_dot{};
    for (int i = 0; i < n; ++i) ref_dot += conj_v(x[i]) * y[i];

    // Device buffers.
    DeviceBuffer<T> d_A(handle, n * n);
    DeviceBuffer<T> d_L(handle, n * n);
    DeviceBuffer<T> d_x(handle, n);
    DeviceBuffer<T> d_y(handle, n);
    DeviceBuffer<T> d_r(handle, n);
    DeviceBuffer<T> d_C(handle, n * n);
    DeviceBuffer<T> d_AB(handle, n * k2);
    DeviceBuffer<T> d_BB(handle, n * k2);
    DeviceBuffer<T> d_C2(handle, n * n);   // xHER2K's C is n x n (full lower triangle written)
    upload(handle, d_y.ptr, y);

    auto check_max = [&](const std::string& name, const std::vector<T>& got,
                         const std::vector<T>& ref, rt tol) {
        rt err{};
        size_t worst = 0;
        for (size_t idx = 0; idx < ref.size(); ++idx) {
            const rt e = static_cast<rt>(std::abs(got[idx] - ref[idx]));
            if (e > err) { err = e; worst = idx; }
        }
        if (err > tol) {
            std::ostringstream os;
            os << "  DIAG r" << ddlaGetRank(handle) << " " << name
               << ": worst idx=" << worst << " got=" << got[worst] << " ref=" << ref[worst] << " |";
            for (size_t idx = 0; idx < ref.size(); ++idx)
                os << " " << got[idx] << "/" << ref[idx];
            std::cout << os.str() << std::endl;
        }
        require_close(handle, name, static_cast<double>(err), static_cast<double>(tol));
    };
    const rt tol = static_cast<rt>(n) * (std::is_same<rt, float>::value ? 1e-4 : 1e-12);

    // gemv: y = A x
    upload(handle, d_A.ptr, A_full);
    upload(handle, d_x.ptr, x);
    ddla::gemv<Backend, T>(handle, 'N', n, n, T(1.0),
                           d_A.ptr, n, d_x.ptr, 1, T(0.0), d_r.ptr, 1);
    check_max(tagged("gemv").c_str(), download<T>(handle, d_r.ptr, n), ref_ax, tol);

    // hemv (lower) on the same A: y = A x, same reference.
    upload(handle, d_A.ptr, A_full);
    ddla::hemv<Backend, T>(handle, 'L', n, T(1.0),
                           d_A.ptr, n, d_x.ptr, 1, T(0.0), d_r.ptr, 1);
    check_max(tagged("hemv").c_str(), download<T>(handle, d_r.ptr, n), ref_ax, tol);

    // trmv (lower, non-unit): x := L x
    upload(handle, d_L.ptr, L_full);
    upload(handle, d_x.ptr, x);
    ddla::trmv<Backend, T>(handle, 'L', 'N', 'N',
                           n, d_L.ptr, n, d_x.ptr, 1);
    check_max(tagged("trmv").c_str(), download<T>(handle, d_x.ptr, n), ref_trx, tol);

    // nrm2(x), dotc(x, y): results are written through a pointer whose address
    // space follows the backend -- device memory for GPU handles, host memory
    // for CPU handles.  The GPU arm puts the vendor handle into DEVICE pointer
    // mode around the call; without that (the vendor default is HOST mode) the
    // result buffer used to be written with a host store, which on the
    // stream-ordered ddlaMallocAsync pool showed up as a host-side SEGV_ACCERR
    // inside libcublas.  That was a pointer-mode bug, not an allocation-pool
    // one.
    {
        upload(handle, d_x.ptr, x);   // trmv above overwrote x in place
        upload(handle, d_y.ptr, y);
        if constexpr (Backend == ddla::DdlaBackend::CPU) {
            rt nrm_v{};
            ddla::nrm2<Backend, T>(handle, n, d_x.ptr, 1, &nrm_v);
            require_close(handle, tagged("nrm2").c_str(), std::abs(nrm_v - ref_nrm),
                          static_cast<double>(tol));
            T dot_v{};
            ddla::dotc<Backend, T>(handle, n, d_x.ptr, 1, d_y.ptr, 1, &dot_v);
            check_max(tagged("dotc").c_str(), std::vector<T>{dot_v}, {ref_dot}, tol);
        } else {
            DeviceBuffer<rt> d_nrm(handle, 1);
            ddla::nrm2<Backend, T>(handle, n, d_x.ptr, 1, d_nrm.ptr);
            const auto nrm_v = download<rt>(handle, d_nrm.ptr, 1);
            require_close(handle, tagged("nrm2").c_str(), std::abs(nrm_v[0] - ref_nrm),
                          static_cast<double>(tol));
            DeviceBuffer<T> d_dot(handle, 1);
            ddla::dotc<Backend, T>(handle, n, d_x.ptr, 1, d_y.ptr, 1, d_dot.ptr);
            check_max(tagged("dotc").c_str(), download<T>(handle, d_dot.ptr, 1), {ref_dot}, tol);
        }
    }

    // her2 (lower): A := A + x y^H + y x^H
    {
        std::vector<T> A_ref(A_full);
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i)  // lower triangle, symmetric update
                A_ref[i + j * n] += x[i] * conj_v(y[j]) + y[i] * conj_v(x[j]);
        upload(handle, d_x.ptr, x);   // trmv above overwrote x in place
        upload(handle, d_y.ptr, y);
        upload(handle, d_A.ptr, A_full);
        ddla::her2<Backend, T>(handle, 'L', n, T(1.0),
                               d_x.ptr, 1, d_y.ptr, 1, d_A.ptr, n);
        std::vector<T> got = download<T>(handle, d_A.ptr, n * n);
        std::vector<T> ref_lower(n * n, T{});
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i) ref_lower[i + j * n] = A_ref[i + j * n];
        // Compare only the lower triangle (upper is untouched vendor output).
        rt err{};
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i)
                err = std::max(err, static_cast<rt>(std::abs(got[i + j * n] - ref_lower[i + j * n])));
        require_close(handle, tagged("her2").c_str(), static_cast<double>(err), static_cast<double>(tol));
    }

    // herk (lower, 'N'): C := A A^H
    {
        std::vector<T> C_ref(n * n, T{});
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i) {
                T acc{};
                for (int l = 0; l < n; ++l) acc += A_full[i + l * n] * conj_v(A_full[j + l * n]);
                C_ref[i + j * n] = acc;
            }
        upload(handle, d_A.ptr, A_full);
        ddla::herk<Backend, T>(handle, 'L', 'N', n, n,
                               rt(1.0), d_A.ptr, n, rt(0.0), d_C.ptr, n);
        std::vector<T> got = download<T>(handle, d_C.ptr, n * n);
        rt err{};
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i)
                err = std::max(err, static_cast<rt>(std::abs(got[i + j * n] - C_ref[i + j * n])));
        require_close(handle, tagged("herk").c_str(), static_cast<double>(err), static_cast<double>(tol));
    }

    // hemm (left, lower): C := A B with B n x k2
    {
        std::vector<T> B_host(n * k2);
        for (int j = 0; j < k2; ++j)
            for (int i = 0; i < n; ++i) B_host[i + j * n] = redist_value<T>(i, j + 9);
        upload(handle, d_AB.ptr, B_host);
        ddla::hemm<Backend, T>(handle, 'L', 'L', n, k2,
                               T(1.0), d_A.ptr, n, d_AB.ptr, n, T(0.0), d_C2.ptr, n);
        check_max(tagged("hemm").c_str(), download<T>(handle, d_C2.ptr, n * k2), ref_hemm, tol);
    }

    // her2k (lower, 'N'): C := A B^H + B A^H with A, B n x k2
    {
        std::vector<T> A2(n * k2), B2(n * k2), C2_ref(n * n, T{});
        for (int j = 0; j < k2; ++j)
            for (int i = 0; i < n; ++i) {
                A2[i + j * n] = redist_value<T>(i + 1, j + 5);
                B2[i + j * n] = redist_value<T>(i + 2, j + 6);
            }
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i) {
                T acc{};
                for (int l = 0; l < k2; ++l)
                    acc += A2[i + l * n] * conj_v(B2[j + l * n])
                         + B2[i + l * n] * conj_v(A2[j + l * n]);
                C2_ref[i + j * n] = acc;
            }
        upload(handle, d_AB.ptr, A2);
        upload(handle, d_BB.ptr, B2);
        ddla::her2k<Backend, T>(handle, 'L', 'N', n, k2,
                                T(1.0), d_AB.ptr, n, d_BB.ptr, n, rt(0.0), d_C2.ptr, n);
        std::vector<T> got = download<T>(handle, d_C2.ptr, n * n);
        rt err{};
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i)
                err = std::max(err, static_cast<rt>(std::abs(got[i + j * n] - C2_ref[i + j * n])));
        require_close(handle, tagged("her2k").c_str(), static_cast<double>(err), static_cast<double>(tol));
    }

    // her2k with trans='C': for real types this must behave exactly like a
    // plain transpose (vendor BLAS rejects 'C' on real operands), and for the
    // complex types it is the genuine conjugate-transpose form.
    //
    // With trans='C' the operands are k x n, so LDA must be at least k (not n)
    // and the storage is A2[row + col*k2].
    {
        std::vector<T> A2(static_cast<std::size_t>(k2) * n), B2(static_cast<std::size_t>(k2) * n);
        std::vector<T> C2_ref(n * n, T{});
        for (int i = 0; i < n; ++i)
            for (int l = 0; l < k2; ++l) {
                A2[l + i * k2] = redist_value<T>(l + 1, i + 5);
                B2[l + i * k2] = redist_value<T>(l + 2, i + 6);
            }
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i) {
                T acc{};
                for (int l = 0; l < k2; ++l)
                    acc += conj_v(A2[l + i * k2]) * B2[l + j * k2]
                         + conj_v(B2[l + i * k2]) * A2[l + j * k2];
                C2_ref[i + j * n] = acc;
            }
        upload(handle, d_AB.ptr, A2);
        upload(handle, d_BB.ptr, B2);
        upload(handle, d_C2.ptr, std::vector<T>(n * n, T{}));
        ddla::her2k<Backend, T>(handle, 'L', 'C', n, k2,
                                T(1.0), d_AB.ptr, k2, d_BB.ptr, k2, rt(0.0), d_C2.ptr, n);
        std::vector<T> got = download<T>(handle, d_C2.ptr, n * n);
        rt err{};
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i)
                err = std::max(err, static_cast<rt>(std::abs(got[i + j * n] - C2_ref[i + j * n])));
        require_close(handle, tagged("her2k('C')").c_str(), static_cast<double>(err),
                      static_cast<double>(tol));
    }

    // ---- backend-templated entry layer -------------------------------------
    // The same entries the eigensolver stages call.  The backend must come
    // from the HANDLE: template and handle backend are required to agree
    // (every entry validates that), and the two arms consume different pointer
    // spaces -- instantiating the CPU arm against a GPU handle would read
    // device memory as host memory.  @p Backend is that arm.
    {
        // Stage 1's exact call shape: herk('C') on the upper triangle.
        std::vector<T> C_ref(n * n, T{});
        for (int j = 0; j < n; ++j)
            for (int i = 0; i <= j; ++i) {
                T acc{};
                for (int l = 0; l < n; ++l) acc += A_full[i + l * n] * conj_v(A_full[j + l * n]);
                C_ref[i + j * n] = acc;
            }
        upload(handle, d_A.ptr, A_full);
        upload(handle, d_C.ptr, std::vector<T>(n * n, T{}));
        ddla::herk<Backend, T>(handle, 'U', 'C', n, n,
                               rt(1.0), d_A.ptr, n, rt(0.0), d_C.ptr, n);
        std::vector<T> got = download<T>(handle, d_C.ptr, n * n);
        rt err{};
        for (int j = 0; j < n; ++j)
            for (int i = 0; i <= j; ++i)
                err = std::max(err, static_cast<rt>(std::abs(got[i + j * n] - C_ref[i + j * n])));
        require_close(handle, tagged("herk<Backend>('C','U')").c_str(), static_cast<double>(err),
                      static_cast<double>(tol));
    }
    {
        // nrm2/dotc through the templated layer.  Their result pointer is a
        // DEVICE pointer on the GPU arm and a HOST pointer on the CPU arm, so
        // the branch has to follow the same backend choice.
        upload(handle, d_x.ptr, x);
        upload(handle, d_y.ptr, y);
        if constexpr (Backend == ddla::DdlaBackend::CPU) {
            rt got_nrm{};
            ddla::nrm2<Backend, T>(handle, n, d_x.ptr, 1, &got_nrm);
            require_close(handle, tagged("nrm2<CPU>").c_str(),
                          static_cast<double>(std::abs(got_nrm - ref_nrm)), static_cast<double>(tol));
            T got_dot{};
            ddla::dotc<Backend, T>(handle, n, d_x.ptr, 1, d_y.ptr, 1, &got_dot);
            require_close(handle, tagged("dotc<CPU>").c_str(),
                          static_cast<double>(std::abs(got_dot - ref_dot)), static_cast<double>(tol));
        } else {
            DeviceBuffer<rt> d_nrm2(handle, 1);
            ddla::nrm2<Backend, T>(handle, n, d_x.ptr, 1, d_nrm2.ptr);
            const auto got_nrm = download<rt>(handle, d_nrm2.ptr, 1);
            require_close(handle, tagged("nrm2<GPU>").c_str(),
                          static_cast<double>(std::abs(got_nrm[0] - ref_nrm)), static_cast<double>(tol));
            DeviceBuffer<T> d_dot2(handle, 1);
            ddla::dotc<Backend, T>(handle, n, d_x.ptr, 1, d_y.ptr, 1, d_dot2.ptr);
            require_close(handle, tagged("dotc<GPU>").c_str(),
                          static_cast<double>(std::abs(download<T>(handle, d_dot2.ptr, 1)[0] - ref_dot)),
                          static_cast<double>(tol));
        }
    }
}

} // namespace

template <typename T>
void check_pvector_redist(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    using rt = typename ddla::real_type<T>::type;
    const int n = round_up_for_grid(base.m, base.nb, 2);
    const int nb = base.nb;
    const int nvc = 5;  // vector count (panel width in real use)

    int nprows = 0, npcols = 0, myprow = -1, mypcol = -1;
    ddlaGetGridDims(handle, nprows, npcols);
    ddlaGetGridCoords(handle, myprow, mypcol);

    // ---- row-distributed source: identical copy in every process column ---
    const int ld_src = ddla::num_loc(n, nb, myprow, 0, nprows);
    const size_t src_count = static_cast<size_t>(ld_src) * nvc;
    std::vector<T> h_src(src_count, T{});
    for (int lc = 0; lc < nvc; ++lc) {
        for (int l = 0; l < ld_src; ++l) {
            const int gi = ddla::indxl2g(l, nb, myprow, 0, nprows);
            if (gi >= n) continue;
            h_src[l + static_cast<size_t>(lc) * ld_src] = redist_value<T>(gi, lc);
        }
    }
    DeviceBuffer<T> d_src(handle, src_count);
    upload(handle, d_src.ptr, h_src);

    // ---- ptranspose_vectors: overwrite mode -------------------------------
    const int ld_dst = ddla::num_loc(n, nb, mypcol, 0, npcols);
    const size_t dst_count = static_cast<size_t>(ld_dst) * nvc;
    DeviceBuffer<T> d_dst(handle, dst_count);
    ddla::ptranspose_vectors(handle, d_src.ptr, ld_src, d_dst.ptr, ld_dst, n, nvc, nb);
    check_ddla_sync(handle);
    {
        auto out = download(handle, d_dst.ptr, dst_count);
        double err = 0.0;
        for (int lc = 0; lc < nvc; ++lc) {
            for (int l = 0; l < ld_dst; ++l) {
                const int gj = ddla::indxl2g(l, nb, mypcol, 0, npcols);
                if (gj >= n) continue;
                err = std::max(err, static_cast<double>(std::abs(
                    out[l + static_cast<size_t>(lc) * ld_dst] - redist_value<T>(gj, lc))));
            }
        }
        require_close(handle, "ptranspose_vectors overwrite", err, 1e-13);
    }

    // ---- ptranspose_vectors: accumulate mode (dst += transposed) ----------
    std::vector<T> h_two(dst_count);
    {
        // Pre-fill dst with a constant so the expected result is c + f(gj, lc).
        const T c = static_cast<rt>(2.0) * redist_value<T>(1, 2);
        for (size_t k = 0; k < dst_count; ++k) h_two[k] = c;
        upload(handle, d_dst.ptr, h_two);
        ddla::ptranspose_vectors(handle, d_src.ptr, ld_src, d_dst.ptr, ld_dst, n, nvc, nb,
                                 /*accumulate_dst=*/true);
        check_ddla_sync(handle);
        auto out = download(handle, d_dst.ptr, dst_count);
        double err = 0.0;
        for (int lc = 0; lc < nvc; ++lc) {
            for (int l = 0; l < ld_dst; ++l) {
                const int gj = ddla::indxl2g(l, nb, mypcol, 0, npcols);
                if (gj >= n) continue;
                err = std::max(err, static_cast<double>(std::abs(
                    out[l + static_cast<size_t>(lc) * ld_dst]
                    - (c + redist_value<T>(gj, lc)))));
            }
        }
        require_close(handle, "ptranspose_vectors accumulate", err, 1e-13);
    }

    // ---- ptranspose_vectors: caller-supplied workspace (2026-10-02) --------
    // phe2hb sizes one buffer for its largest panel and reuses it for every
    // transpose.  Same check as overwrite mode, through a workspace pre-filled
    // with junk, then reused for a smaller call (fewer rows and vectors), then
    // an undersized workspace, which every rank must reject before any
    // communication.
    {
        const std::size_t ws_elems = ddla::ptranspose_vectors_work_elems(handle, n, nvc, nb);
        DeviceBuffer<T> d_ws(handle, ws_elems);
        upload(handle, d_ws.ptr, std::vector<T>(ws_elems, static_cast<rt>(7.0) * redist_value<T>(3, 4)));
        auto check_prefix = [&](int n_chk, int nvc_chk, const char* label) {
            auto out = download(handle, d_dst.ptr, dst_count);
            double err = 0.0;
            for (int lc = 0; lc < nvc_chk; ++lc) {
                for (int l = 0; l < ld_dst; ++l) {
                    const int gj = ddla::indxl2g(l, nb, mypcol, 0, npcols);
                    if (gj >= n_chk) continue;
                    err = std::max(err, static_cast<double>(std::abs(
                        out[l + static_cast<size_t>(lc) * ld_dst] - redist_value<T>(gj, lc))));
                }
            }
            require_close(handle, label, err, 1e-13);
        };
        upload(handle, d_dst.ptr, std::vector<T>(dst_count, T{}));
        ddla::ptranspose_vectors(handle, d_src.ptr, ld_src, d_dst.ptr, ld_dst, n, nvc, nb,
                                 /*accumulate_dst=*/false, d_ws.ptr, ws_elems);
        check_ddla_sync(handle);
        check_prefix(n, nvc, "ptranspose_vectors workspace");

        const int n2 = std::max(n - nb - 1, 1);
        const int nvc2 = std::max(nvc - 2, 1);
        upload(handle, d_dst.ptr, std::vector<T>(dst_count, T{}));
        ddla::ptranspose_vectors(handle, d_src.ptr, ld_src, d_dst.ptr, ld_dst, n2, nvc2, nb,
                                 /*accumulate_dst=*/false, d_ws.ptr, ws_elems);
        check_ddla_sync(handle);
        check_prefix(n2, nvc2, "ptranspose_vectors workspace reuse (smaller call)");

        bool threw = false;
        try {
            ddla::ptranspose_vectors(handle, d_src.ptr, ld_src, d_dst.ptr, ld_dst, n, nvc, nb,
                                     /*accumulate_dst=*/false, d_ws.ptr, ws_elems - 1);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        require_close(handle, "ptranspose_vectors rejects an undersized workspace",
                      threw ? 0.0 : 1.0, 0.5);
    }

    // ---- preduce_add_vectors: column partials summed over process rows ----
    // Partial on process column c holds value * (c + 1); the summed result is
    // value * npcols*(npcols+1)/2, transposed into the column layout.
    std::vector<T> h_part(src_count, T{});
    const rt col_factor = static_cast<rt>(mypcol + 1);
    for (int lc = 0; lc < nvc; ++lc) {
        for (int l = 0; l < ld_src; ++l) {
            const int gi = ddla::indxl2g(l, nb, myprow, 0, nprows);
            if (gi >= n) continue;
            h_part[l + static_cast<size_t>(lc) * ld_src] = col_factor * redist_value<T>(gi, lc);
        }
    }
    DeviceBuffer<T> d_part(handle, src_count);
    upload(handle, d_part.ptr, h_part);
    DeviceBuffer<T> d_acc(handle, dst_count);  // zero-initialized? ddlaMalloc is not
    {
        std::vector<T> zeros(dst_count, T{});
        upload(handle, d_acc.ptr, zeros);
    }
    ddla::preduce_add_vectors(handle, d_part.ptr, ld_src, d_acc.ptr, ld_dst, n, nvc, nb);
    check_ddla_sync(handle);
    {
        auto out = download(handle, d_acc.ptr, dst_count);
        const rt total = static_cast<rt>(npcols * (npcols + 1) / 2);
        double err = 0.0;
        for (int lc = 0; lc < nvc; ++lc) {
            for (int l = 0; l < ld_dst; ++l) {
                const int gj = ddla::indxl2g(l, nb, mypcol, 0, npcols);
                if (gj >= n) continue;
                err = std::max(err, static_cast<double>(std::abs(
                    out[l + static_cast<size_t>(lc) * ld_dst]
                    - total * redist_value<T>(gj, lc))));
            }
        }
        require_close(handle, "preduce_add_vectors", err, 1e-13);
    }
}

// run_grid_test creates the handle with ddlaInit(handle), i.e. on
// default_backend_v (GPU when the build has one, CPU otherwise), so that is the
// arm the node-local entries must be instantiated with.  The distribution
// helpers keep their defaulted template argument for the same reason.
template <ddla::DdlaBackend Backend>
static void check_redist_impl(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    check_node_local_arms<Backend, double>(handle);
    check_node_local_arms<Backend, Complex>(handle);
    check_pvector_redist<double>(handle, base);
    check_pvector_redist<Complex>(handle, base);
}

static void check_redist(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    check_redist_impl<ddla::default_backend_v>(handle, base);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pvector_redist", check_redist);
}
