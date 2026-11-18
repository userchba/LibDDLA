#ifndef DDLA_DIAG_LOGDET_IMPL_H
#define DDLA_DIAG_LOGDET_IMPL_H

#include <ddla/ddla.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)
#define DDLA_LOGDET_HD __host__ __device__
#else
#define DDLA_LOGDET_HD
#endif

namespace ddla { namespace detail {

// Iterate the more distributed local axis and filter by ownership on the
// other axis. This also bounds work on 1xP/Px1 grids. Local uses a 1x1 grid.
struct DiagLogLayout {
    int n, count, lda, mb, nb, pr, pc, row, col, rsrc, csrc;
    bool along_rows;
};

struct DiagLogState {
    double magnitude = 0;
    double phase = 0;
    int flags = 0; // 1: nonfinite, 2: zero, 4: negative-real parity
};

template<class T>
inline void check_diag_span(int rows, int columns, int lda)
{
    if (rows == 0 || columns == 0) return;
    const auto last = std::uint64_t(columns - 1) * std::uint64_t(lda) + rows - 1;
    if (last > std::uint64_t(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(T))
        throw std::overflow_error("diag_logdet: matrix address span is not representable");
}

struct DiagLogJoin {
    DDLA_LOGDET_HD DiagLogState operator()(DiagLogState a, DiagLogState b) const
    {
        return {a.magnitude + b.magnitude, a.phase + b.phase,
                ((a.flags | b.flags) & 3) | ((a.flags ^ b.flags) & 4)};
    }
};

template<class T> struct DiagLogTerm {
    const T* a;
    DiagLogLayout layout;

    DDLA_LOGDET_HD DiagLogState operator()(int item) const
    {
        const int block = layout.along_rows ? layout.mb : layout.nb;
        const int peers = layout.along_rows ? layout.pr : layout.pc;
        const int coordinate = layout.along_rows ? layout.row : layout.col;
        const int source = layout.along_rows ? layout.rsrc : layout.csrc;
        const int offset = (coordinate - source + peers) % peers;
        const long long g = (static_cast<long long>(item / block) * peers + offset)
                            * block + item % block;
        if (g >= layout.n || (layout.rsrc + g / layout.mb) % layout.pr != layout.row ||
            (layout.csrc + g / layout.nb) % layout.pc != layout.col)
            return {};
        const std::size_t row = static_cast<std::size_t>(
            g / (static_cast<long long>(layout.mb) * layout.pr) * layout.mb + g % layout.mb);
        const std::size_t column = static_cast<std::size_t>(
            g / (static_cast<long long>(layout.nb) * layout.pc) * layout.nb + g % layout.nb);
        const std::size_t index = row + column * layout.lda;
        using R = typename real_type<T>::type;
        const R* data = reinterpret_cast<const R*>(a);
        double x, y = 0;
        if constexpr (std::is_same<T, std::complex<R>>::value) {
            // std::complex arrays have interleaved real components. Avoid
            // calling host-only std::complex members from device code.
            x = data[2 * index];
            y = data[2 * index + 1];
        } else {
            x = data[index];
        }
        const double limit = std::numeric_limits<double>::max();
        if (!(x <= limit && x >= -limit && y <= limit && y >= -limit))
            return {0, 0, 1};
        const double ax = ::fabs(x), ay = ::fabs(y);
        const double hi = ax > ay ? ax : ay, lo = ax > ay ? ay : ax;
        if (hi == 0) return {0, 0, 2};
        const double ratio = lo / hi;
        // hypot can overflow even though log(hypot) is representable.
        const double magnitude = ::log(hi) + 0.5 * ::log1p(ratio * ratio);
        if (y == 0) return {magnitude, 0, x < 0 ? 4 : 0};
        return {magnitude, ::atan2(y, x), 0};
    }
};

template<DdlaBackend B, class T>
DiagLogState diagonal_logsum(const DdlaHandle_t&, const T*, const DiagLogLayout&);

#if DDLA_HAS_GPU
template<class T>
DiagLogState diagonal_logsum_gpu(const DdlaHandle_t&, const T*, const DiagLogLayout&);
#endif

template<class R>
inline std::complex<R> finish_diagonal_log(double magnitude, double phase,
                                         bool zero, int negative_parity)
{
    if (zero) return {-std::numeric_limits<R>::infinity(), R(0)};
    const double pi = std::acos(-1.0);
    phase = std::remainder(phase + (negative_parity & 1) * pi, 2 * pi);
    if (phase <= -pi) phase = pi;
    return {static_cast<R>(magnitude), static_cast<R>(phase)};
}

} } // namespace ddla::detail
#undef DDLA_LOGDET_HD
#endif
