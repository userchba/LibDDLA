#ifndef DDLA_LDLT_PAIR_SOLVE_H
#define DDLA_LDLT_PAIR_SOLVE_H

#include <ddla/ddla.h>
#include <cmath>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define DDLA_PAIR_HD __host__ __device__
#else
#define DDLA_PAIR_HD
#endif

namespace ddla::ldlt_impl {

template<class R> struct PairComponents { R real, imag; };

template<class R> struct PairRow {
    int row;
    PairComponents<R> diagonal, partner_diagonal, coupling;
};

// Keep exponents separate in the extreme-range fallback. Products that
// would overflow or underflow in R can still cancel or form a finite quotient.
template<class R> struct PairScaled {
    R fraction;
    int exponent;

    DDLA_PAIR_HD static PairScaled make(R value, int exponent = 0)
    {
        int adjustment = 0;
        const R fraction = std::is_same_v<R, float> ? R(::frexpf(value, &adjustment))
                                                    : R(::frexp(value, &adjustment));
        return {fraction, exponent + adjustment};
    }

    DDLA_PAIR_HD static R scale(R value, int exponent)
    {
        if constexpr (std::is_same_v<R, float>) return ::scalbnf(value, exponent);
        else return ::scalbn(value, exponent);
    }

    DDLA_PAIR_HD friend PairScaled operator*(PairScaled a, PairScaled b)
    {
        return make(a.fraction * b.fraction, a.exponent + b.exponent);
    }

    DDLA_PAIR_HD friend PairScaled operator-(PairScaled a, PairScaled b)
    {
        if (b.fraction == R(0)) return a;
        if (a.fraction == R(0)) return {-b.fraction, b.exponent};
        if (a.exponent >= b.exponent)
            return make(a.fraction - scale(b.fraction, b.exponent - a.exponent), a.exponent);
        return make(scale(a.fraction, a.exponent - b.exponent) - b.fraction, b.exponent);
    }

    DDLA_PAIR_HD friend PairScaled operator+(PairScaled a, PairScaled b)
    {
        return a - PairScaled{-b.fraction, b.exponent};
    }

    DDLA_PAIR_HD R divide(PairScaled denominator) const
    {
        return scale(fraction / denominator.fraction, exponent - denominator.exponent);
    }
};

template<class S> struct PairComplex {
    S real, imag;

    DDLA_PAIR_HD friend PairComplex operator*(PairComplex a, PairComplex b)
    {
        return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
    }

    DDLA_PAIR_HD friend PairComplex operator-(PairComplex a, PairComplex b)
    {
        return {a.real - b.real, a.imag - b.imag};
    }
};

template<class R> DDLA_PAIR_HD auto extend_pair(PairComponents<R> v)
{
    using S = PairScaled<R>;
    return PairComplex<S>{S::make(v.real), S::make(v.imag)};
}

template<class R, class T> DDLA_PAIR_HD PairComponents<R> pair_components(T v)
{
    if constexpr (std::is_arithmetic_v<T>) return {v, R(0)};
    else return {v.real(), v.imag()};
}

template<class T, class R> DDLA_PAIR_HD T pair_scalar(PairComponents<R> v)
{
    if constexpr (std::is_arithmetic_v<T>) return v.real;
    else return T(v.real, v.imag);
}

template<class R> DDLA_PAIR_HD bool ordinary_pair_component(R x)
{
    constexpr R bound = std::is_same_v<R, float> ? R(0x1p12) : R(0x1p120);
    const R a = x < R(0) ? -x : x;
    return a == R(0) || (a >= R(1) / bound && a <= bound);
}

template<class R> DDLA_PAIR_HD bool ordinary_pair_value(PairComponents<R> x)
{
    return ordinary_pair_component(x.real) && ordinary_pair_component(x.imag);
}

template<bool Hermitian, class T, class R>
DDLA_PAIR_HD T solve_pair_row(const PairRow<R>& p, T y, T partner_y)
{
    const PairComponents<R> other_coupling{p.coupling.real,
        Hermitian ? -p.coupling.imag : p.coupling.imag};
    const auto rhs = pair_components<R>(y);
    const auto partner_rhs = pair_components<R>(partner_y);
    if (ordinary_pair_value(p.diagonal) && ordinary_pair_value(p.partner_diagonal) &&
        ordinary_pair_value(p.coupling) && ordinary_pair_value(rhs) && ordinary_pair_value(partner_rhs)) {
        const T diagonal = pair_scalar<T>(p.diagonal);
        const T partner_diagonal = pair_scalar<T>(p.partner_diagonal);
        const T coupling = pair_scalar<T>(p.coupling);
        const T determinant = diagonal * partner_diagonal - coupling * pair_scalar<T>(other_coupling);
        return (partner_diagonal * y - coupling * partner_y) / determinant;
    }

    if constexpr (std::is_arithmetic_v<T>) {
        using S = PairScaled<R>;
        const auto diagonal = S::make(p.diagonal.real);
        const auto partner_diagonal = S::make(p.partner_diagonal.real);
        const auto coupling = S::make(p.coupling.real);
        return (partner_diagonal * S::make(y) - coupling * S::make(partner_y))
            .divide(diagonal * partner_diagonal - coupling * coupling);
    } else {
        const auto diagonal = extend_pair(p.diagonal);
        const auto partner_diagonal = extend_pair(p.partner_diagonal);
        const auto coupling = extend_pair(p.coupling);
        const auto determinant = diagonal * partner_diagonal - coupling * extend_pair(other_coupling);
        const auto numerator = partner_diagonal * extend_pair(rhs) - coupling * extend_pair(partner_rhs);
        if constexpr (Hermitian) {
            return T(numerator.real.divide(determinant.real), numerator.imag.divide(determinant.real));
        } else {
            const auto a = determinant.real, b = determinant.imag;
            const auto square = a * a + b * b;
            return T((numerator.real * a + numerator.imag * b).divide(square),
                     (numerator.imag * a - numerator.real * b).divide(square));
        }
    }
}

#if DDLA_HAS_GPU
template<bool Hermitian, class T, class R = typename real_type<T>::type>
void solve_pair_rows_gpu(const DdlaHandle_t& h, int rows, int columns, const PairRow<R>* plan,
                         T* x, int ldx, const T* partner, int ldp);
#endif

} // namespace ddla::ldlt_impl

#undef DDLA_PAIR_HD
#endif
