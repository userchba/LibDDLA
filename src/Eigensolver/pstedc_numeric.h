#ifndef DDLA_PSTEDC_NUMERIC_H
#define DDLA_PSTEDC_NUMERIC_H

#include <ddla/ddla_handle_t.h>
#include <algorithm>
#include <cmath>

// Typed LAPACK dispatch for distributed D&C.
// LAED8 runs with ICOMPQ=0: matrix storage stays distributed.
namespace ddla { namespace spectral_impl {
// scaled 2-norm (the ?NRM2 guard against overflowing the squares)
template <typename rt>
rt scaled_nrm2(const rt* x, int n)
{
    rt amax = 0;
    for (int i = 0; i < n; ++i) amax = std::max(amax, static_cast<rt>(std::fabs(x[i])));
    if (amax == rt(0)) return rt(0);
    rt s = 0;
    for (int i = 0; i < n; ++i) {
        const rt v = x[i] / amax;
        s += v * v;
    }
    return amax * std::sqrt(s);
}

#define DDLA_PSTEDC_NUMERIC(R) \
void stedc(char,int,R*,R*,R*,int,R*,int,int*,int,int&); \
void laed8_decide(int&,int,R*,int*,R&,int,R*,R*,R*,int*,int&,int*,R*,int*,int*,int&); \
void laed4(int,int,const R*,const R*,R*,const R&,R&,int&)
DDLA_PSTEDC_NUMERIC(float);
DDLA_PSTEDC_NUMERIC(double);
#undef DDLA_PSTEDC_NUMERIC

} }
#endif
