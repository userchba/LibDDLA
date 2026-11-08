// Test-only replacement of the root's pivot-record operation.
#include "LDLT/hetrf_native.h"
#include <complex>
#include <stdexcept>
namespace ddla::ldlt_impl
{
template <>
void summarize_column<std::complex<double>>(const DdlaHandle_t&, std::complex<double>*, int, int,
                                            int, PivotSummary*)
{
    throw std::runtime_error("injected root pivot failure");
}
} // namespace ddla::ldlt_impl
