#include "api_grid_test_common.h"

#include <type_traits>

using namespace api_grid_test;

namespace {

template <typename T>
struct is_complex : std::false_type {};
template <typename U>
struct is_complex<std::complex<U>> : std::true_type {};

// C++11 has no `if constexpr`: scalar construction dispatches on an
// is_complex<> tag so the two-argument constructor is only instantiated
// for complex T.
template <typename T>
T scalar_impl(double real, double imag, std::true_type)
{
    using Real = typename T::value_type;
    return T(static_cast<Real>(real), static_cast<Real>(imag));
}

template <typename T>
T scalar_impl(double real, double imag, std::false_type)
{
    (void)imag;
    return static_cast<T>(real);
}

template <typename T>
T scalar(double real, double imag = 0.0)
{
    return scalar_impl<T>(real, imag, is_complex<T>());
}

template <typename T>
const char* scalar_name();

template <>
const char* scalar_name<float>() { return "float"; }

template <>
const char* scalar_name<double>() { return "double"; }

template <>
const char* scalar_name<std::complex<float>>() { return "complex<float>"; }

template <>
const char* scalar_name<std::complex<double>>() { return "complex<double>"; }

template <typename T, typename Matrix>
void run_case(const ddla::DdlaHandle_t& handle, const ddla::DdlaDesc& desc,
              const std::string& case_name, int panel_width, Matrix matrix,
              int expected_info, int expected_pivot = -1)
{
    auto h_A = make_local<T>(desc, matrix);
    DeviceBuffer<T> d_A(handle, h_A.size());
    upload(handle, d_A.ptr, h_A);
    check_ddla_sync(handle);

    std::vector<int> ipiv(desc.m_loc(), -1);
    int info = -1;
    ddla::pgetf2(desc.m(), desc.n(), panel_width, d_A.ptr, 0, desc, ipiv.data(), info);

    const std::string label = std::string("pgetf2(") + scalar_name<T>()
                            + "/" + case_name + ")";
    require_close(handle, label + " info", std::abs(info - expected_info), 0.0);
    if(expected_pivot >= 0){
        int myprow = 0, mypcol_unused = 0;
        ddla_get_grid_coords(handle, myprow, mypcol_unused);
        const double pivot_error = myprow == 0
                                 ? std::abs(ipiv[0] - expected_pivot) : 0.0;
        require_close(handle, label + " pivot", pivot_error, 0.0);
    }
}

template <typename T>
void check_scalar_type(const ddla::DdlaHandle_t& handle,
                       const ddla::DdlaDesc& desc, int n, int nb)
{
    int nprows = 0, npcols_unused = 0;
    ddla_get_grid_dims(handle, nprows, npcols_unused);

    run_case<T>(handle, desc, "normal", std::min(nb, n), [=](int i, int j) -> T {
        if(i == j){
            return scalar<T>(4.0 + 0.1 * i);
        }
        return scalar<T>(0.015 * ((i + 2 * j) % 5 - 2),
                         0.01 * ((2 * i + j) % 7 - 3));
    }, 0);

    const int remote_row = nprows > 1 ? nb : 1;
    run_case<T>(handle, desc, "cross-row", 1, [=](int i, int j) -> T {
        if(j == 0){
            return scalar<T>(i == remote_row ? 9.0 : 1.0 / (i + 1));
        }
        return i == j ? scalar<T>(2.0) : scalar<T>(0.0);
    }, 0, remote_row + 1);

    const int competing_row = nprows > 1 ? nb : 3;
    run_case<T>(handle, desc, "tie", 1, [=](int i, int j) -> T {
        if(j == 0){
            if(i == 0){
                return is_complex<T>::value ? scalar<T>(4.0, 4.0) : scalar<T>(8.0);
            }
            if(i == competing_row){
                return scalar<T>(8.0);
            }
            return scalar<T>(0.25);
        }
        return i == j ? scalar<T>(2.0) : scalar<T>(0.0);
    }, 0, 1);

    if (is_complex<T>::value){
        run_case<T>(handle, desc, "complex-metric", 1, [=](int i, int j) -> T {
            if(j == 0){
                if(i == 0) return scalar<T>(4.0, 4.0);
                if(i == competing_row) return scalar<T>(7.0);
                return scalar<T>(0.25);
            }
            return i == j ? scalar<T>(2.0) : scalar<T>(0.0);
        }, 0, 1);
    }

    run_case<T>(handle, desc, "tiny-nonzero", 1, [](int i, int j) -> T {
        if(i != j) return scalar<T>(0.0);
        return scalar<T>(i == 0 ? 1.0e-12 : 1.0);
    }, 0);

    run_case<T>(handle, desc, "exact-singular", 1, [](int i, int j) -> T {
        if(j == 0) return scalar<T>(0.0);
        return i == j ? scalar<T>(1.0) : scalar<T>(0.0);
    }, 1);
}

} // namespace

void check_pgetf2(const ddla::DdlaHandle_t& handle, const Shape& base)
{
    const int nb = base.nb;
    const int n = square_size(handle, base);
    ddla::DdlaDesc descA(handle);
    descA.init(n, n, nb, nb, 0, 0);

    check_scalar_type<float>(handle, descA, n, nb);
    check_scalar_type<double>(handle, descA, n, nb);
    check_scalar_type<std::complex<float>>(handle, descA, n, nb);
    check_scalar_type<std::complex<double>>(handle, descA, n, nb);
}

int main(int argc, char** argv)
{
    return run_grid_test(argc, argv, "test_api_grid_pgetf2", check_pgetf2);
}
