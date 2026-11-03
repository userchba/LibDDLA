#include <ddla/ddla_vector.h>
#include "workspace_test_common.h"

#include <complex>
#include <cstdint>
#include <limits>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using namespace ddla;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void check_values(const ddla::vector<T>& storage, const std::vector<T>& expected)
{
    std::vector<T> actual(expected.size());
    DDLA_CHECK(ddlaMemcpy(storage.handle(), actual.data(), storage.data(), actual.size() * sizeof(T)));
    DDLA_CHECK(ddlaSynchronize(storage.handle()));
    require(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T)) == 0,
            "vector did not preserve element bytes");
}

template <typename T>
void check_type(DdlaHandle_t first, DdlaHandle_t second)
{
    static_assert(!std::is_copy_constructible<ddla::vector<T>>::value, "vector must have unique ownership");
    static_assert(!std::is_copy_assignable<ddla::vector<T>>::value, "vector must not share ownership");
    static_assert(std::is_nothrow_move_constructible<ddla::vector<T>>::value, "move must not throw");
    static_assert(std::is_nothrow_move_assignable<ddla::vector<T>>::value, "move must not throw");
    static_assert(std::is_same<decltype(std::declval<const ddla::vector<T>&>().data()), const T*>::value,
                  "const vector must not expose mutable elements");

    ddla::vector<T> a(first);
    require(a.empty() && a.size() == 0 && a.capacity() == 0 && a.data() == nullptr, "empty vector");
    a.resize(0);
    a.reserve(0);
    require(a.data() == nullptr, "zero allocation must stay empty");
    std::vector<T> expected{T(-0.0), T(1.25), T(-3.5), T(7)};
    a.resize(expected.size());
    DDLA_CHECK(ddlaMemcpy(first, a.data(), expected.data(), expected.size() * sizeof(T)));
    a.reserve(19);
    require(a.size() == expected.size() && a.capacity() >= 19, "reserve changed size");
    check_values(a, expected);
    T* saved = a.data();
    a.resize(2);
    a.resize(12);
    require(a.data() == saved && a.size() == 12, "resize failed to reuse allocation");
    expected.resize(2);
    check_values(a, expected);
    a.resize(2);
    a.resize(31);
    check_values(a, expected);
    a.resize(2);
    saved = a.data();
    const auto capacity = a.capacity();
    bool rejected = false;
    try { a.resize(a.max_size() + 1); }
    catch (const std::length_error&) { rejected = true; }
    require(rejected && a.data() == saved && a.size() == 2 && a.capacity() == capacity,
            "overflow changed existing allocation");
    check_values(a, expected);

    ddla::vector<T> b(std::move(a));
    require(a.empty() && a.data() == nullptr && b.data() == saved, "move construction");
    check_values(b, expected);
    a.resize(3); // A moved-from vector retains its handle and can be reused.
    ddla::vector<T> c(second, 5);
    c = std::move(b); // Must free the old allocation with second, then adopt first.
    require(c.handle() == first && b.empty() && b.data() == nullptr, "move assignment handle");
    check_values(c, expected);
    c = std::move(c);
    check_values(c, expected);
    c.swap(a);
    check_values(a, expected);
    saved = a.data();
    a.clear();
    require(a.empty() && a.data() == saved, "clear released reusable capacity");
    a.resize(2);
    require(a.data() == saved, "reuse after clear");
    a.reset();
    require(a.empty() && a.capacity() == 0 && a.data() == nullptr, "reset did not release storage");
    a.resize(3);
}

template<class Function> Function diagnostic(const char* name)
{
    void* symbol = dlsym(RTLD_DEFAULT, name);
    Function function = nullptr;
    static_assert(sizeof(symbol) == sizeof(function), "dlsym function pointer ABI");
    std::memcpy(&function, &symbol, sizeof(function));
    return function;
}

template<class T> void check_growth_failure(DdlaHandle_t h)
{
    if (!workspace_test::available()) return;
    using Arm = void(*)(const void*, uint64_t);
    using Faults = uint64_t(*)();
    const auto arm = diagnostic<Arm>("workspace_trace_fail_memcpy_after_upload");
    const auto faults = diagnostic<Faults>("workspace_trace_memcpy_faults");
    require(arm != nullptr && faults != nullptr, "vector growth fault diagnostic missing");
    ddla::vector<T> values(h, 3);
    const std::vector<T> initial{T(2), T(4), T(8)};
    DDLA_CHECK(ddlaMemcpy(h, values.data(), initial.data(), initial.size() * sizeof(T)));
    DDLA_CHECK(ddlaSynchronize(h));
    auto* pointer = values.data();
    const auto size = values.size(), capacity = values.capacity();
    arm(pointer, size * sizeof(T));
    bool rejected = false;
    workspace_test::Scope scope;
    try { values.reserve(capacity + 7); }
    catch (const std::runtime_error&) { rejected = true; }
    const auto count = scope.finish();
    const auto hits = faults();
    arm(nullptr, 0);
    require(rejected && hits == 1 && count.ddla_alloc == 1 && count.ddla_free == 1,
            "failed vector growth did not retire only the new allocation");
    require(values.data() == pointer && values.size() == size && values.capacity() == capacity,
            "failed vector growth changed original metadata");
    check_values(values, initial);
}

void check_handles(DdlaHandle_t first, DdlaHandle_t second)
{
    check_type<float>(first, second);
    check_type<double>(first, second);
    check_type<std::complex<float>>(first, second);
    check_type<std::complex<double>>(first, second);
    check_growth_failure<float>(first); check_growth_failure<double>(first);
    check_growth_failure<std::complex<float>>(first); check_growth_failure<std::complex<double>>(first);
}
} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int failures = 0;
    try {
        const char* required = std::getenv("DDLA_WORKSPACE_TRACE_REQUIRED");
        if (required && std::strcmp(required, "1") == 0)
            require(workspace_test::available(), "required vector diagnostics are missing");
        bool rejected = false;
        try { ddla::vector<double> invalid(nullptr); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "null handle accepted");
        std::vector<ddla::DdlaHandle_t> handles;
        auto add_handle = [&](ddla::DdlaBackend backend) {
            ddla::DdlaHandle_t h = nullptr;
            ddla::DDLA_CHECK(ddla::ddlaInit(h, backend));
            ddla::DDLA_CHECK(ddla::ddlaSet(h, MPI_COMM_WORLD));
            handles.push_back(h);
        };
        add_handle(ddla::DdlaBackend::CPU);
#if DDLA_HAS_GPU
        add_handle(ddla::DdlaBackend::GPU);
#endif
        for (auto first : handles) for (auto second : handles) check_handles(first, second);
        for (auto& h : handles) {
            ddla::DDLA_CHECK(ddla::ddlaSynchronize(h));
            ddla::DDLA_CHECK(ddla::ddlaDestroy(h));
        }
    } catch (const std::exception& e) {
        std::cerr << "rank " << rank << ": " << e.what() << '\n';
        failures = 1;
    }
    int total = 0;
    MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) std::cout << "test_backend_vector " << (total == 0 ? "passed" : "failed") << '\n';
    MPI_Finalize();
    return total == 0 ? 0 : 1;
}
