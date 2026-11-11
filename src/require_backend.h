#ifndef REQUIRE_BACKEND_H
#define REQUIRE_BACKEND_H

// ---------------------------------------------------------------------------
// Internal helper: the entry check of every backend-templated routine.  A null handle, or a handle whose backend is
// not the template's, is rejected before any work starts (the same check pgemm performs inline).  The static_assert
// turns a call for a backend this build does not contain into a compile error instead of a missing symbol.
// ---------------------------------------------------------------------------

#include <ddla/ddla_config.h>
#include <ddla/ddla_handle_t.h>
#include <stdexcept>
#include <string>

namespace ddla {
namespace detail {

// For the phases that take the backend as an argument instead of a template parameter.
inline void require_backend(const DdlaHandle_t& handle, DdlaBackend backend, const char* routine_name)
{
    if (handle == nullptr) {
        throw std::runtime_error(std::string(routine_name) + ": null handle");
    }
    if (ddlaGetBackend(handle) != backend) {
        throw std::runtime_error(
            std::string(routine_name) + ": template backend " +
            (backend == DdlaBackend::CPU ? "CPU" : "GPU") +
            " does not match the handle backend " +
            (backend == DdlaBackend::CPU ? "GPU" : "CPU"));
    }
}

template <DdlaBackend Backend>
inline void require_backend(const DdlaHandle_t& handle, const char* routine_name)
{
    static_assert(Backend != DdlaBackend::GPU || DDLA_HAS_GPU,
                  "the GPU backend is not available in this LibDDLA build");
    require_backend(handle, Backend, routine_name);
}

} // namespace detail
} // namespace ddla

#endif // REQUIRE_BACKEND_H
