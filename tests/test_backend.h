#ifndef DDLA_TEST_BACKEND_H
#define DDLA_TEST_BACKEND_H
#include <ddla/ddla_handle_t.h>
namespace ddla_test {
// Existing single-backend fixtures exercise GPU in a GPU build and CPU otherwise.
constexpr ddla::DdlaBackend execution_backend = DDLA_HAS_GPU
    ? ddla::DdlaBackend::GPU : ddla::DdlaBackend::CPU;
}
#endif
