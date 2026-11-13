#ifndef DDLA_PHERM_COMPLETE_WORKSPACE_H
#define DDLA_PHERM_COMPLETE_WORKSPACE_H

#include <ddla/ddla.h>
#include <cstddef>

namespace ddla { namespace detail {

// Private solver entry point. GPU handles may use scratch for the combined
// send/receive packs when scratch_elements is sufficient on this rank.
// The caller supplies contiguous T storage disjoint from the full A span;
// its initial values are not read, and its ownership stays with the caller.
// Insufficient/null scratch and CPU handles retain the owned-buffer path.
// Every rank follows the same communication sequence even when only some
// ranks borrow storage. Plan/workspace allocation failures are agreed before
// exchange; borrowed storage is never passed to ddlaFree.
template <DdlaBackend Backend, typename T>
void pherm_complete_with_workspace(const DdlaHandle_t& handle, char uplo, int n,
                                   T* A, const int* descA,
                                   T* scratch, std::size_t scratch_elements, bool hermitian=true);

} }
#endif
