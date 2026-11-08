#ifndef DDLA_PUNMTR_HE2HB_GROUPED_H
#define DDLA_PUNMTR_HE2HB_GROUPED_H
#include <ddla/ddla.h>
#include <cstddef>

namespace ddla {
namespace spectral_impl {
struct UnmtrHe2hbPlan;
}

namespace detail {
// Grouped compact-WY application, shared by CPU and GPU backends.
struct GroupedStage5Options
{
    int group_cap = 3; // 1, 2 or 3; identical on every rank.
};

struct GroupedStage5Plan
{
    char trans = 'N';
    int n = 0;
    int nbw = 1;
    int nb = 1;
    int steps = 0;
    int zcols = 0;
    int max_rows = 0;
    int pr = 1;
    int pc = 1;
    int mr = 0;
    int mc = 0;
    GroupedStage5Options options{};
    int lldA = 1;
    int lldZ = 1;
};

GroupedStage5Plan grouped_stage5_geometry(const spectral_impl::UnmtrHe2hbPlan&,
                                          const GroupedStage5Options&);

template <class T>
GroupedStage5Plan grouped_stage5_geometry(
    const DdlaHandle_t&, char, int, int, const int*, const int*, const GroupedStage5Options&);

template <class T>
GroupedStage5Plan grouped_stage5_check(const DdlaHandle_t&,
                                       char,
                                       int,
                                       int,
                                       const T*,
                                       const int*,
                                       const T*,
                                       T*,
                                       const int*,
                                       const GroupedStage5Options&);

// Backend bytes, in carving order: V (max_rows x K), W and W2
// (K x zcols), S (K x K), G and G2 (the Gram blocks of one group), where K is
// the widest group's column count.
template <class T>
void grouped_stage5_workspace_size(const DdlaHandle_t&,
                                   const GroupedStage5Plan&,
                                   std::size_t* backend_bytes);

template <DdlaBackend Backend, class T>
void grouped_stage5_run(const DdlaHandle_t&,
                        const GroupedStage5Plan&,
                        const T* A,
                        const T* tmat,
                        T* Z,
                        int& info,
                        void* backend_work,
                        std::size_t backend_bytes);

// A valid handle and consistent arguments on every rank are required.
// Argument checks are local. A's source coordinates are zero; its square block size is divisible
// by nbw. Z's row distribution matches A; its column layout is otherwise free.
// A's leading n-by-n data and tmat are read-only. Local A/Z active spans must be
// disjoint, and active Z must not overlap tmat's complete panel-slot span.
// Matrix spans are ((local_cols-1)*LLD+local_rows)*sizeof(T), or zero when
// either local extent is zero; intermediate LLD gaps remain conservative.
// Disjoint active spans may share a larger parent allocation, including unused
// parent columns or final-column padding. Parent descriptors and LLDs are supported;
// nonempty active spans require nonnull data. As in the current operation,
// complex 'T' aliases 'C'. No panels means identity and permits null data pointers.
//
// No metadata/error-agreement collectives or failure-drain schedule are added.
// The caller handles distributed failures. info=0 on success; -1/-2/-3 report
// local validation/allocation/execution errors. Buffer destructors never enter
// MPI; no full matrix is allocated.
template <DdlaBackend Backend, class T>
void punmtr_he2hb_grouped(const DdlaHandle_t& handle,
                          char trans,
                          int n,
                          int nbw,
                          const T* A,
                          const int* descA,
                          const T* tmat,
                          T* Z,
                          const int* descZ,
                          const GroupedStage5Options& options,
                          int& info);
}
}
#endif
