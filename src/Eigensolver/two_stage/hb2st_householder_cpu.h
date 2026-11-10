#ifndef DDLA_HB2ST_HOUSEHOLDER_CPU_H
#define DDLA_HB2ST_HOUSEHOLDER_CPU_H

#include <ddla/ddla.h>
#include <cstddef>
#include <memory>
#include <vector>

namespace ddla { namespace detail {

// Private verification controls. No public algorithm switch or environment
// setting selects this prototype.
struct Hb2stHouseholderCpuOptions {
    enum class Failure { None, Metadata, Band, History, Factors, Reduce, Apply, Packet };
    enum class Stage2 { Scalar, Blas, Gpu };
    Stage2 stage2 = Stage2::Blas;
    enum class Stage4 {
        Wy=0, Paired=1, FmaChain=2, FmaIndependent=3,
        GroupedReal4=4, GroupedReal6=5
    };
    Stage4 stage4 = Stage4::Paired;
    Failure failure = Failure::None;
    int failure_rank = -1;
    bool reject_memory_model = false;
};

struct Hb2stHouseholderCpuPlan {
    int n=0,band=1,rank=0,ranks=1,threads=1,chase_threads=1;bool keep_history=false;
    Hb2stHouseholderCpuOptions options{};
};
// Trusted geometry is local, pointer-free and policy-free. Composite checks
// freeze the thread budget/options before composing this plan.
Hb2stHouseholderCpuPlan hh_cpu_geometry(const DdlaHandle_t&,int,int,bool,const Hb2stHouseholderCpuOptions&,int threads);
Hb2stHouseholderCpuPlan hh_cpu_check(const DdlaHandle_t&,int,int,bool,const Hb2stHouseholderCpuOptions& = {});
// Backend bytes (CPU handle) of reduce_run: the reflector v, its product A v
// and w, band elements each (band = the plan's band).
template<class T> void hh_cpu_reduce_workspace_size(const DdlaHandle_t&,int band,int chase_threads,std::size_t* backend_bytes);
// Backend bytes (CPU handle) of apply_columns_run: the 32 x columns product V^H C.
template<class T> void hh_cpu_apply_columns_workspace_size(const DdlaHandle_t&,int columns,std::size_t* backend_bytes);

inline bool hh_gpu_compact_band(int band) noexcept {
    return band>=2 && band<=64 && (band&(band-1))==0;
}
template<class T> void hh_gpu_apply_workspace_size(int band,int columns,bool wy,std::size_t* bytes);

namespace hh { template<class T> struct BandView;struct Event; }
// Private event entry for focused mapping/orientation and wrapper-failure tests.
// Scratch arrays each contain at least band b elements; no allocation or MPI.
template<class T>
bool hh_event_blas_cpu(const DdlaHandle_t&,hh::BandView<T>,const hh::Event&,
                       T* v,T* cross,T* w,typename real_type<T>::type& tau) noexcept;

template<class T> class Hb2stHouseholderCpu {
public:
    using R = typename real_type<T>::type;
    Hb2stHouseholderCpu(const DdlaHandle_t&, int n, int band,
                       bool keep_history,
                       const Hb2stHouseholderCpuOptions& = {});
    // Prepared MPI/shared-window owner of a checked plan. Construction and
    // release are outside check/size/numerical runs.
    Hb2stHouseholderCpu(const DdlaHandle_t&,const Hb2stHouseholderCpuPlan&);
    ~Hb2stHouseholderCpu();
    Hb2stHouseholderCpu(const Hb2stHouseholderCpu&) = delete;
    Hb2stHouseholderCpu& operator=(const Hb2stHouseholderCpu&) = delete;

    // reduce returns the normalized tridiagonal; multiply eigenvalues from
    // pstedc by eigenvalue_scale() only after its normalized solve.
    // All methods are collective on the handle communicator, including calls
    // by ranks with no input or output elements. Complex phases store D(row)
    // at phases[row-1], D(0)=1. Real e remains signed and phases is empty
    // (reduce_run does not write it).
    // As for the distributed BLAS, descriptor global fields agree across
    // ranks; LLD may differ by rank. Only the active stored upper band is read.
    void reduce(const T* A, const int* descA, R* d, R* e,
                std::vector<T>& phases);
    void reduce_check(const T*,const int*,R*,R*,T* phases,std::size_t phase_count) const;
    // packed_upper is an optional host (band+1)-by-n upper band for a GPU solve.
    void reduce_run(const T*,const int*,R*,R*,T* phases,void* backend_work,std::size_t backend_bytes,
                    const T* packed_upper=nullptr);
#if DDLA_HAS_GPU
    // Device row-major n-by-ncols strip (the existing GPU replay layout).
    // A GPU driver constructs this host-chase owner from a trusted plan and
    // supplies packed_upper to reduce_run. Its device apply uses bounded
    // short-reflector packets, or WY factors through existing GPU BLAS.
    void apply_rows_gpu_run(T*,int ld,int ncols,void*,std::size_t);
#endif
    // Applies Q_band only: callers apply D while converting their real rows.
    void apply_columns(T* columns, int ld, int ncols);
    void apply_columns_check(T*,int,int) const;
    void apply_columns_run(T*,int,int,void* backend_work,std::size_t backend_bytes);
    // Split-real row tiles; kernel widths8/16/32/64, inactive lanes untouched.
    void apply_tiles(R* tiles, int width, int ncols);
    void apply_tiles_check(R*,int,int) const;
    void apply_tiles_run(R*,int,int);
    // All accepted methods except Wy use split-real tiles in the rows entry.
    // Direct apply_columns remains a WY reference for every selected method.
    bool paired_replay() const;
    // Explicit, idempotent collective cleanup. The destructor never calls MPI.
    void release();
    std::size_t local_history_elements() const;
    std::size_t total_history_elements() const;
    std::size_t local_history_bytes() const;
    int panel_width() const;
    int order() const;
    int threads() const;
    R eigenvalue_scale() const;

private:
    void apply_wy_run(T*,int,int,void*,std::size_t,bool device_rows);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};


} }
#endif
