#ifndef DDLA_HB2ST_REPLAY_STORAGE_H
#define DDLA_HB2ST_REPLAY_STORAGE_H

#include <ddla/ddla.h>
#include "distributed_history.h"
#include <cstddef>
#include <vector>
#include <optional>

namespace ddla { namespace detail {

struct Hb2stReplayStoragePlan {
    bool enabled=false,borrow_y=false;
    int n=0,P=1,rank=0,npr=1,npc=1,mb=1,nb=1,rsrc=0,csrc=0,ldy=1;
    int first=0,count=0,spill=0;char major='R';
    std::size_t cells=0,real_capacity=0,y_count=0;
};
template<class T>Hb2stReplayStoragePlan hb2st_replay_storage_check(const DdlaHandle_t&,int,T*,const int*,T*,const int*);
// Exact historical capacity/n allocation, then descending remainder with rank
// tie-breaking. No owning ranking container is required by size or run.
int hb2st_replay_storage_prefix(const Hb2stReplayStoragePlan&,int boundary);
// Host bytes of save_run / restore_run: the compact column prefix (P + 1 ints)
// and the host backup of A (cells), both live from save_run through
// restore_run.  Nothing when the plan is disabled.  Y is the caller's: Z itself
// when plan.borrow_y, otherwise a separate backend slab of plan.y_count reals;
// stage 3 may use A as scratch once save_run returned.
template<class T>void hb2st_replay_storage_workspace_size(const DdlaHandle_t&,const Hb2stReplayStoragePlan&,std::size_t* host_bytes);
// Writes the column prefix at the front of host_work (the replay's compact
// prefix, see hb2st_replay_gpu_check) and backs A up; the backup copy is
// complete when it returns.  restore_run copies the backup into A.  Neither
// releases any caller-owned storage.
template<class T>void hb2st_replay_storage_save_run(const DdlaHandle_t&,const Hb2stReplayStoragePlan&,const T* A,
    void* host_work,std::size_t host_bytes);
template<class T>void hb2st_replay_storage_restore_run(const DdlaHandle_t&,const Hb2stReplayStoragePlan&,T* A,
    void* host_work,std::size_t host_bytes);

// Private GPU-driver storage reuse for separate, identical compact A/Z
// descriptors. Eligibility and all preparation/restoration are collective.
// CPU, P1, parent/padded/different descriptors and overlapping buffers use the
// existing driver path. No user allocation is ever freed by this class.
template<class T>
class Hb2stReplayStorage {
public:
    using R=typename real_type<T>::type;
    Hb2stReplayStorage(const DdlaHandle_t& handle,int n,T* A,const int* descA,T* Z,const int* descZ);
    ~Hb2stReplayStorage() noexcept;
    Hb2stReplayStorage(const Hb2stReplayStorage&)=delete;
    Hb2stReplayStorage& operator=(const Hb2stReplayStorage&)=delete;
    bool enabled() const noexcept { return enabled_; }

    // Back up A before lending it to pstedc. Y borrows Z when its compact
    // real slab fits, otherwise this class owns a separate Y allocation.
    void prepare();
    R* input() const noexcept { return Y_; }
    int ldy() const noexcept { return ldy_; }
    R* scratch() const noexcept { return enabled_?reinterpret_cast<R*>(A_):nullptr; }
    std::size_t scratch_elements() const noexcept { return enabled_?cells_*(sizeof(T)/sizeof(R)):0; }
    const std::vector<int>& column_offsets() const noexcept { return columns_; }

    // Consume Y (free only when owned), replay into A's reusable storage,
    // and redistribute into Z after every rank completes input conversion.
    // After successful consumption input()==nullptr, even if replay fails
    // later. Before consumption a throwing call retains input() for cleanup.
    void replay(const DistributedHistory<T>& history,const std::vector<T>& phases,int& info,
                int panel_columns=0,int replay_records=0,bool use_waves=true);
    // Restore A and release its host backup before unchanged stage5(A,Z).
    void restore();

private:
    DdlaHandle_t h_;
    int n_,P_=1,rank_=0,ldy_=1;
    T* A_;
    T* Z_;
    const int* descA_;
    const int* descZ_;
    std::size_t cells_=0;
    bool enabled_=false,prepared_=false,restored_=false,owns_Y_=false;
    R* Y_=nullptr;
    std::vector<int> columns_;
    Hb2stReplayStoragePlan plan_{};
    std::vector<char> host_work_;   // the prefix and the backup of A, prepare() through restore()
};

} }
#endif
