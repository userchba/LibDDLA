#ifndef DDLA_DISTRIBUTED_HISTORY_H
#define DDLA_DISTRIBUTED_HISTORY_H

#include <ddla/ddla.h>
#include "ddla_connector.h"
#include "check_agree.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ddla { namespace detail {

// Integer-only copy of the chase's recording cursor. A checkpoint is the
// state BEFORE one step, so chunks that split a step retain an offset too.
struct ChaseCursor {
    int n = 0, kd = 0, i = 1, k = 0, j1 = 0, j2 = 1;
    long long base = 0;
    ChaseCursor() = default;
    ChaseCursor(int n_, int kd_) : n(n_), kd(kd_), k(std::min(n-1,kd)+1), j1(k+1) {}
    bool done() const { return kd <= 1 || i > n-2; }
    int first() const { return j1 + std::min(n-1,kd) - (k>2 ? std::min(n-1,kd)+1 : 0) - 1; }
    int count() const {
        if (done()) return 0;
        const int end = std::min(n, j2+std::min(n-1,kd));
        return end > first() ? (end-first()+kd)/(kd+1) : 0;
    }
    void next() {
        if (done()) return;
        base += count();
        const int kdn = std::min(n-1,kd);
        j1 += kdn; j2 += kdn;
        if (k>2) j1 -= kdn+1;
        if (j2+kdn>n) j2 -= kdn+1;
        if (--k<2) { k=kdn+1; ++i; }
    }
};

struct DistributedHistoryHeader {ChaseCursor cursor;int offset=0,count=0;};
struct DistributedHistoryPlan {int n=0,kd=1,rank=0,ranks=1;};
inline DistributedHistoryPlan distributed_history_check(int n,int kd,int rank,int ranks)
{
    if(n<0||kd<1||ranks<1||rank<0||rank>=ranks)
        throw std::invalid_argument("distributed history: invalid dimensions or rank");
    return {n,kd,rank,ranks};
}
// The checkpoint of every chunk this rank owns: chunk q belongs to rank
// q % ranks and is its header q / ranks.
inline void distributed_history_headers(const DistributedHistoryPlan& p,long long chunks,
    long long total,DistributedHistoryHeader* headers)
{
    ChaseCursor cursor(p.n,p.kd);long long q=0;
    while(!cursor.done()) {
        const auto end=cursor.base+cursor.count();
        while(q<chunks&&q*262144<end) {
            if(q%p.ranks==p.rank) {
                auto& h=headers[static_cast<std::size_t>(q/p.ranks)];h.cursor=cursor;
                h.offset=static_cast<int>(q*262144-cursor.base);h.count=static_cast<int>(std::min(262144LL,total-q*262144));
            }
            ++q;
        }
        cursor.next();
    }
}

template<class T> class DistributedHistory {
public:
    using rt = typename real_type<T>::type;
    static constexpr long long chunk_size = 262144;
    using Header=DistributedHistoryHeader;
    bool ready = false;
    long long total = 0;
    int rank = 0, ranks = 1, n = 0, kd = 0;
    std::vector<Header> headers;
    std::unique_ptr<rt[]> cosines;
    std::unique_ptr<T[]> sines;
    std::size_t local_count = 0;

    void clear() { ready=false; total=0; local_count=0; headers.clear(); cosines.reset(); sines.reset(); }
    long long chunks() const { return (total+chunk_size-1)/chunk_size; }
    void prepare(int n_, int kd_, MPI_Comm comm) {
        clear(); n=n_; kd=kd_;
        MPI_CHECK(MPI_Comm_rank(comm,&rank)); MPI_CHECK(MPI_Comm_size(comm,&ranks));
        std::exception_ptr failure;
        try {
        const auto plan=distributed_history_check(n,kd,rank,ranks);
        ChaseCursor cursor(n,kd);
        while (!cursor.done()) cursor.next();
        total=cursor.base;
        const long long nc=chunks();
        const long long own=nc>rank ? (nc-1-rank)/ranks+1 : 0;
        headers.resize(static_cast<std::size_t>(own));
        if(static_cast<unsigned long long>(own)>std::numeric_limits<std::size_t>::max()/chunk_size)
            throw std::overflow_error("distributed history: local count overflow");
        local_count=static_cast<std::size_t>(own)*chunk_size;
        if (own && (nc-1)%ranks==rank) local_count-=static_cast<std::size_t>(chunk_size-(total-(nc-1)*chunk_size));
        if(local_count>std::numeric_limits<std::size_t>::max()/sizeof(T) ||
           local_count>std::numeric_limits<std::size_t>::max()/sizeof(rt))
            throw std::overflow_error("distributed history: local byte count overflow");
        cosines.reset(new rt[std::max<std::size_t>(local_count,1)]);
        sines.reset(new T[std::max<std::size_t>(local_count,1)]);
        distributed_history_headers(plan,nc,total,headers.data());
        ready=true;
        } catch(...) { failure=std::current_exception(); }
        int ok=failure?0:1,all_ok=0;
        MPI_CHECK(MPI_Allreduce(&ok,&all_ok,1,MPI_INT,MPI_MIN,comm));
        if(!all_ok) {
            clear();
            if(failure) std::rethrow_exception(failure);
            throw std::runtime_error("distributed history: allocation failed on another rank");
        }
    }
    // Each node currently has the same bounded producer prefix. Each rank
    // copies ONLY its cyclic ownership; there is no retained node copy.
    void capture(long long begin,long long end,const hb2st_rot<T>* ring,long long capacity) {
        if (!ready || begin<0 || end>total || end<begin || end-begin>capacity)
            throw std::logic_error("distributed history: invalid producer interval");
        for (long long q=begin/chunk_size;q<chunks() && q*chunk_size<end;++q) {
            if (q%ranks!=rank) continue;
            const long long a=std::max(begin,q*chunk_size),b=std::min(end,(q+1)*chunk_size);
            const std::size_t off=static_cast<std::size_t>(q/ranks)*chunk_size;
            for (long long g=a;g<b;++g) {
                const auto& r=ring[g%capacity];
                cosines[off+static_cast<std::size_t>(g-q*chunk_size)]=r.c;
                sines[off+static_cast<std::size_t>(g-q*chunk_size)]=r.s;
            }
        }
    }
    void store_owned(long long g,const hb2st_rot<T>& record) {
        if(!ready || g<0 || g>=total || (g/chunk_size)%ranks!=rank)
            throw std::logic_error("distributed history: wrong coefficient owner");
        const long long q=g/chunk_size;
        const std::size_t at=static_cast<std::size_t>(q/ranks)*chunk_size+static_cast<std::size_t>(g%chunk_size);
        cosines[at]=record.c; sines[at]=record.s;
    }
    // Collective; caller visits q in descending order. All temporary data
    // are bounded by one chunk, including the regenerated record indices.
    // out holds the chunk's records: min(chunk_size, total) of them suffice.
    void fetch(long long q,MPI_Comm comm,hb2st_rot<T>* out, Header* checkpoint=nullptr,
               bool compact_coefficients=false) const {
        fetch_into(q,comm,[out](std::size_t){return out;},checkpoint,compact_coefficients);
    }
    void fetch(long long q,MPI_Comm comm,std::vector<hb2st_rot<T>>& out, Header* checkpoint=nullptr,
               bool compact_coefficients=false) const {
        fetch_into(q,comm,[&out](std::size_t count){out.resize(count);return out.data();},checkpoint,compact_coefficients);
    }
    // The body of both: records(count) returns room for the chunk's count records.
    template<class Records>
    void fetch_into(long long q,MPI_Comm comm,Records&& records,Header* checkpoint,bool compact_coefficients) const {
        std::exception_ptr failure;
        int count=0;hb2st_rot<T>* out=nullptr;
        try {
            if (!ready || q<0 || q>=chunks()) throw std::logic_error("distributed history: invalid chunk");
            count=static_cast<int>(std::min(chunk_size,total-q*chunk_size));
            if(static_cast<std::size_t>(count)>static_cast<std::size_t>(std::numeric_limits<int>::max())/sizeof(hb2st_rot<T>))
                throw std::overflow_error("distributed history: MPI byte count overflow");
            out=records(static_cast<std::size_t>(count));
        } catch(...) { failure=std::current_exception(); }
        detail::agree_failure(comm,failure,"distributed history: fetch allocation");
        const int owner=static_cast<int>(q%ranks);
        Header h{};
        std::int64_t wire[9]={};
        if (rank==owner) {
            h=headers[static_cast<std::size_t>(q/ranks)];
            wire[0]=h.cursor.n; wire[1]=h.cursor.kd; wire[2]=h.cursor.i; wire[3]=h.cursor.k;
            wire[4]=h.cursor.j1; wire[5]=h.cursor.j2; wire[6]=h.cursor.base; wire[7]=h.offset; wire[8]=h.count;
        }
        MPI_CHECK(MPI_Bcast(wire,9,MPI_INT64_T,owner,comm));
        h.cursor.n=static_cast<int>(wire[0]); h.cursor.kd=static_cast<int>(wire[1]);
        h.cursor.i=static_cast<int>(wire[2]); h.cursor.k=static_cast<int>(wire[3]);
        h.cursor.j1=static_cast<int>(wire[4]); h.cursor.j2=static_cast<int>(wire[5]);
        h.cursor.base=wire[6]; h.offset=static_cast<int>(wire[7]); h.count=static_cast<int>(wire[8]);
        if (h.count!=count || h.cursor.base+h.offset!=q*chunk_size || h.cursor.n!=n || h.cursor.kd!=kd)
            throw std::logic_error("distributed history: invalid checkpoint");
        if (checkpoint) *checkpoint=h;
        if (compact_coefficients) {
            // The cursor regenerates j0, so neither it nor record padding
            // belongs on the wire. Reuse the existing record allocation as
            // a packed byte buffer; expand backward after the broadcast so
            // no unread coefficient is overwritten. No extra chunk is kept.
            using Record=hb2st_rot<T>;
            static_assert(std::is_trivially_copyable<Record>::value,"history coefficients require trivial storage");
            constexpr std::size_t stride=sizeof(rt)+sizeof(T);
            static_assert(stride<=sizeof(Record),"packed history exceeds record storage");
            auto* packed=reinterpret_cast<unsigned char*>(out);
            if(rank==owner) {
                const std::size_t off=static_cast<std::size_t>(q/ranks)*chunk_size;
                for(int j=0;j<count;++j) {
                    std::memcpy(packed+std::size_t(j)*stride,&cosines[off+j],sizeof(rt));
                    std::memcpy(packed+std::size_t(j)*stride+sizeof(rt),&sines[off+j],sizeof(T));
                }
            }
            MPI_CHECK(MPI_Bcast(packed,static_cast<int>(static_cast<std::size_t>(count)*stride),MPI_BYTE,owner,comm));
            for(int j=count;j-->0;) {
                rt c; T s;
                std::memcpy(&c,packed+std::size_t(j)*stride,sizeof(rt));
                std::memcpy(&s,packed+std::size_t(j)*stride+sizeof(rt),sizeof(T));
                std::memcpy(&out[j].c,&c,sizeof(rt));
                std::memcpy(&out[j].s,&s,sizeof(T));
            }
        } else {
            // The reference wire format includes initialized padding.
            std::memset(static_cast<void*>(out),0,static_cast<std::size_t>(count)*sizeof(hb2st_rot<T>));
            if (rank==owner) {
                const std::size_t off=static_cast<std::size_t>(q/ranks)*chunk_size;
                for(int j=0;j<count;++j) { out[j].c=cosines[off+j]; out[j].s=sines[off+j]; }
            }
            MPI_CHECK(MPI_Bcast(out,static_cast<int>(static_cast<std::size_t>(count)*sizeof(hb2st_rot<T>)),MPI_BYTE,owner,comm));
        }
        ChaseCursor cursor=h.cursor;
        int offset=h.offset,filled=0;
        while(filled<count && !cursor.done()) {
            const int nr=cursor.count();
            while(offset<nr && filled<count) out[filled++].j0=cursor.first()+(offset++)*(kd+1);
            if(offset==nr) { cursor.next(); offset=0; }
        }
        if(filled!=count || cursor.base+offset!=q*chunk_size+count)
            throw std::logic_error("distributed history: checkpoint length mismatch");
    }
};

} } // namespace ddla::detail
#endif
