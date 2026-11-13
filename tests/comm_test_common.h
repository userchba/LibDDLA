#ifndef DDLA_COMM_TEST_COMMON_H
#define DDLA_COMM_TEST_COMMON_H
#include "mpi_test_init.h"
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "comm_traits.h"
#include "workspace_test_common.h"

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace ddla;
namespace comm_test {
void check(DdlaHandle_t h,bool good,const char* what)
{
    int bad=!good,any=0;
    MPI_CHECK(MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,h->comm));
    if(any) throw std::runtime_error(what);
}
template<class T>T value(int rank,std::size_t i)
{
    using R=typename real_type<T>::type;
    const R re=R(32*(rank+1)+i%31);
    if constexpr(std::is_arithmetic_v<T>)return re;
    else return T(re,R(8*(rank+1)+i%7));
}
template<class T>void storage(DdlaHandle_t h)
{
    DdlaStream first,second;
    const auto empty=first.host_buffers<T>(0);
    check(h,!empty[0]&&!empty[1],"empty host slots allocate no storage");
    auto [send,recv]=first.host_buffers<T>(257,131);
    check(h,send && recv && send!=recv &&
        reinterpret_cast<std::uintptr_t>(send)%alignof(T)==0 &&
        reinterpret_cast<std::uintptr_t>(recv)%alignof(T)==0,"host slots separate and aligned");
    for(int i=0;i<257;++i)send[i]=value<T>(0,i);
    for(int i=0;i<131;++i)recv[i]=value<T>(1,i);
    const auto smaller=first.host_buffers<T>(1,1);
    check(h,smaller[0]==send && smaller[1]==recv,"host slots retain high-water addresses");
    const auto grown=first.host_buffers<T>(1025,131);
    bool good=grown[1]==recv;
    for(int i=0;i<257;++i)good=good && grown[0][i]==value<T>(0,i);
    for(int i=0;i<131;++i)good=good && grown[1][i]==value<T>(1,i);
    const auto other=second.host_buffers<T>(257,131);
    check(h,good && other[0]!=grown[0] && other[1]!=grown[1],"host growth preserves data and handle independence");
    const auto bytes=first.host_buffers<std::byte>(sizeof(T),sizeof(T));
    check(h,bytes[0]==reinterpret_cast<std::byte*>(grown[0]) &&
        bytes[1]==reinterpret_cast<std::byte*>(grown[1]),"host storage reused across scalar types");
    if constexpr(sizeof(T)>1) {
        bool caught=false;
        try {first.host_buffers<T>(std::numeric_limits<std::size_t>::max()/sizeof(T)+1);}
        catch(const std::length_error&){caught=true;}
        check(h,caught,"host byte-count overflow rejected");
    }
}
template<class T>std::vector<T> download(DdlaHandle_t h,const vector<T>& d)
{
    std::vector<T> result(d.size());
    DDLA_CHECK(ddlaMemcpy(h,result.data(),d.data(),result.size()*sizeof(T)));
    DDLA_CHECK(ddlaSynchronize(h));return result;
}
// This handle has never communicated: verify both cold growth and the hot path
// without an algorithm reserving any cache in advance.
template<DdlaBackend B>void lazy_growth(DdlaHandle_t h)
{
    if constexpr(B==DdlaBackend::GPU) {
        if(ddlaGetCommunication(h)!=DdlaCommunication::HostStagedMPI)return;
        if(!workspace_test::available()) {
            workspace_test::Scope required;
            return;
        }
        const auto initial=h->host_buffers<std::byte>(0);
        check(h,!initial[0]&&!initial[1],"fresh handle has no host communication cache");
        std::size_t largest=0;
        for(int count:{0,19,131073,131073,7}) {
            vector<double> send(h,count+3),recv(h,count+3);
            std::vector<double> input(count+3,-937),zero(count+3,-937);
            for(int i=0;i<count;++i)input[i]=value<double>(h->myid,i);
            DDLA_CHECK(ddlaMemcpy(h,send.data(),input.data(),input.size()*sizeof(double)));
            DDLA_CHECK(ddlaMemcpy(h,recv.data(),zero.data(),zero.size()*sizeof(double)));
            DDLA_CHECK(ddlaSynchronize(h));
            workspace_test::Scope scope;
            commAllReduce<B>(h,CommScope::Grid,count?send.data():nullptr,count?recv.data():nullptr,count,cclSum);
            const auto trace=scope.finish();
            const bool grows=std::size_t(count)>largest;
            const bool no_algorithm_storage=!trace.ddla_alloc&&!trace.ddla_free&&!trace.pinned_alloc&&
                !trace.raw_gpu_alloc&&!trace.vendor_query&&!trace.resource_setup;
            check(h,no_algorithm_storage && (grows?trace.host_new>0:trace.host_new==0),
                "only a new message maximum allocates host communication storage");
            largest=std::max(largest,std::size_t(count));
            const auto got=download(h,recv);bool good=true;
            for(int i=0;i<count;++i){double sum=0;for(int rank=0;rank<h->nprocs;++rank)sum+=value<double>(rank,i);good=good&&got[i]==sum;}
            for(std::size_t i=count;i<got.size();++i)good=good&&got[i]==-937;
            check(h,good,"lazy allreduce values and guards");
        }
        if(!h->myid)std::cout<<"COMM_LAZY_GROWTH_PASS grid="<<h->nprows_<<'x'<<h->npcols_<<std::endl;
    }
}
template<DdlaBackend B,class T>void messages(DdlaHandle_t h,CommScope scope,int count)
{
    const MPI_Comm comm=Communicator::mpi_comm(h,scope);
    int rank,size;MPI_CHECK(MPI_Comm_rank(comm,&rank));MPI_CHECK(MPI_Comm_size(comm,&size));
    const T guard=T(-937);
    vector<T> send(h,count+3),recv(h,count+3);
    std::vector<T> input(count+3,guard),initial(count+3,guard);
    for(int i=0;i<count;++i)input[i]=value<T>(rank,i);
    const auto upload=[&]{
        DDLA_CHECK(ddlaMemcpy(h,send.data(),input.data(),input.size()*sizeof(T)));
        DDLA_CHECK(ddlaMemcpy(h,recv.data(),initial.data(),initial.size()*sizeof(T)));
    };
    const auto verify=[&](const vector<T>& d,int length,auto expected,const char* label){
        const auto got=download(h,d);bool good=true;
        for(std::size_t i=0;i<got.size();++i)good=good && got[i]==(i<std::size_t(length)?expected(i):guard);
        check(h,good,label);
    };
    const auto sum=[&](std::size_t i){T total=0;for(int p=0;p<size;++p)total+=value<T>(p,i);return total;};
    const int root=size-1;
    upload();
    if(rank==root)DDLA_CHECK(ddlaMemcpy(h,recv.data(),input.data(),input.size()*sizeof(T)));
    commBcast<B>(h,scope,count?recv.data():nullptr,count,root);
    verify(recv,count,[&](std::size_t i){return value<T>(root,i);},"broadcast value/guard");

    upload();
    const int peer=rank%2 ? rank-1 : std::min(rank+1,size-1);
    commSendrecv<B>(h,scope,count?send.data():nullptr,count?recv.data():nullptr,count,peer);
    verify(recv,count,[&](std::size_t i){return value<T>(peer,i);},"equal-count sendrecv value/guard");

    upload();
    const int dest=(rank+1)%size,source=(rank+size-1)%size;
    const int sc=count-std::min(count,rank%3),rc=count-std::min(count,source%3);
    commSendrecv<B>(h,scope,sc?send.data():nullptr,sc,dest,rc?recv.data():nullptr,rc,source);
    verify(recv,rc,[&](std::size_t i){return value<T>(source,i);},"unequal-count sendrecv value/guard");

    upload();
    if(peer!=rank) {
        if(rank%2==0) {
            commSend<B>(h,scope,count?send.data():nullptr,count,peer);
            commRecv<B>(h,scope,count?recv.data():nullptr,count,peer);
        } else {
            commRecv<B>(h,scope,count?recv.data():nullptr,count,peer);
            commSend<B>(h,scope,count?send.data():nullptr,count,peer);
        }
    }
    verify(recv,peer!=rank?count:0,[&](std::size_t i){return value<T>(peer,i);},"send/recv value/guard");

    upload();
    const auto op=cclSum;
    commAllReduce<B>(h,scope,count?send.data():nullptr,count?recv.data():nullptr,count,op);
    verify(recv,count,sum,"allreduce value/guard");
    upload();
    // In place: sbuf == rbuf is legal for NCCL/the tunnel and maps to
    // MPI_IN_PLACE on the MPI paths (aliased MPI buffers are erroneous).
    commAllReduce<B>(h,scope,count?send.data():nullptr,count?send.data():nullptr,count,op);
    verify(send,count,sum,"in-place allreduce value/guard");
    upload();
    // The root contributes and receives through the same device/host buffer.
    commReduceSum<B>(h,scope,send.data(),rank==root?send.data():recv.data(),count,root);
    if(rank==root) {
        const auto got=download(h,send);bool good=true;
        for(int i=0;i<count;++i)good=good && got[i]==sum(i);
        for(std::size_t i=count;i<got.size();++i)good=good && got[i]==guard;
        // All ranks still enter the single grid-wide check below.
        int bad=!good;MPI_CHECK(MPI_Bcast(&bad,1,MPI_INT,root,comm));
        check(h,!bad,"reduce in-place root value/guard");
    } else {
        int bad=0;MPI_CHECK(MPI_Bcast(&bad,1,MPI_INT,root,comm));
        check(h,!bad,"reduce in-place root value/guard");
    }
    verify(recv,0,[&](std::size_t){return guard;},"reduce non-root output untouched");

    // Allgather is also used by the CCL one-stage reduction. Exercise all
    // transports with zero counts, every scalar type and guarded output.
    upload();
    vector<T> gathered(h,std::size_t(count)*size+3);
    const std::vector<T> gather_initial(gathered.size(),guard);
    DDLA_CHECK(ddlaMemcpy(h,gathered.data(),gather_initial.data(),gather_initial.size()*sizeof(T)));
    commAllGather<B>(h,scope,count?send.data():nullptr,count?gathered.data():nullptr,count);
    verify(gathered,count*size,[&](std::size_t i){return value<T>(int(i/count),i%count);},
           "allgather values and guards");
    verify(send,count,[&](std::size_t i){return value<T>(rank,i);},"allgather immutable send buffer");

    std::vector<int> ns(size),nr(size),ds(size),dr(size);
    int st=0,rt=0;
    for(int p=0;p<size;++p) {
        const int mode=(rank+p)%3;
        ns[p]=nr[p]=count && mode ? count/size+mode : 0;
        ds[p]=st;dr[p]=rt;
        st+=ns[p];rt+=nr[p];
    }
    std::vector<T> sh(st+3,guard),rh(rt+3,guard),want=rh;
    for(int p=0;p<size;++p)for(int i=0;i<ns[p];++i)sh[ds[p]+i]=value<T>(rank,std::size_t(p)*7+i);
    for(int p=0;p<size;++p)for(int i=0;i<nr[p];++i)want[dr[p]+i]=value<T>(p,std::size_t(rank)*7+i);
    vector<T> a(h,sh.size()),b(h,rh.size());
    DDLA_CHECK(ddlaMemcpy(h,a.data(),sh.data(),sh.size()*sizeof(T)));
    DDLA_CHECK(ddlaMemcpy(h,b.data(),rh.data(),rh.size()*sizeof(T)));
    commAlltoallv<B>(h,scope,size,count?a.data():nullptr,ns.data(),ds.data(),count?b.data():nullptr,nr.data(),dr.data());
    check(h,download(h,b)==want && download(h,a)==sh,"alltoallv packed values, guards and immutable send buffer");
}
template<DdlaBackend B>void run(MPI_Comm comm,int pr,int pc,bool staged=false)
{
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,B));
    if constexpr(B==DdlaBackend::GPU)if(staged)DDLA_CHECK(ddlaSetCommunication(h,DdlaCommunication::HostStagedMPI));
    DDLA_CHECK(ddlaSet(h,comm,pr,pc));
    lazy_growth<B>(h);
    storage<float>(h);storage<double>(h);storage<std::complex<float>>(h);storage<std::complex<double>>(h);
    for(auto scope:{CommScope::Grid,CommScope::Row,CommScope::Col})for(int count:{0,1,37,131073}) {
        messages<B,float>(h,scope,count);messages<B,double>(h,scope,count);
        messages<B,std::complex<float>>(h,scope,count);messages<B,std::complex<double>>(h,scope,count);
    }
    if(!h->myid)std::cout<<"COMM_HOST_BUFFERS_PASS backend="<<int(B)<<" grid="<<pr<<'x'<<pc<<std::endl;
    DDLA_CHECK(ddlaDestroy(h));
}
}

#endif
