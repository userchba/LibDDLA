#include "api_grid_test_common.h"
#include "LU/lu_internal.h"
#include "LU/pgetrs_impl.h"
#include "Cholesky/cholesky_workspace.h"
#include "QR/pgeqrf_impl.h"
#include "random_generate_impl.h"
#include "workspace_test_common.h"
#include <array>
#include <cstddef>
#include <cstring>
#include <type_traits>

using namespace api_grid_test;
namespace {
template<class T> T scalar(double r,double i=0){if constexpr(std::is_arithmetic_v<T>)return T(r);else return T(r,i);}
template<class T> double tolerance(){return std::is_same_v<typename real_type<T>::type,float>?2e-4:3e-11;}
bool no_growth(const workspace_test::Snapshot& s){return !s.ddla_alloc && !s.ddla_free && !s.pinned_alloc && !s.raw_gpu_alloc && !s.host_new && !s.resource_setup;}
void query_clean(DdlaHandle_t h,const workspace_test::Snapshot& s){require_close(h,"factorization query is local and allocation-free",no_growth(s)&&!s.copies&&!s.launches&&!s.syncs&&!s.collectives?0:1,0);}
void run_clean(DdlaHandle_t h,const workspace_test::Snapshot& s){require_close(h,"factorization run performs no scratch growth or vendor query",no_growth(s)&&!s.vendor_query?0:1,0);}
template<class T> void same(DdlaHandle_t h,const std::vector<T>& a,const std::vector<T>& b,const std::string& label="checked and raw factorization agree including padding"){double err=0;for(std::size_t i=0;i<a.size();++i)err=std::max(err,workspace_test::relative_error(b[i],a[i]));require_close(h,label,err,tolerance<T>());}
template<class T>const char* scalar_name(){if constexpr(std::is_same_v<T,float>)return "S";else if constexpr(std::is_same_v<T,double>)return "D";else if constexpr(std::is_same_v<T,std::complex<float>>)return "C";else return "Z";}
template<class T>void analytic_solve_compare(DdlaHandle_t h,const int* desc,int active_rows,int active_columns,
    const std::vector<T>& expected,const std::vector<T>& actual,const std::string& label)
{
    const int rows=ddla_test::m_loc(h,desc),columns=ddla_test::n_loc(h,desc),ld=desc[DDLA_LLD_];double local[3]={0,0,0};
    double worst=0;int wr=-1,wc=-1,wi=-1,wj=-1,region=-1;
    for(int c=0;c<columns;++c)for(int r=0;r<ld;++r){const int i=r<rows?indx_l2g_r(desc,h,r):-1,j=indx_l2g_c(desc,h,c);const int kind=r>=rows?2:i<active_rows&&j<active_columns?0:1;const auto at=r+std::size_t(c)*ld;
        const auto error=workspace_test::relative_error(actual[at],expected[at]);local[kind]=std::max(local[kind],error);if(error>worst){worst=error;wr=r;wc=c;wi=i;wj=j;region=kind;}}
    double global[3]={};MPI_CHECK(MPI_Allreduce(local,global,3,MPI_DOUBLE,MPI_MAX,h->comm));
    if(worst>tolerance<T>()){const auto at=wr+std::size_t(wc)*ld;std::cerr<<label<<" rank="<<h->myid<<" local=("<<wr<<","<<wc<<") global=("<<wi<<","<<wj<<") region="<<(region==0?"active":region==1?"parent":"LLD-padding")<<" expected="<<expected[at]<<" actual="<<actual[at]<<" relative_error="<<worst<<std::endl;}
    if(!h->myid&&std::max({global[0],global[1],global[2]})>tolerance<T>())std::cerr<<label<<" errors active="<<global[0]<<" parent="<<global[1]<<" LLD-padding="<<global[2]<<std::endl;
    same(h,expected,actual,label+" analytic solution and parent/padding");
}
// Exactly the sized backend and host bytes, between 256-byte guards; one allocation serves every run of a case.
class Guarded {
    static constexpr std::size_t guard=256;
    DdlaHandle_t h_;std::size_t backend_bytes_,host_bytes_;ddla::vector<char> backend_;std::vector<char> host_;
public:
    Guarded(DdlaHandle_t h,std::size_t backend_bytes,std::size_t host_bytes):h_(h),backend_bytes_(backend_bytes),host_bytes_(host_bytes),
        backend_(h,backend_bytes+2*guard),host_(host_bytes+2*guard,char(0x57)){DDLA_CHECK(ddlaMemset(h,backend_.data(),0x57,backend_.size()));DDLA_CHECK(ddlaSynchronize(h));}
    void* backend(){return backend_.data()+guard;}
    void* host(){return host_.data()+guard;}
    void verify(){std::array<char,2*guard> d{};DDLA_CHECK(ddlaMemcpy(h_,d.data(),backend_.data(),guard));DDLA_CHECK(ddlaMemcpy(h_,d.data()+guard,backend_.data()+guard+backend_bytes_,guard));DDLA_CHECK(ddlaSynchronize(h_));bool okay=true;for(std::size_t i=0;i<guard;++i)okay=okay&&host_[i]==char(0x57)&&host_[guard+host_bytes_+i]==char(0x57)&&d[i]==char(0x57)&&d[guard+i]==char(0x57);require_close(h_,"factorization exact-size workspace guards",okay?0:1,0);}
};
struct Env {
    const char* name;bool present;std::string saved;
    Env(const char* n,const char* value):name(n),present(std::getenv(n)!=nullptr),saved(present?std::getenv(n):""){setenv(name,value,1);}
    ~Env(){if(present)setenv(name,saved.c_str(),1);else unsetenv(name);}
};
template<DdlaBackend B,class T> void cases(DdlaHandle_t h)
{
    const int nb=3,n=std::max(11,nb*std::max(h->nprows_,h->npcols_)+2),nrhs=4;
    int da[DDLA_DLEN_],db[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(da,h,n+2,n+2,nb,nb,h->nprows_-1,h->npcols_-1));da[DDLA_LLD_]+=2;
    auto initial=make_local<T>(h,da,[&](int i,int j){return scalar<T>(i==j?30+i:.015*(i+j+1),.002*(i-j));});
    for(int j=0;j<ddla_test::n_loc(h,da);++j)for(int i=ddla_test::m_loc(h,da);i<da[DDLA_LLD_];++i)initial[i+std::size_t(j)*da[DDLA_LLD_]]=scalar<T>(-731);
    DeviceBuffer<T> a(h,initial.size()),ref(h,initial.size());upload(h,a.ptr,initial);upload(h,ref.ptr,initial);
    const int pivcount=std::max(1,num_loc(n,nb,h->myprow_,da[DDLA_RSRC_],h->nprows_));
    std::vector<int> piv(pivcount,-17),pivref(pivcount,-17);DeviceBuffer<int> dp(h,pivcount),dpr(h,pivcount);
    MPI_Barrier(h->comm);bool pure=true;
    if(h->myid==0){workspace_test::Scope scope(true);std::size_t bb=0,hb=0;lu_impl::pgetrf_workspace_size<T>(h,true,n,n,a.ptr,1,1,da,&bb,&hb);const auto s=scope.finish();pure=no_growth(s)&&!s.copies&&!s.launches&&!s.syncs&&!s.collectives;}
    MPI_Barrier(h->comm);require_close(h,"factorization query runs on one rank",pure?0:1,0);
    // All LU panel policies and full/block pivot variants reuse one exactly sized scratch.
    for(bool gather:{false,true}){
        Env policy("DDLA_LU_PANEL",gather?"gather":"inplace");
        // The size query must leave A as it found it (the previous iteration's panel run factored A).
        const auto before=download(h,a.ptr,initial.size());
        workspace_test::Scope qs(true);std::size_t bb=0,hb=0;lu_impl::pgetrf_workspace_size<T>(h,gather,n,n,a.ptr,1,1,da,&bb,&hb);query_clean(h,qs.finish());same(h,before,download(h,a.ptr,initial.size()));Guarded work(h,bb,hb);
        for(int repeat=0;repeat<2;++repeat){upload(h,a.ptr,initial);upload(h,ref.ptr,initial);std::fill(piv.begin(),piv.end(),-17);std::fill(pivref.begin(),pivref.end(),-17);int got=-1,want=-1;pgetrf<T>(h,n,n,ref.ptr,1,1,da,pivref.data(),want);workspace_test::Scope run;lu_impl::pgetrf_run<B,T>(h,gather,n,n,a.ptr,1,1,da,piv.data(),got,work.backend(),bb,work.host(),hb);run_clean(h,run.finish());same(h,download(h,ref.ptr,initial.size()),download(h,a.ptr,initial.size()));require_close(h,"LU info and pivots",got==want&&piv==pivref?0:1,0);work.verify();}
        workspace_test::Scope ps(true);std::size_t pb=0,ph=0;lu_impl::pgetf2_workspace_size<T>(h,gather,n,n,nb,a.ptr,1,1,da,1,&pb,&ph);query_clean(h,ps.finish());Guarded panelwork(h,pb,ph);
        upload(h,a.ptr,initial);upload(h,ref.ptr,initial);int got=0,want=0;pgetf2<B,T>(h,n,n,nb,ref.ptr,1,1,da,pivref.data(),want);workspace_test::Scope pr;lu_impl::pgetf2_run<B,T>(h,gather,n,n,nb,a.ptr,1,1,da,piv.data(),got,1,panelwork.backend(),pb,panelwork.host(),ph);run_clean(h,pr.finish());same(h,download(h,ref.ptr,initial.size()),download(h,a.ptr,initial.size()));panelwork.verify();
    }
    for(bool bpiv:{false,true}){
        workspace_test::Scope qs(true);std::size_t bb=0,hb=0;if(bpiv)lu_impl::pgetrf_bpiv_workspace_size<T>(h,n,n,a.ptr,da,&bb,&hb);else lu_impl::pgetrf_nopiv_workspace_size<T>(h,n,n,da,&bb);query_clean(h,qs.finish());Guarded work(h,bb,hb);
        for(int repeat=0;repeat<2;++repeat){upload(h,a.ptr,initial);upload(h,ref.ptr,initial);int got=0,want=0;if(bpiv)pgetrf_bpiv<T>(h,n,n,ref.ptr,da,dpr.ptr,want);else pgetrf_nopiv<T>(h,n,n,ref.ptr,da,want);workspace_test::Scope run;if(bpiv)lu_impl::pgetrf_bpiv_run<B,T>(h,n,n,a.ptr,da,dp.ptr,got,work.backend(),bb,work.host(),hb);else lu_impl::pgetrf_nopiv_run<B,T>(h,n,n,a.ptr,da,got,work.backend(),bb);run_clean(h,run.finish());same(h,download(h,ref.ptr,initial.size()),download(h,a.ptr,initial.size()));require_close(h,"block LU info",got==want?0:1,0);work.verify();}
    }
    for(char side:{'L','R'})for(char trans:{'N','C'}){
        DDLA_CHECK(ddlaDescInit(db,h,side=='L'?n+2:nrhs+2,side=='L'?nrhs+2:n+2,nb,nb,da[DDLA_RSRC_],da[DDLA_CSRC_]));db[DDLA_LLD_]+=2;
        auto matrix=[](int i,int j){return scalar<T>(i==j?30+i:.015*(i+j+1),.002*(i-j));};
        auto solution=[](int i,int j){return scalar<T>(.07*(i+1)+.02*j,.004*(i-j));};
        auto outside=[](int i,int j){return scalar<T>(.1*(i+1)+.02*j,.003*(i-j));};
        auto rhs=make_local<T>(h,db,[&](int i,int j){
            if(i>=(side=='L'?n:nrhs)||j>=(side=='L'?nrhs:n))return outside(i,j);
            T sum=0;for(int p=0;p<n;++p)sum+=side=='L'?matrix(i,p)*solution(p,j):solution(i,p)*matrix(p,j);return sum;
        });
        const auto expected=make_local<T>(h,db,[&](int i,int j){return i<(side=='L'?n:nrhs)&&j<(side=='L'?nrhs:n)?solution(i,j):outside(i,j);});
        DeviceBuffer<T> b(h,rhs.size()),bref(h,rhs.size());
        auto compare=[&](const std::string& method,int repeat){const auto label=method+" type="+scalar_name<T>()+" grid="+std::to_string(h->nprows_)+"x"+std::to_string(h->npcols_)+" major="+h->major+" side="+side+" trans="+trans+" repeat="+std::to_string(repeat);
            same(h,download(h,ref.ptr,initial.size()),download(h,a.ptr,initial.size()),label+" checked/raw A");const auto got=download(h,b.ptr,rhs.size());same(h,download(h,bref.ptr,rhs.size()),got,label+" checked/raw B");analytic_solve_compare(h,db,side=='L'?n:nrhs,side=='L'?nrhs:n,expected,got,label);};
        for(int mode=0;mode<3;++mode){
            const bool gather=true;Env policy("DDLA_LU_PANEL","gather");
            std::size_t bb=0,hb=0;
            auto size=[&]{if(mode==0)lu_impl::pgesv_workspace_size<T>(h,gather,side,trans,n,nrhs,a.ptr,da,db,&bb,&hb);else if(mode==1)lu_impl::pgesv_bpiv_workspace_size<T>(h,side,trans,n,nrhs,a.ptr,da,db,&bb,&hb);else lu_impl::pgesv_nopiv_workspace_size<T>(h,side,trans,n,nrhs,da,db,&bb,&hb);};
            size();workspace_test::Scope qs(true);size();query_clean(h,qs.finish());Guarded work(h,bb,hb);
            for(int repeat=0;repeat<2;++repeat){upload(h,a.ptr,initial);upload(h,ref.ptr,initial);upload(h,b.ptr,rhs);upload(h,bref.ptr,rhs);int got=0,want=0;if(mode==0)pgesv<T>(h,side,trans,n,nrhs,ref.ptr,da,bref.ptr,db);else if(mode==1)pgesv_bpiv<T>(h,side,trans,n,nrhs,ref.ptr,da,bref.ptr,db);else pgesv_nopiv<T>(h,side,trans,n,nrhs,ref.ptr,da,bref.ptr,db);workspace_test::Scope run;if(mode==0)lu_impl::pgesv_run<B,T>(h,gather,side,trans,n,nrhs,a.ptr,da,b.ptr,db,got,work.backend(),bb,work.host(),hb);else if(mode==1)lu_impl::pgesv_bpiv_run<B,T>(h,side,trans,n,nrhs,a.ptr,da,b.ptr,db,got,work.backend(),bb,work.host(),hb);else lu_impl::pgesv_nopiv_run<B,T>(h,side,trans,n,nrhs,a.ptr,da,b.ptr,db,got,work.backend(),bb,work.host(),hb);run_clean(h,run.finish());compare(mode==0?"pgesv":mode==1?"pgesv_bpiv":"pgesv_nopiv",repeat);require_close(h,"LU solve info",got==want?0:1,0);work.verify();}
        }
        if(h->nprows_==h->npcols_)for(char uplo:{'L','U'}){
            std::size_t bb=0,hb=0;cholesky_impl::pposv_workspace_size<T>(h,side,uplo,n,nrhs,a.ptr,da,db,&bb,&hb);Guarded work(h,bb,hb);
            workspace_test::Scope qs(true);cholesky_impl::pposv_workspace_size<T>(h,side,uplo,n,nrhs,a.ptr,da,db,&bb,&hb);query_clean(h,qs.finish());
            for(int repeat=0;repeat<2;++repeat){upload(h,a.ptr,initial);upload(h,ref.ptr,initial);upload(h,b.ptr,rhs);upload(h,bref.ptr,rhs);int got=0,want=0;pposv<T>(h,side,uplo,trans,n,nrhs,ref.ptr,1,1,da,bref.ptr,1,1,db,want);workspace_test::Scope run;cholesky_impl::pposv_run<B,T>(h,side,uplo,trans,n,nrhs,a.ptr,1,1,da,b.ptr,1,1,db,got,work.backend(),bb,work.host(),hb);run_clean(h,run.finish());compare(std::string("pposv uplo=")+uplo,repeat);require_close(h,"Cholesky solve info",got==want?0:1,0);work.verify();}
        }
    }
    for(char uplo:{'L','U'}){
        if(h->nprows_==h->npcols_){
        std::size_t bb=0,hb=0;cholesky_impl::ppotrf_bottom_right_workspace_size<T>(h,uplo,n,a.ptr,da,&bb,&hb);Guarded work(h,bb,hb);workspace_test::Scope qs(true);cholesky_impl::ppotrf_bottom_right_workspace_size<T>(h,uplo,n,a.ptr,da,&bb,&hb);query_clean(h,qs.finish());
        for(int repeat=0;repeat<2;++repeat){upload(h,a.ptr,initial);upload(h,ref.ptr,initial);int got=0,want=0;ppotrf_bottom_right<T>(h,uplo,n,ref.ptr,da,want);workspace_test::Scope run;cholesky_impl::ppotrf_bottom_right_run<B,T>(h,uplo,n,a.ptr,da,got,work.backend(),bb,work.host(),hb);run_clean(h,run.finish());same(h,download(h,ref.ptr,initial.size()),download(h,a.ptr,initial.size()));require_close(h,"bottom-right info",got==want?0:1,0);work.verify();}
        }
        for(bool inplace:{false,true}){
            Env policy("DDLA_QR_PANEL",inplace?"inplace":"gather");const int ntau=std::max(1,num_loc(n,nb,h->mypcol_,da[DDLA_CSRC_],h->npcols_));std::vector<T> tau(ntau,scalar<T>(-19)),tauref=tau;
            qr_impl::pgeqrf_check<B>(h,uplo,n,n,da,tau.data());workspace_test::Scope qs(true);std::size_t bb=0,hb=0;qr_impl::pgeqrf_workspace_size<T>(h,uplo,inplace,n,n,a.ptr,da,a.ptr,&bb,&hb);query_clean(h,qs.finish());Guarded work(h,bb,hb);
            for(int repeat=0;repeat<2;++repeat){upload(h,a.ptr,initial);upload(h,ref.ptr,initial);std::fill(tau.begin(),tau.end(),scalar<T>(-19));std::fill(tauref.begin(),tauref.end(),scalar<T>(-19));int info=0;if(uplo=='L')pgeqlf<T>(h,n,n,ref.ptr,da,tauref.data(),info);else pgeqrf<T>(h,n,n,ref.ptr,da,tauref.data(),info);workspace_test::Scope run;qr_impl::pgeqrf_run<B,T>(h,uplo,inplace,n,n,a.ptr,da,tau.data(),work.backend(),bb,work.host(),hb);run_clean(h,run.finish());same(h,download(h,ref.ptr,initial.size()),download(h,a.ptr,initial.size()));same(h,tauref,tau);work.verify();}
        }
    }
    // RNG needs a prepared generator, with no empty workspace component.
    constexpr int64_t count=128;DeviceBuffer<T> random(h,count);random_impl::PreparedRng rng(h,731);
    random_impl::random_generate_check<B,T>(h,random.ptr,count);
    for(int repeat=0;repeat<2;++repeat){workspace_test::Scope run;random_impl::random_generate_run<B,T>(h,rng,random.ptr,count);run_clean(h,run.finish());auto values=download(h,random.ptr,count);double bad=0;for(auto x:values)if(!std::isfinite(double(std::abs(x))))bad=1;require_close(h,"prepared random generator produces finite values",bad,0);}
    rng.reset();
}
template<DdlaBackend B,class T>void tiny_case(DdlaHandle_t h)
{
    int d[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(d,h,1,1,3,3,h->nprows_-1,h->npcols_-1));
    const auto input=make_local<T>(h,d,[](int,int){return T(4);});DeviceBuffer<T> a(h,input.size());upload(h,a.ptr,input);
    std::vector<int> piv(num_loc(1,3,h->myprow_,d[DDLA_RSRC_],h->nprows_),-1);
    for(bool gather:{false,true}){
        std::size_t bb=0,hb=0;lu_impl::pgetrf_workspace_size<T>(h,gather,1,1,a.ptr,1,1,d,&bb,&hb);Guarded w(h,bb,hb);
        for(int pass=0;pass<2;++pass){upload(h,a.ptr,input);int info=-1;workspace_test::Scope run;lu_impl::pgetrf_run<B,T>(h,gather,1,1,a.ptr,1,1,d,piv.data(),info,w.backend(),bb,w.host(),hb);run_clean(h,run.finish());require_close(h,"zero-local-extent LU info",info,0);same(h,input,download(h,a.ptr,input.size()));w.verify();}
    }
    for(bool inplace:{false,true}){
        std::size_t bb=0,hb=0;qr_impl::pgeqrf_workspace_size<T>(h,'L',inplace,1,1,a.ptr,d,a.ptr,&bb,&hb);Guarded w(h,bb,hb);
        std::vector<T> tau(num_loc(1,3,h->mypcol_,d[DDLA_CSRC_],h->npcols_),T(-1));
        for(int pass=0;pass<2;++pass){upload(h,a.ptr,input);workspace_test::Scope run;qr_impl::pgeqrf_run<B,T>(h,'L',inplace,1,1,a.ptr,d,tau.data(),w.backend(),bb,w.host(),hb);run_clean(h,run.finish());same(h,input,download(h,a.ptr,input.size()));w.verify();}
    }
}
template<DdlaBackend B>void run(MPI_Comm comm,int pr,int pc,char major){DdlaHandle_t h=nullptr;ddlaInit(h,B);ddlaSet(h,comm,pr,pc,major);cases<B,float>(h);cases<B,double>(h);cases<B,std::complex<float>>(h);cases<B,std::complex<double>>(h);tiny_case<B,float>(h);tiny_case<B,double>(h);tiny_case<B,std::complex<float>>(h);tiny_case<B,std::complex<double>>(h);if(h->myid==0)std::cout<<"workspace factorizations PASS backend="<<int(B)<<" grid="<<pr<<"x"<<pc<<" major="<<major<<std::endl;ddlaDestroy(h);}
}
int main(int argc,char** argv){ddla_test_mpi_init(&argc,&argv);int size;MPI_Comm_size(MPI_COMM_WORLD,&size);TestOptions opts;std::string error;if(!parse_options(argc,argv,size,opts,error)){std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);}for(auto grid:opts.grids)for(char major:{'R','C'}){
run<DdlaBackend::CPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#if DDLA_HAS_GPU
run<DdlaBackend::GPU>(MPI_COMM_WORLD,grid.first,grid.second,major);
#endif
}MPI_Finalize();return 0;}
