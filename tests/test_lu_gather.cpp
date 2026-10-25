#include "api_grid_test_common.h"
#include "LU/lu_internal.h"
#include "mpi_datatype.h"
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <fstream>

using namespace api_grid_test;
namespace {
std::ofstream binary_dump;
std::uint64_t fingerprint = 1469598103934665603ULL;
void hash_bytes(const void* p, std::size_t bytes) {
    if(binary_dump.is_open()) binary_dump.write(static_cast<const char*>(p), bytes);
    auto s = static_cast<const unsigned char*>(p);
    for (std::size_t i=0; i<bytes; ++i) { fingerprint ^= s[i]; fingerprint *= 1099511628211ULL; }
}
template<class T> T scalar(double r, double i=0) {
    if constexpr (std::is_same_v<T,float> || std::is_same_v<T,double>) return T(r);
    else return T(r,i);
}
template<class T> double metric(T x) {
    if constexpr (std::is_same_v<T,float> || std::is_same_v<T,double>) return std::abs(x);
    else return std::abs(x.real())+std::abs(x.imag());
}
template<class T> std::vector<int> oracle(std::vector<T>& a, int m, int n, int start, int width, int& info, int swap_start) {
    std::vector<int> piv(m,-1); info=0;
    for(int j=start;j<start+width;++j) {
        int p=j; for(int i=j+1;i<m;++i) if(metric(a[i+j*m])>metric(a[p+j*m])) p=i;
        piv[j]=p+1;
        for(int c=swap_start;c<n;++c) std::swap(a[j+c*m],a[p+c*m]);
        if(metric(a[j+j*m])==0) {info=j+1;break;}
        for(int i=j+1;i<m;++i) a[i+j*m]/=a[j+j*m];
        for(int c=j+1;c<start+width;++c) for(int i=j+1;i<m;++i) a[i+c*m]-=a[i+j*m]*a[j+c*m];
    }
    return piv;
}
// A blocked panel stops before TRSM/GEMM when an inner subpanel is singular:
// the columns after the failing one stay untouched, which the oracle mirrors.
template<class T> std::vector<int> blocked_oracle(std::vector<T>& a,int m,int n,int start,int width,int nb,int& info,int row_shift=0) {
    std::vector<int> piv(m,-1);info=0;
    const int inner=std::min(32,std::max(1,nb/2));
    for(int j=start;j<start+width;) {
        int b=std::min({inner,start+width-j,nb-(row_shift+j)%nb});
        auto sub=oracle(a,m,n,j,b,info,start);
        for(int k=j;k<j+b;++k)piv[k]=sub[k];
        if(info)return piv;
        for(int c=j+b;c<start+width;++c) {
            for(int r=j;r<j+b;++r) for(int k=j;k<r;++k)a[r+c*m]-=a[r+k*m]*a[k+c*m];
            for(int r=j+b;r<m;++r) for(int k=j;k<j+b;++k)a[r+c*m]-=a[r+k*m]*a[k+c*m];
        }
        j+=b;
    }
    return piv;
}
template<DdlaBackend B,class T>
void one(const DdlaHandle_t& h,int m,int n,int nb,int start,int width,int kind,int routine,int rs,int cs,int row_shift=0,int col_shift=0) {
    int desc[DDLA_DLEN_]; DDLA_CHECK(ddlaDescInit(desc,h,m+3+row_shift,n+2+col_shift,nb,nb,rs,cs));
    if(row_shift || col_shift) desc[DDLA_LLD_]+=3;
    auto value=[=](int i,int j)->T {
        i-=row_shift;j-=col_shift;
        if(i<0 || j<0 || i>=m || j>=n)
            return scalar<T>(-731.25+i*0.125+j*0.03125,211.5+i*0.0625-j*0.015625);
        if(kind==4) {
            using Real=std::conditional_t<std::is_same_v<T,float> || std::is_same_v<T,std::complex<float>>,float,double>;
            return scalar<T>(i==j ? (i==start ? double(std::numeric_limits<Real>::min())*16 : 1.0) : 0.0);
        }
        if(kind==5 && j==start) return T(0);
        if(kind==3 && j==start+width/2) return T(0);
        if(kind==2 && j==start) {
            if(i==start) return scalar<T>(4,4);
            if(i==m-1) return scalar<T>((std::is_same_v<T,float> || std::is_same_v<T,double>)?4:8);
            return scalar<T>(0.25);
        }
        int source=kind==1 ? (i+nb+1)%m : i;
        std::uint64_t x=std::uint64_t(source+1)*0x9e3779b97f4a7c15ULL+std::uint64_t(j+1)*0xbf58476d1ce4e5b9ULL;
        x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL; x=(x^(x>>27))*0x94d049bb133111ebULL; x^=x>>31;
        double r=double(x&65535)/65536.0-0.5, im=double((x>>16)&65535)/65536.0-0.5;
        if(kind==1 && source==j) r+=8;
        return scalar<T>(r,im);
    };
    std::vector<T> original(std::size_t(m)*n);
    for(int c=0;c<n;++c) for(int r=0;r<m;++r) original[r+c*m]=value(r+row_shift,c+col_shift);
    auto reference=original;
    int refinfo=0;
    std::vector<int> refpiv;
    if(routine==0)refpiv=blocked_oracle(reference,m,n,start,width,nb,refinfo,row_shift);
    if(refinfo>0)refinfo-=start;
    auto host=make_local<T>(h,desc,value);
    DeviceBuffer<T> d(h,host.size()); upload(h,d.ptr,host);
    std::vector<int> piv(std::max(1,ddla_test::m_loc(h,desc)),-1);
    int info=-1;
    if(routine==0) ddla::pgetf2<B>(h,m-start,n-start,width,d.ptr,row_shift+start+1,col_shift+start+1,desc,piv.data(),info);
    else ddla::pgetrf<B>(h,m,n,d.ptr,row_shift+1,col_shift+1,desc,piv.data(),info);
    check_ddla_sync(h);
    auto out=download(h,d.ptr,host.size());
    hash_bytes(out.data(),out.size()*sizeof(T)); hash_bytes(piv.data(),piv.size()*sizeof(int));hash_bytes(&info,sizeof(info));
    const double tol=(std::is_same_v<T,float> || std::is_same_v<T,std::complex<float>>) ? 2e-4 : 2e-11;
    std::string label="LU mode="+std::string(std::getenv("DDLA_LU_PANEL")?std::getenv("DDLA_LU_PANEL"):"default")+
        " routine="+std::to_string(routine)+" kind="+std::to_string(kind)+" start="+std::to_string(start)+
        " m="+std::to_string(m)+" n="+std::to_string(n)+" bytes="+std::to_string(sizeof(T))+
        " ia="+std::to_string(row_shift+start+1)+" ja="+std::to_string(col_shift+start+1);
    double err=0,scale=1,pad=0;
    std::vector<T> dense(std::size_t(m)*n,T(0));
    std::vector<int> global_piv(m,-1);
    for(int c=0;c<ddla_test::n_loc(h,desc);++c) for(int r=0;r<desc[DDLA_LLD_];++r) {
        std::size_t k=r+std::size_t(c)*desc[DDLA_LLD_];
        int gr=indx_l2g_r(desc,h,r)-row_shift,gc=indx_l2g_c(desc,h,c)-col_shift;
        if(gr<0 || gc<0 || gr>=m || gc>=n) { if(std::memcmp(&out[k],&host[k],sizeof(T))) pad=1; }
        else {
            if(routine==0 && (gr<start || gc<start) && std::memcmp(&out[k],&host[k],sizeof(T)))pad=1;
            dense[gr+gc*m]=out[k];
            if(!std::isfinite(double(std::abs(out[k])))) err=std::numeric_limits<double>::infinity();
            if(routine==0) {err=std::max(err,double(std::abs(out[k]-reference[gr+gc*m]))); scale=std::max(scale,double(std::abs(reference[gr+gc*m])));}
        }
    }
    for(int i=0;i<m;++i) {int r=indx_g2l_r(desc,h,i+row_shift);if(r>=0) global_piv[i]=piv[r];}
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,global_piv.data(),m,MPI_INT,MPI_MAX,ddlaGetCommunicator(h)));
    double pivot_layout=0;
    for(int i=0;i<m;++i) {
        const int r=indx_g2l_r(desc,h,i+row_shift);
        if(r>=0) pivot_layout=std::max(pivot_layout,double(std::abs(piv[r]-global_piv[i])));
    }
    require_close(h,label+" pivots on every process column",pivot_layout,0);
    require_close(h,label+" padding",pad,0);
    double pivot_padding=0;
    for(int r=0;r<ddla_test::m_loc(h,desc);++r) {
        const int gr=indx_l2g_r(desc,h,r)-row_shift;
        const int begin=routine==1?0:start,end=routine==1?std::min(m,n):start+width;
        if(gr<begin || gr>=end || (routine==1 && info>0 && gr>=info))
            pivot_padding=std::max(pivot_padding,double(std::abs(piv[r]+1)));
    }
    require_close(h,label+" pivot sentinels",pivot_padding,0);
    if(routine==0) {
        require_close(h,label+" info",std::abs(info-refinfo),0);
        require_close(h,label+" serial matrix",err/scale,tol);
        for(int j=start;j<start+width;++j) require_close(h,label+" pivots",std::abs(global_piv[j]-(refpiv[j]>0?refpiv[j]+row_shift:-1)),0);
    } else if(info==0) {
        MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE,dense.data(),m*n,ddla::detail::mpi_datatype<T>(),MPI_SUM,ddlaGetCommunicator(h)));
        auto perm=original;
        for(int k=0;k<std::min(m,n);++k) {
            int p=global_piv[k]-1-row_shift;
            if(p<k || p>=m) MPI_Abort(ddlaGetCommunicator(h),7);
            for(int c=0;c<n;++c) std::swap(perm[k+c*m],perm[p+c*m]);
        }
        for(int c=0;c<n;++c) for(int r=0;r<m;++r) {
            T product=0;
            for(int k=0;k<std::min(m,n);++k) {
                T l=r==k ? T(1) : (r>k ? dense[r+k*m] : T(0));
                T u=k<=c ? dense[k+c*m] : T(0);
                product+=l*u;
            }
            err=std::max(err,double(std::abs(product-perm[r+c*m])));
            scale=std::max(scale,double(std::abs(perm[r+c*m])));
        }
        require_close(h,label+" PA-LU",err/scale,tol);
        if(m==n && kind==1) {
            std::vector<T> x(n),rhs(n,T(0));
            for(int i=0;i<n;++i)x[i]=scalar<T>(1+0.01*i,0.005*(i%7));
            for(int j=0;j<n;++j)for(int i=0;i<n;++i)rhs[i]+=original[i+j*n]*x[j];
            for(int k=0;k<n;++k)std::swap(rhs[k],rhs[global_piv[k]-1-row_shift]);
            for(int i=0;i<n;++i)for(int k=0;k<i;++k)rhs[i]-=dense[i+k*n]*rhs[k];
            for(int i=n-1;i>=0;--i){for(int k=i+1;k<n;++k)rhs[i]-=dense[i+k*n]*rhs[k];rhs[i]/=dense[i+i*n];}
            double solve_error=0;for(int i=0;i<n;++i)solve_error=std::max(solve_error,double(std::abs(rhs[i]-x[i])));
            require_close(h,label+" embedded solve",solve_error,tol);
        }
    } else {
        require_close(h,label+" singular expected",(kind==3 || kind==5)?0:1,0);
        require_close(h,label+" singular index",std::abs(info-(kind==5 ? start+1 : start+width/2+1)),0);
    }
}
template<DdlaBackend B,class T> void scalar_cases(const DdlaHandle_t& h) {
    int pr,pc;ddlaGetGridDims(h,pr,pc);
    for(int source=0;source<2;++source) {
        int rs=source ? pr-1:0,cs=source ? pc-1:0;
        // Embed the logical matrix at independent global row/column offsets.
        // Unique outside values expose accidental swaps beyond the region.
        for(int nb : {4,9,64}) for(auto shift : {std::pair<int,int>{1,2},{5,9},{9,4},{8,4}}) {
            const int r0=shift.first,c0=shift.second;
            for(auto shape : {std::pair<int,int>{2*nb+7,nb+11},{nb+3,2*nb+13},{nb+5,nb+5}}) {
                const int m=shape.first,n=shape.second;
                for(int start : {0,1}) {
                    const int width=std::min({nb-(c0+start)%nb,m-start,n-start});
                    for(int kind=0;kind<6;++kind)
                        one<B,T>(h,m,n,nb,start,width,kind,0,rs,cs,r0,c0);
                }
                for(int kind : {0,1,3,4,5})
                    one<B,T>(h,m,n,nb,0,std::min(m,n),kind,1,rs,cs,r0,c0);
            }
        }
        // Exercise multiple inner factorizations, a short inner tail, and a
        // nonzero start within one descriptor block. The scalar oracle also
        // checks swaps in columns outside the selected subpanel.
        for(int nb : {9,64,128}) {
            const int m=2*nb+7,n=nb+11,start=nb/4;
            const int width=nb-start-1;
            for(int kind=0;kind<6;++kind)
                one<B,T>(h,m,n,nb,start,width,kind,0,rs,cs);
            for(int kind : {0,1,3,4,5})
                one<B,T>(h,m,n,nb,0,std::min(m,n),kind,1,rs,cs);
        }
        for(auto shape : {std::pair<int,int>{31,27},{17,29},{18,29},{39,13},{3,2}}) {
            int m=shape.first,n=shape.second,nb=4;
            if(m==18 && std::getenv("LU_SKIP_RECT_TAIL_TEST")) continue;
            for(int start : {0,4,5}) {
                int width=std::min({nb-start%nb,m-start,n-start});if(width<=0)continue;
                for(int kind=0;kind<6;++kind)
                    one<B,T>(h,m,n,nb,start,width,kind,0,rs,cs);
            }
            for(int kind : {0,1,3,4,5})one<B,T>(h,m,n,nb,0,std::min(m,n),kind,1,rs,cs);
        }
    }
}
template<DdlaBackend B> void selector_checks(const DdlaHandle_t& h) {
    if(std::getenv("LU_SKIP_SELECTOR_TEST")) return;
    const char* v=std::getenv("DDLA_LU_PANEL");bool had=v;std::string saved=v?v:"";
    int rank,size;MPI_Comm_rank(ddlaGetCommunicator(h),&rank);MPI_Comm_size(ddlaGetCommunicator(h),&size);
    int desc[DDLA_DLEN_];DDLA_CHECK(ddlaDescInit(desc,h,4,4,2,2,0,0));
    auto initial=make_local<double>(h,desc,[](int i,int j){return i==j?1.0:0.0;});
    DeviceBuffer<double> a(h,initial.size());std::vector<int> piv(std::max(1,ddla_test::m_loc(h,desc)),-1);
    auto call=[&](int routine,int m,int n,int width,int& info,int ia=1,int ja=1) {
        if(routine==0)ddla::pgetf2<B>(h,m,n,width,a.ptr,ia,ja,desc,piv.data(),info);
        else ddla::pgetrf<B>(h,m,n,a.ptr,ia,ja,desc,piv.data(),info);
    };
    upload(h,a.ptr,initial);
    for(int routine=0;routine<2;++routine) {
        if(size>1) {
            setenv("DDLA_LU_PANEL",rank==0?"gather":"inplace",1);
            bool caught=false;int info=-1;
            try{call(routine,4,4,1,info);}catch(const std::runtime_error&){caught=true;}
            require_close(h,"LU selector disagreement rejected",caught?0:1,0);
        }
        setenv("DDLA_LU_PANEL","inplace",1);
        for(auto offset : {std::pair<int,int>{0,1},{1,0},{2,1},{1,2}}) {
            bool caught=false;int info=-1;
            try{call(routine,4,4,1,info,rank==0?offset.first:1,rank==0?offset.second:1);}
            catch(const std::invalid_argument&){caught=true;}
            require_close(h,"LU invalid rank-local IA/JA rejected collectively",caught?0:1,0);
        }
        for(int storage=0;storage<2;++storage) {
            double* ap=rank==0 && storage==0?nullptr:a.ptr;
            int* pp=rank==0 && storage==1?nullptr:piv.data();
            bool caught=false;int info=-1;
            try {
                if(routine==0)ddla::pgetf2<B,double>(h,4,4,1,ap,1,1,desc,pp,info);
                else ddla::pgetrf<B,double>(h,4,4,ap,1,1,desc,pp,info);
            } catch(const std::invalid_argument&){caught=true;}
            require_close(h,"LU missing rank-local storage rejected collectively",caught?0:1,0);
        }
        if(routine!=1) {
            bool caught=false;int info=-1;
            try{call(routine,4,4,rank==0?-1:1,info);}catch(const std::invalid_argument&){caught=true;}
            require_close(h,"LU negative panel width rejected collectively",caught?0:1,0);
        }
        setenv("DDLA_LU_PANEL","invalid",1);upload(h,a.ptr,initial);int info=-1;call(routine,4,4,1,info);
        require_close(h,"LU invalid selector defaults to inplace",std::abs(info),0);
        setenv("DDLA_LU_PANEL","gather",1);info=-1;call(routine,0,0,0,info);
        require_close(h,"LU empty operation",std::abs(info),0);
        if(routine==0)ddla::pgetf2<B,double>(h,0,0,0,nullptr,5,5,desc,nullptr,info);
        else ddla::pgetrf<B,double>(h,0,0,nullptr,5,5,desc,nullptr,info);
        require_close(h,"LU empty edge offsets and null storage",std::abs(info),0);
    }
    if(had)setenv("DDLA_LU_PANEL",saved.c_str(),1);else unsetenv("DDLA_LU_PANEL");
}
template<DdlaBackend B> void run(const DdlaHandle_t& h) {
    selector_checks<B>(h);
    scalar_cases<B,float>(h); scalar_cases<B,double>(h);
    scalar_cases<B,std::complex<float>>(h); scalar_cases<B,std::complex<double>>(h);
}
}
int main(int argc,char**argv) {
    MPI_Init(&argc,&argv);int rank,size;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm work_comm=MPI_COMM_WORLD;
    const bool subcomm=std::getenv("LU_TEST_SUBCOMM") && size>=4;
    if(subcomm) {
        bool active=rank%2==0;
        MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD,active?0:1,rank,&work_comm));
        if(!active) {MPI_Comm_free(&work_comm);MPI_Barrier(MPI_COMM_WORLD);MPI_Finalize();return 0;}
        MPI_Comm_rank(work_comm,&rank);MPI_Comm_size(work_comm,&size);
    }
    if(const char* prefix=std::getenv("LU_TEST_DUMP")) binary_dump.open(std::string(prefix)+"."+std::to_string(rank),std::ios::binary);
    TestOptions opts;std::string error;
    if(!parse_options(argc,argv,size,opts,error)){if(rank==0)std::cerr<<error<<std::endl;MPI_Abort(MPI_COMM_WORLD,2);}
    for(auto grid:opts.grids) {
#if DDLA_HAS_CPU
        {DdlaHandle_t h=nullptr;ddlaInit(h,DdlaBackend::CPU);ddlaSet(h,work_comm,grid.first,grid.second,std::getenv("LU_TEST_COLUMN_MAJOR")?'C':'R');run<DdlaBackend::CPU>(h);ddlaDestroy(h);}
#endif
#if DDLA_HAS_GPU
        {DdlaHandle_t h=nullptr;ddlaInit(h,DdlaBackend::GPU);ddlaSet(h,work_comm,grid.first,grid.second,std::getenv("LU_TEST_COLUMN_MAJOR")?'C':'R');run<DdlaBackend::GPU>(h);ddlaDestroy(h);}
#endif
    }
    unsigned long long local=fingerprint,global=0;MPI_Reduce(&local,&global,1,MPI_UNSIGNED_LONG_LONG,MPI_BXOR,0,work_comm);
    if(rank==0)std::cout<<"test_lu_gather PASS fingerprint="<<std::hex<<global<<std::endl;
    if(subcomm){MPI_Comm_free(&work_comm);MPI_Barrier(MPI_COMM_WORLD);}
    MPI_Finalize();
}
