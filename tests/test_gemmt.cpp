#include "mpi_test_init.h"
#include "matrix_test_common.h"
#include <ddla/ddla_vector.h>
#include <cctype>
#include <string>
#include <utility>
#include <array>
#include <cstdlib>
#include "PBLAS/pgemmt_impl.h"

namespace {
using Wide = std::complex<long double>;

template <typename T> Wide extended(T x) {
    if constexpr (std::is_arithmetic_v<T>) return {x,0};
    else return {x.real(),x.imag()};
}
template <typename T> T conjugate(T x) {
    if constexpr (std::is_arithmetic_v<T>) return x;
    else return std::conj(x);
}
// A = X M and B = M^-1 D X^H (or X^T), where each pair of
// columns uses M = [[1, 1/2], [0, 1]]. The full product is structured,
// but a one-column k-panel generally is not. Dyadic values make the
// cancellation exact in the input scalar type, including odd k tails.
template <typename T> T effective_a(int i,int l) {
    return value<T>(i,l,1)+(l%2 ? T(.5)*value<T>(i,l-1,1) : T(0));
}
template <typename T> T weighted_row(char symmetry,int l,int j) {
    const T x=value<T>(j,l,1);
    const T weight=T((l%2 ? -1 : 1)*(l%5+1)/8.);
    return weight*(symmetry=='H' ? conjugate(x) : x);
}
template <typename T> T effective_b(char symmetry,int l,int j,int k) {
    if(symmetry=='G') return value<T>(l,j,2);
    const T x=weighted_row<T>(symmetry,l,j);
    return l%2==0 && l+1<k ? x-T(.5)*weighted_row<T>(symmetry,l+1,j) : x;
}
template <typename T> T stored_a(char op,int i,int j) {
    const T x=op=='N' ? effective_a<T>(i,j) : effective_a<T>(j,i);
    return op=='C' ? conjugate(x) : x;
}
template <typename T> T stored_b(char symmetry,char op,int i,int j,int k) {
    const T x=op=='N' ? effective_b<T>(symmetry,i,j,k) : effective_b<T>(symmetry,j,i,k);
    return op=='C' ? conjugate(x) : x;
}
template <typename T> T initial_c(char symmetry,int i,int j) {
    if(symmetry=='G') return value<T>(i,j,3);
    if(i==j && symmetry=='H') return scalar<T>(wide(value<T>(i,j,3)).real());
    const T x=value<T>(std::min(i,j),std::max(i,j),3);
    return symmetry=='H' && i>j ? conjugate(x) : x;
}
template <typename T> T poisoned_c(char symmetry,char uplo,int i,int j,const T& beta) {
    const double nan=std::numeric_limits<double>::quiet_NaN();
    if(beta==T(0) || (uplo=='U' ? i>j : i<j)) return scalar<T>(nan,nan);
    return initial_c<T>(symmetry,i,j);
}
template <typename T> std::pair<Wide,long double> reference(
    char symmetry,int i,int j,int k,const T& alpha,const T& beta) {
    Wide sum=0;long double scale=0;
    if(alpha!=T(0)) for(int l=0;l<k;++l) {
        const auto a=extended(effective_a<T>(i,l)),b=extended(effective_b<T>(symmetry,l,j,k));
        sum+=a*b;scale+=std::abs(a)*std::abs(b);
    }
    sum*=extended(alpha);scale*=std::abs(extended(alpha));
    if(beta!=T(0)) {
        const auto old=extended(beta)*extended(initial_c<T>(symmetry,i,j));
        sum+=old;scale+=std::abs(old);
    }
    return {sum,std::max(1.L,scale)};
}

template <typename T> void verify_value(char symmetry,int i,int j,int k,
    const T& alpha,const T& beta,const T& actual,long double& error,int& bad) {
    const auto ref=reference<T>(symmetry,i,j,k,alpha,beta);
    const long double residual=std::abs(extended(actual)-ref.first)/ref.second;
    if(!std::isfinite(residual)) bad=1;
    else error=std::max(error,residual);
}

template <typename T> void finish_check(DdlaHandle_t h,bool distributed,char symmetry,
    char uplo,char ta,char tb,int n,int k,int mode,long double error,int bad,
    int& cases,long double& worst) {
    long double global=0;int any=0;
    MPI_Allreduce(&error,&global,1,MPI_LONG_DOUBLE,MPI_MAX,ddlaGetCommunicator(h));
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
    using R=typename ddla::real_type<T>::type;
    if(any || global>64*std::numeric_limits<R>::epsilon()) {
        if(ddlaGetRank(h)==0) std::cerr<<"GEMMT_FAIL distributed="<<distributed
            <<" symmetry="<<symmetry<<" uplo="<<uplo<<" flags="<<ta<<tb
            <<" n="<<n<<" k="<<k<<" mode="<<mode<<" type_bytes="<<sizeof(T)
            <<" is_complex="<<!std::is_arithmetic_v<T><<" real_bytes="<<sizeof(R)
            <<" normalized_error="<<double(global)<<" untouched_storage="<<any<<std::endl;
        throw std::runtime_error("triangular product scalar oracle or untouched-storage check failed");
    }
    ++cases;worst=std::max(worst,global);
}

template <typename T> std::pair<T,T> coefficients(int mode) {
    const T alpha=mode>=2 && mode<=4 ? T(0) : scalar<T>(-.75,.25);
    const T beta=mode==1 || mode==3 ? T(0) : mode==4 ? T(1) : scalar<T>(.5,-.25);
    return {alpha,beta};
}

template <DdlaBackend Backend,typename T>
void local_case(DdlaHandle_t h,char symmetry,char uplo,char ta,char tb,int n,int k,
                int mode,int& cases,long double& worst) {
    const auto coeff=coefficients<T>(mode);
    const T alpha=coeff.first,beta=coeff.second;
    const int ar=ta=='N' ? n : k,ac=ta=='N' ? k : n;
    const int br=tb=='N' ? k : n,bc=tb=='N' ? n : k;
    const int lda=std::max(1,ar)+3,ldb=std::max(1,br)+4,ldc=std::max(1,n)+5;
    const T guard=scalar<T>(-987.25,654.5);
    std::vector<T> ah(std::size_t(lda)*(ac+2),guard),bh(std::size_t(ldb)*(bc+2),guard);
    std::vector<T> ch(std::size_t(ldc)*(n+2),guard);
    for(int j=0;j<ac;++j) for(int i=0;i<ar;++i) ah[i+std::size_t(j)*lda]=stored_a<T>(ta,i,j);
    for(int j=0;j<bc;++j) for(int i=0;i<br;++i) bh[i+std::size_t(j)*ldb]=stored_b<T>(symmetry,tb,i,j,k);
    for(int j=0;j<n;++j) for(int i=0;i<n;++i) ch[i+std::size_t(j)*ldc]=(mode==6 && (uplo=='U'?i>j:i<j)) ? guard : poisoned_c<T>(symmetry,uplo,i,j,beta);
    ddla::vector<T> a(h,ah.size()),b(h,bh.size()),c(h,ch.size());
    DDLA_CHECK(ddlaMemcpy(h,a.data(),ah.data(),ah.size()*sizeof(T)));
    DDLA_CHECK(ddlaMemcpy(h,b.data(),bh.data(),bh.size()*sizeof(T)));
    DDLA_CHECK(ddlaMemcpy(h,c.data(),ch.data(),ch.size()*sizeof(T)));
    const bool product=n>0 && k>0 && alpha!=T(0);
    const auto flag=[&](char x) { return mode==6 ? char(std::tolower(static_cast<unsigned char>(x))) : x; };
    gemmt<Backend,T>(h,flag(uplo),flag(ta),flag(tb),n,k,
        alpha,product ? a.data() : nullptr,lda,product ? b.data() : nullptr,ldb,
        beta,n ? c.data() : nullptr,ldc);
    std::vector<T> out(ch.size()),aa(ah.size()),bb(bh.size());
    DDLA_CHECK(ddlaMemcpy(h,out.data(),c.data(),out.size()*sizeof(T)));
    DDLA_CHECK(ddlaMemcpy(h,aa.data(),a.data(),aa.size()*sizeof(T)));
    DDLA_CHECK(ddlaMemcpy(h,bb.data(),b.data(),bb.size()*sizeof(T)));
    DDLA_CHECK(ddlaSynchronize(h));
    int bad=std::memcmp(aa.data(),ah.data(),aa.size()*sizeof(T)) ||
            std::memcmp(bb.data(),bh.data(),bb.size()*sizeof(T));
    unsigned reported=0;
    const auto diagnose=[&](unsigned bit,const char* category,std::size_t index,Wide actual,Wide expected) {
        if(reported&bit) return;
        reported|=bit;
        std::cerr<<"GEMMT_LOCAL_DETAIL rank="<<ddlaGetRank(h)<<" category="<<category
            <<" linear_index="<<index<<" actual="<<actual<<" expected="<<expected
            <<" is_complex="<<!std::is_arithmetic_v<T>
            <<" real_bytes="<<sizeof(typename ddla::real_type<T>::type)
            <<" flags="<<ta<<tb<<" symmetry="<<symmetry<<" uplo="<<uplo
            <<" lda="<<lda<<" ldb="<<ldb<<" ldc="<<ldc<<std::endl;
    };
    if(bad) {
        for(std::size_t p=0;p<aa.size();++p) if(std::memcmp(&aa[p],&ah[p],sizeof(T))) {
            diagnose(1,"input_A_mutated",p,extended(aa[p]),extended(ah[p]));break;
        }
        for(std::size_t p=0;p<bb.size();++p) if(std::memcmp(&bb[p],&bh[p],sizeof(T))) {
            diagnose(2,"input_B_mutated",p,extended(bb[p]),extended(bh[p]));break;
        }
    }
    long double error=0;
    for(std::size_t p=0;p<out.size();++p) {
        const int i=p%ldc,j=p/ldc;
        if(i<n && j<n && (uplo=='U' ? i<=j : i>=j)) {
            verify_value<T>(symmetry,i,j,k,alpha,beta,out[p],error,bad);
            const auto actual=extended(out[p]);
            if(!std::isfinite(actual.real()) || !std::isfinite(actual.imag()))
                diagnose(4,"nonfinite_output",p,actual,reference<T>(symmetry,i,j,k,alpha,beta).first);
        } else if(std::memcmp(&out[p],&ch[p],sizeof(T))) {
            bad=1;diagnose(32,"output_guard_changed",p,extended(out[p]),extended(ch[p]));
        }
    }
    finish_check<T>(h,false,symmetry,uplo,ta,tb,n,k,mode,error,bad,cases,worst);
}

template <DdlaBackend Backend,typename T>
void distributed_case(DdlaHandle_t h,char symmetry,char uplo,char ta,char tb,int n,int k,
                      int nb,int mode,bool shifted,int& cases,long double& worst) {
    const auto coeff=coefficients<T>(mode);
    const T alpha=coeff.first,beta=coeff.second;
    int pr,pc;ddlaGetGridDims(h,pr,pc);
    const int ar=ta=='N' ? n : k,ac=ta=='N' ? k : n;
    const int br=tb=='N' ? k : n,bc=tb=='N' ? n : k;
    Matrix<T> a(h,ar,ac,nb,nb,shifted ? pr-1 : 0,0,1,false);
    Matrix<T> b(h,br,bc,nb,nb,0,shifted ? pc-1 : 0,2,false);
    Matrix<T> c(h,n,n,nb,nb,shifted ? pr/2 : 0,shifted ? pc/2 : 0,3,false);
    const auto fill=[&](Matrix<T>& x,int rows,int cols,auto value_at) {
        for(int j=0;j<x.nc;++j) for(int i=0;i<x.nr;++i) {
            const int gi=indxl2g(i,nb,x.r,x.d[DDLA_RSRC_],pr);
            const int gj=indxl2g(j,nb,x.c,x.d[DDLA_CSRC_],pc);
            if(gi<rows && gj<cols) x.host[i+std::size_t(j)*x.d[DDLA_LLD_]]=value_at(gi,gj);
        }
        x.reset();
    };
    fill(a,ar,ac,[&](int i,int j){return stored_a<T>(ta,i,j);});
    fill(b,br,bc,[&](int i,int j){return stored_b<T>(symmetry,tb,i,j,k);});
    fill(c,n,n,[&](int i,int j){return mode==6 && (uplo=='U'?i>j:i<j) ? scalar<T>(-987.25,654.5) : poisoned_c<T>(symmetry,uplo,i,j,beta);});
    const auto owns=[&](const Matrix<T>& x,int rows,int cols) {
        return num_loc(rows,nb,x.r,x.d[DDLA_RSRC_],pr)>0 &&
               num_loc(cols,nb,x.c,x.d[DDLA_CSRC_],pc)>0;
    };
    const bool product=n>0 && k>0 && alpha!=T(0);
    const auto flag=[&](char x) { return mode==6 ? char(std::tolower(static_cast<unsigned char>(x))) : x; };
    pgemmt<Backend,T>(h,flag(uplo),flag(ta),flag(tb),n,k,
        alpha,product && owns(a,ar,ac) ? a.data : nullptr,a.d,
        product && owns(b,br,bc) ? b.data : nullptr,b.d,beta,owns(c,n,n) ? c.data : nullptr,c.d);
    const auto out=c.download(),aa=a.download(),bb=b.download();
    int bad=std::memcmp(aa.data(),a.host.data(),aa.size()*sizeof(T)) ||
            std::memcmp(bb.data(),b.host.data(),bb.size()*sizeof(T));
    long double error=0;
    for(std::size_t p=0;p<out.size();++p) {
        const int i=p%c.d[DDLA_LLD_],j=p/c.d[DDLA_LLD_];
        const int gi=indxl2g(i,nb,c.r,c.d[DDLA_RSRC_],pr);
        const int gj=indxl2g(j,nb,c.c,c.d[DDLA_CSRC_],pc);
        if(i<c.nr && j<c.nc && gi<n && gj<n && (uplo=='U' ? gi<=gj : gi>=gj)) {
            verify_value<T>(symmetry,gi,gj,k,alpha,beta,out[p],error,bad);
        } else if(std::memcmp(&out[p],&c.host[p],sizeof(T))) bad=1;
    }
    finish_check<T>(h,true,symmetry,uplo,ta,tb,n,k,mode,error,bad,cases,worst);
}

// A rank-two X makes X D X^(T/H) admit a full O(n^2+k) independent
// oracle. Pairwise column mixing makes partial k panels unstructured, so the
// large batched path cannot accidentally rely on rank-k input identities.
template <DdlaBackend Backend,typename T>
void analytic_large(DdlaHandle_t h,char symmetry,char uplo,int& cases,long double& worst,
    int n=1057,int k=1031,int nb=128,bool borrowed=false)
{
    int pr,pc;ddlaGetGridDims(h,pr,pc);
    const char ta=uplo=='U' ? 'N' : 'C',tb=ta;
    const auto coeff=coefficients<T>(uplo=='U' ? 1 : 0);
    const T alpha=coeff.first,beta=coeff.second;
    const auto feature=[](int i,int axis) {
        return axis==0 ? scalar<T>((i%7-3)/8.,(i%5-2)/8.) : scalar<T>((i%11-5)/16.,(i%3-1)/8.);
    };
    const auto x=[&](int i,int l){return feature(i,0)*feature(l+5,0)+feature(i,1)*feature(l+5,1);};
    const auto row=[&](int l,int j){const T v=x(j+(symmetry=='G'?3:0),l);return T((l%3+1)/4.)*(symmetry=='H' ? conjugate(v) : v);};
    const auto av=[&](int i,int l){return x(i,l)+(l%2 ? T(.5)*x(i,l-1) : T(0));};
    const auto bv=[&](int l,int j){return row(l,j)-(l%2==0 && l+1<k ? T(.5)*row(l+1,j) : T(0));};
    Matrix<T> a(h,ta=='N'?n:k,ta=='N'?k:n,nb,nb,pr-1,0,1,false);
    Matrix<T> b(h,tb=='N'?k:n,tb=='N'?n:k,nb,nb,0,pc-1,2,false);
    Matrix<T> c(h,n,n,nb,nb,0,0,3,false);
    const auto fill=[&](Matrix<T>& z,int rows,int cols,auto value_at) {
        for(int j=0;j<z.nc;++j)for(int i=0;i<z.nr;++i) {
            const int gi=indxl2g(i,nb,z.r,z.d[DDLA_RSRC_],pr),gj=indxl2g(j,nb,z.c,z.d[DDLA_CSRC_],pc);
            if(gi<rows&&gj<cols)z.host[i+std::size_t(j)*z.d[DDLA_LLD_]]=value_at(gi,gj);
        }
        z.reset();
    };
    fill(a,ta=='N'?n:k,ta=='N'?k:n,[&](int i,int j){const T v=ta=='N'?av(i,j):av(j,i);return ta=='C'?conjugate(v):v;});
    fill(b,tb=='N'?k:n,tb=='N'?n:k,[&](int i,int j){const T v=tb=='N'?bv(i,j):bv(j,i);return tb=='C'?conjugate(v):v;});
    fill(c,n,n,[&](int i,int j){return poisoned_c<T>(symmetry,uplo,i,j,beta);});
    Wide sigma[2][2]={};long double bound[2][2]={};
    for(int l=0;l<k;++l)for(int i=0;i<2;++i)for(int j=0;j<2;++j) {
        const T v=feature(l+5,j);
        const Wide term=extended(T((l%3+1)/4.))*extended(feature(l+5,i))*extended(symmetry=='H'?conjugate(v):v);
        sigma[i][j]+=term;bound[i][j]+=std::abs(term);
    }
    if (!borrowed) {
        pgemmt<Backend,T>(h,uplo,ta,tb,n,k,alpha,a.data,a.d,b.data,b.d,beta,c.data,c.d);
    } else {
        std::size_t backend_bytes=0,host_bytes=0;
        pgemmt_impl::workspace_size<T>(h,uplo,ta,tb,true,n,k,a.d,b.d,c.d,&backend_bytes,&host_bytes);
        ddla::vector<char> backend_work(h,backend_bytes+512);
        std::vector<char> host_work(host_bytes+512,char(0x5a));
        std::array<char,256> guard;guard.fill(char(0x5a));
        DDLA_CHECK(ddlaMemcpy(h,backend_work.data(),guard.data(),256));
        DDLA_CHECK(ddlaMemcpy(h,backend_work.data()+256+backend_bytes,guard.data(),256));
        for (int repeat=0;repeat<2;++repeat) {
            c.reset();
            pgemmt_impl::run<Backend,T>(h,uplo,ta,tb,true,n,k,alpha,a.data,a.d,b.data,b.d,beta,c.data,c.d,
                backend_work.data()+256,backend_bytes,host_work.data()+256,host_bytes);
        }
        std::array<char,256> first,last;
        DDLA_CHECK(ddlaMemcpy(h,first.data(),backend_work.data(),256));
        DDLA_CHECK(ddlaMemcpy(h,last.data(),backend_work.data()+256+backend_bytes,256));
        DDLA_CHECK(ddlaSynchronize(h));
        const bool guards=first==guard && last==guard &&
            std::all_of(host_work.begin(),host_work.begin()+256,[](char c){return c==char(0x5a);}) &&
            std::all_of(host_work.end()-256,host_work.end(),[](char c){return c==char(0x5a);});
        int any=guards ? 0 : 1;
        MPI_Allreduce(MPI_IN_PLACE,&any,1,MPI_INT,MPI_MAX,ddlaGetCommunicator(h));
        if (any) throw std::runtime_error("PGEMMT reusable workspace bounds");
    }
    const auto out=c.download(),aa=a.download(),bb=b.download();
    int bad=std::memcmp(aa.data(),a.host.data(),aa.size()*sizeof(T)) || std::memcmp(bb.data(),b.host.data(),bb.size()*sizeof(T));
    long double error=0;
    for(std::size_t p=0;p<out.size();++p) {
        const int i=p%c.d[DDLA_LLD_],j=p/c.d[DDLA_LLD_];
        const int gi=indxl2g(i,nb,c.r,0,pr),gj=indxl2g(j,nb,c.c,0,pc);
        if(i<c.nr&&j<c.nc&&gi<n&&gj<n && (uplo=='U' ? gi<=gj : gi>=gj)) {
            Wide sum=0;long double scale=0;
            for(int q=0;q<2;++q)for(int r=0;r<2;++r) {
                const T v=feature(gj+(symmetry=='G'?3:0),r);const Wide xi=extended(feature(gi,q)),xj=extended(symmetry=='H'?conjugate(v):v);
                sum+=xi*sigma[q][r]*xj;scale+=std::abs(xi)*bound[q][r]*std::abs(xj);
            }
            Wide old=beta==T(0)?Wide(0):extended(beta)*extended(initial_c<T>(symmetry,gi,gj));
            const Wide want=extended(alpha)*sum+old;
            const long double e=std::abs(extended(out[p])-want)/std::max(1.L,std::abs(extended(alpha))*scale+std::abs(old));
            if(!std::isfinite(e))bad=1;else error=std::max(error,e);
        } else if(std::memcmp(&out[p],&c.host[p],sizeof(T)))bad=1;
    }
    finish_check<T>(h,true,symmetry,uplo,ta,tb,n,k,1,error,bad,cases,worst);
}

template <DdlaBackend Backend,typename T>
void scalar_cases(DdlaHandle_t h,bool local,int& cases,long double& worst) {
    const auto one=[&](char s,char u,char ta,char tb,int n,int k,int nb,int mode,bool shifted) {
        if(local) local_case<Backend,T>(h,s,u,ta,tb,n,k,mode,cases,worst);
        distributed_case<Backend,T>(h,s,u,ta,tb,n,k,nb,mode,shifted,cases,worst);
    };
    for(char s:{'G','S','H'}) for(char u:{'U','L'}) {
        for(char ta:{'N','T','C'}) for(char tb:{'N','T','C'}) {
            one(s,u,ta,tb,13,17,4,0,true);
            one(s,u,ta,tb,1,3,4,1,true);
        }
        for(int mode:{1,2,3,4,6}) one(s,u,'C','T',11,7,4,mode,true);
        one(s,u,'T','C',7,0,4,0,true);
        one(s,u,'N','C',0,7,4,0,false);
        one(s,u,'T','N',17,9,1,0,true);
        one(s,u,'C','C',13,17,3,0,true);
        one(s,u,'C','C',9,13,32,1,false);
        one(s,u,'N','T',257,19,32,1,true);
        // Contraction and reflection groups cross 1024 with partial tails.
        one(s,u,'C','C',35,1031,16,0,true);
        one(s,u,'N','N',1031,3,128,1,true);
    }
}

template <DdlaBackend Backend> void invalid_cases(DdlaHandle_t h,bool local) {
    Matrix<double> a(h,8,8,4,4,0,0,1,false),b(h,8,8,4,4,0,0,2,false),c(h,8,8,4,4,0,0,3,false);
    const bool root=ddlaGetRank(h)==0;
    const auto reject=[&](auto fn) {
        int caught=0;try { fn(); } catch(const std::exception&) { caught=1; }
        int all=0;MPI_Allreduce(&caught,&all,1,MPI_INT,MPI_MIN,ddlaGetCommunicator(h));
        if(!all) throw std::runtime_error("triangular product invalid argument not rejected on every rank");
    };
    const auto call=[&](char u,char ta,char tb,int n,int k,const int* dc,const double* ap,double* cp) {
        pgemmt<Backend,double>(h,u,ta,tb,n,k,1.,ap,a.d,b.data,b.d,0.,cp,dc);
    };
    reject([&]{call(root ? '?' : 'U','N','N',8,8,c.d,a.data,c.data);});
    reject([&]{call('U',root ? '?' : 'N','N',8,8,c.d,a.data,c.data);});
    reject([&]{call('U','N',root ? '?' : 'N',8,8,c.d,a.data,c.data);});
    reject([&]{call('U','N','N',-1,8,c.d,a.data,c.data);});
    reject([&]{call('U','N','N',8,-1,c.d,a.data,c.data);});
    reject([&]{call('U','N','N',a.d[DDLA_M_]+1,8,c.d,a.data,c.data);});
    int bad_desc[DDLA_DLEN_];std::copy(c.d,c.d+DDLA_DLEN_,bad_desc);
    if(root) bad_desc[DDLA_LLD_]=0;
    reject([&]{call('U','N','N',8,8,bad_desc,a.data,c.data);});
    reject([&]{call('U','N','N',8,8,c.d,root ? nullptr : a.data,c.data);});
    reject([&]{call('U','N','N',8,8,c.d,a.data,root ? nullptr : c.data);});
    reject([&]{call('U','N','N',8,8,c.d,a.data,root ? a.data : c.data);});
#if DDLA_HAS_GPU
    constexpr DdlaBackend other=Backend==DdlaBackend::CPU ? DdlaBackend::GPU : DdlaBackend::CPU;
    reject([&]{pgemmt<other,double>(h,'U','N','N',0,0,1.,nullptr,a.d,nullptr,b.d,0.,nullptr,c.d);});
    if(local) reject([&]{gemmt<other,double>(h,'U','N','N',0,0,1.,nullptr,1,nullptr,1,0.,nullptr,1);});
#endif
    if(local) {
        ddla::vector<double> x(h,64),y(h,64),z(h,64);
        const auto lc=[&](char u,char ta,char tb,int n,int k,int lda,double* cp) {
            gemmt<Backend,double>(h,u,ta,tb,n,k,1.,x.data(),lda,y.data(),8,0.,cp,8);
        };
        reject([&]{lc('?','N','N',8,8,8,z.data());});
        reject([&]{lc('U','?','N',8,8,8,z.data());});
        reject([&]{lc('U','N','?',8,8,8,z.data());});
        reject([&]{lc('U','N','N',-1,8,8,z.data());});
        reject([&]{lc('U','N','N',8,-1,8,z.data());});
        reject([&]{lc('U','N','N',8,8,7,z.data());});
        reject([&]{lc('U','N','N',8,8,8,nullptr);});
        reject([&]{lc('U','N','N',8,8,8,x.data());});
    }
    if(ddlaGetSize(h)>1) {
        reject([&]{call(root?'L':'U','N','N',8,8,c.d,a.data,c.data);});
        reject([&]{call('U',root?'T':'N','N',8,8,c.d,a.data,c.data);});
        reject([&]{call('U','N','N',root?7:8,8,c.d,a.data,c.data);});
    }

}

template <DdlaBackend Backend>
void run(MPI_Comm comm,int pr,int pc,char major,bool local) {
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,Backend));DDLA_CHECK(ddlaSet(h,comm,pr,pc,major));
    int cases=0;long double worst=0;
    scalar_cases<Backend,float>(h,local,cases,worst);
    scalar_cases<Backend,double>(h,local,cases,worst);
    scalar_cases<Backend,std::complex<float>>(h,local,cases,worst);
    scalar_cases<Backend,std::complex<double>>(h,local,cases,worst);
    if(local)
        for(char symmetry:{'G','S','H'})for(char uplo:{'U','L'}) {
            analytic_large<Backend,float>(h,symmetry,uplo,cases,worst);
            analytic_large<Backend,double>(h,symmetry,uplo,cases,worst);
            analytic_large<Backend,std::complex<float>>(h,symmetry,uplo,cases,worst);
            analytic_large<Backend,std::complex<double>>(h,symmetry,uplo,cases,worst);
        }
    if (const char* text=std::getenv("DDLA_GEMMT_EXTRA_LAYOUTS")) {
        const int requested=std::atoi(text);
        const int order=requested>1 ? requested : 1567;
        for (int block : {129,200,256}) for (char uplo : {'U','L'}) {
            analytic_large<Backend,float>(h,'G',uplo,cases,worst,order,1031,block);
            analytic_large<Backend,double>(h,'G',uplo,cases,worst,order,1031,block);
            analytic_large<Backend,std::complex<float>>(h,'G',uplo,cases,worst,order,1031,block);
            analytic_large<Backend,std::complex<double>>(h,'G',uplo,cases,worst,order,1031,block);
            if (block==200) {
                analytic_large<Backend,float>(h,'G',uplo,cases,worst,order,1031,block,true);
                analytic_large<Backend,double>(h,'G',uplo,cases,worst,order,1031,block,true);
                analytic_large<Backend,std::complex<float>>(h,'G',uplo,cases,worst,order,1031,block,true);
                analytic_large<Backend,std::complex<double>>(h,'G',uplo,cases,worst,order,1031,block,true);
            }
        }
    }
    invalid_cases<Backend>(h,local);
    if(ddlaGetRank(h)==0) std::cout<<"GEMMT_PASS backend="<<int(Backend)<<" grid="<<pr<<'x'<<pc
        <<" major="<<major<<" local_tested="<<local<<" cases="<<cases
        <<" worst="<<double(worst)<<std::endl;
    DDLA_CHECK(ddlaDestroy(h));
}
} // namespace

int main(int argc,char** argv) {
    ddla_test_mpi_init(&argc,&argv);
    try {
        const std::string mode=argc>1 ? argv[1] : "all";
        if(mode!="all" && mode!="cpu" && mode!="gpu") throw std::runtime_error("usage: test_gemmt [all|cpu|gpu] [grid_rows grid_cols]");
        if((mode=="gpu" && !DDLA_HAS_GPU)) throw std::runtime_error("requested backend unavailable");
        int size,rank;MPI_Comm_size(MPI_COMM_WORLD,&size);MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        std::vector<std::pair<int,int>> grids;
        if(argc==4) {
            const int pr=std::stoi(argv[2]),pc=std::stoi(argv[3]);
            if(pr<=0 || pc<=0 || pr*pc!=size) throw std::runtime_error("grid does not match rank count");
            grids.emplace_back(pr,pc);
        } else if(argc>2) throw std::runtime_error("both grid dimensions are required");
        else for(int pr=1;pr<=size;++pr) if(size%pr==0) grids.emplace_back(pr,size/pr);
        MPI_Comm reversed;MPI_Comm_split(MPI_COMM_WORLD,0,size-rank,&reversed);
        bool local=true;
        for(auto [pr,pc]:grids) for(char major:{'R','C'}) {
            if(mode!="gpu") run<DdlaBackend::CPU>(reversed,pr,pc,major,local);
#if DDLA_HAS_GPU
            if(mode!="cpu") run<DdlaBackend::GPU>(reversed,pr,pc,major,local);
#endif
            local=false;
        }
        MPI_Comm_free(&reversed);MPI_Finalize();return 0;
    } catch(const std::exception& e) {
        std::cerr<<"GEMMT_TEST_FAILURE "<<e.what()<<std::endl;
        MPI_Abort(MPI_COMM_WORLD,1);return 1;
    }
}
