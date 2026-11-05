#ifndef DDLA_PBLAS_STRUCTURED_TEST_COMMON_H
#define DDLA_PBLAS_STRUCTURED_TEST_COMMON_H

#include <ddla/ddla.h>
#include "ddla_connector.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace structured_test {
using namespace ddla;
enum class Operation { Symm, Syrk, Syr2k, Herk, Her2k };
inline const char* name(Operation op) {
    switch(op) {
        case Operation::Symm:return "psymm";
        case Operation::Syrk:return "psyrk";
        case Operation::Syr2k:return "psyr2k";
        case Operation::Herk:return "pherk";
        case Operation::Her2k:return "pher2k";
    }
    throw std::runtime_error("unknown operation");
}
inline bool hermitian(Operation op) {return op==Operation::Herk || op==Operation::Her2k;}
inline bool two_inputs(Operation op) {return op==Operation::Symm || op==Operation::Syr2k || op==Operation::Her2k;}
template <typename T> T scalar(double re,double im=0) {
    if constexpr(std::is_arithmetic_v<T>) return T(re); else return T(re,im);
}
template <typename T> std::complex<double> wide(T x) {
    if constexpr(std::is_arithmetic_v<T>) return {double(x),0};else return {double(x.real()),double(x.imag())};
}
template <typename T> T value(int i,int j,int salt) {
    const std::uint64_t a=(std::uint64_t(i)+1)*104729+(std::uint64_t(j)+3)*13007+salt*8191;
    return scalar<T>(double(int((a^(a>>7))%997)-498)/512.,
                     double(int(((a*65537)^(a>>3))%991)-495)/512.);
}
template <typename T> const char* type_name() {
    if constexpr(std::is_same_v<T,float>) return "s";
    else if constexpr(std::is_same_v<T,double>) return "d";
    else if constexpr(std::is_same_v<T,std::complex<float>>) return "c";
    else return "z";
}

template <typename T> struct Matrix {
    DdlaHandle_t h;int d[DDLA_DLEN_],nr,nc,pr,pc,r,c;
    std::vector<T> host;ddla::vector<T> data;
    Matrix(DdlaHandle_t handle,int m,int n,int nb,int rs=0,int cs=0,bool padded=false):h(handle),data(h) {
        ddlaGetGridDims(h,pr,pc);ddlaGetGridCoords(h,r,c);
        nr=num_loc(m+(padded ? 3 : 0),nb,r,rs,pr);nc=num_loc(n+(padded ? 5 : 0),nb,c,cs,pc);
        DDLA_CHECK(ddlaDescInit(d,h,m+(padded ? 3 : 0),n+(padded ? 5 : 0),nb,nb,rs,cs,std::max(1,nr+(padded ? 3 : 0))));
        host.assign(std::max(std::size_t(1),std::size_t(d[DDLA_LLD_])*nc),scalar<T>(-987.25,654.5));
        data.resize(host.size());
    }
    int gi(int i) const {return indxl2g(i,d[DDLA_MB_],r,d[DDLA_RSRC_],pr);}
    int gj(int j) const {return indxl2g(j,d[DDLA_NB_],c,d[DDLA_CSRC_],pc);}
    template <typename F> void fill(int m,int n,F f) {
        for(int j=0;j<nc;++j) for(int i=0;i<nr;++i)
            if(gi(i)<m && gj(j)<n) host[i+std::size_t(j)*d[DDLA_LLD_]]=f(gi(i),gj(j));
    }
    void upload() {DDLA_CHECK(ddlaMemcpy(h,data.data(),host.data(),host.size()*sizeof(T)));}
    std::vector<T> download() const {
        std::vector<T> out(host.size());DDLA_CHECK(ddlaMemcpy(h,out.data(),data.data(),out.size()*sizeof(T)));
        DDLA_CHECK(ddlaSynchronize(h));return out;
    }
};

template <DdlaBackend Backend,typename T>
void invoke(DdlaHandle_t h,Operation op,char side,char uplo,char trans,int m,int n,int k,
            T alpha,const T* a,const int* da,const T* b,const int* db,T beta,T* c,const int* dc) {
    using R=typename real_type<T>::type;
    switch(op) {
        case Operation::Symm:psymm<Backend,T>(h,side,uplo,m,n,alpha,a,da,b,db,beta,c,dc);break;
        case Operation::Syrk:psyrk<Backend,T>(h,uplo,trans,n,k,alpha,a,da,beta,c,dc);break;
        case Operation::Syr2k:psyr2k<Backend,T>(h,uplo,trans,n,k,alpha,a,da,b,db,beta,c,dc);break;
        case Operation::Herk:
            if constexpr(!std::is_arithmetic_v<T>) pherk<Backend,T>(h,uplo,trans,n,k,R(alpha.real()),a,da,R(beta.real()),c,dc);
            else throw std::runtime_error("HERK requires complex type");
            break;
        case Operation::Her2k:
            if constexpr(!std::is_arithmetic_v<T>) pher2k<Backend,T>(h,uplo,trans,n,k,alpha,a,da,b,db,R(beta.real()),c,dc);
            else throw std::runtime_error("HER2K requires complex type");
            break;
    }
}

// An independent element-wise oracle: no DDLA, BLAS or redistribution helper.
template <typename T>
std::pair<std::complex<double>,double> reference(Operation op,char side,char uplo,char trans,
    int x,int y,int m,int n,int k,T alpha,T beta) {
    std::complex<double> sum=0;double bound=0;
    const auto av=[&](int i,int l,int salt) {
        auto z=wide(trans=='N' ? value<T>(i,l,salt) : value<T>(l,i,salt));
        return trans=='C' ? std::conj(z) : z;
    };
    if(alpha!=T(0)) {
        if(op==Operation::Symm) {
            const int order=side=='L' ? m : n;
            for(int l=0;l<order;++l) {
                const int ai=side=='L' ? x : l,aj=side=='L' ? l : y;
                const bool stored=uplo=='U' ? ai<=aj : ai>=aj;
                const auto a=wide(value<T>(stored ? ai : aj,stored ? aj : ai,1));
                const auto b=wide(side=='L' ? value<T>(l,y,2) : value<T>(x,l,2));
                sum+=wide(alpha)*a*b;bound+=std::abs(wide(alpha))*std::abs(a)*std::abs(b);
            }
        } else for(int l=0;l<k;++l) {
            const bool h=hermitian(op),two=two_inputs(op);
            const auto ax=av(x,l,1),ay=av(y,l,1),bx=av(x,l,2),by=av(y,l,2);
            const auto right=h ? std::conj(two ? by : ay) : (two ? by : ay);
            sum+=wide(alpha)*ax*right;bound+=std::abs(wide(alpha))*std::abs(ax)*std::abs(right);
            if(two) {
                const auto coefficient=h ? std::conj(wide(alpha)) : wide(alpha);
                const auto ar=h ? std::conj(ay) : ay;
                sum+=coefficient*bx*ar;bound+=std::abs(coefficient)*std::abs(bx)*std::abs(ar);
            }
        }
    }
    if(beta!=T(0)) {
        auto c=wide(value<T>(x,y,3));if(hermitian(op) && x==y)c={c.real(),0};
        sum+=wide(beta)*c;bound+=std::abs(wide(beta))*std::abs(c);
    }
    if(hermitian(op) && x==y)sum={sum.real(),0};
    return {sum,bound};
}
} // namespace structured_test
#endif
