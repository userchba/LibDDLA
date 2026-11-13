#pragma once
#include "api_grid_test_common.h"
#include <complex>
#include <limits>
#include <type_traits>

namespace generalized_test {
template<class T>T conjugate(T x) {
    if constexpr(std::is_arithmetic_v<T>)return x;
    else return std::conj(x);
}
template<class T>T phase(int i) {
    if constexpr(std::is_arithmetic_v<T>)return T(1);
    else return T(std::cos(.23*i),std::sin(.23*i));
}
inline double eigenvalue(int i,bool repeated=false){return (repeated?i/2:i)-3.;}
template<class T>T standard(int i,int j,int n,bool repeated=false) {
    double norm=0,weighted=0;
    for(int k=0;k<n;++k){norm+=(k+1.)*(k+1.);weighted+=(k+1.)*(k+1.)*eigenvalue(k,repeated);}
    const double x=(i==j?eigenvalue(i,repeated):0)+(i+1.)*(j+1.)/norm*
        (4*weighted/norm-2*eigenvalue(i,repeated)-2*eigenvalue(j,repeated));
    return T(x)*phase<T>(i)*conjugate(phase<T>(j));
}
inline double diagonal(int i,int n){return 1.+double(i+1)/(n+1);}
template<class T>T subdiagonal(int i) {
    if constexpr(std::is_arithmetic_v<T>)return T(.17);
    else return T(.17,.09)*phase<T>(i);
}
// B=L L^H, A=L Q Lambda Q^H L^H. L is a well-conditioned lower
// bidiagonal matrix; Q is a phased dense Householder matrix. The known
// spectrum includes negative/repeated eigenvalues and A/B do not commute.
template<class T>T a(int i,int j,int n,bool repeated=false) {
    T x=T(diagonal(i,n)*diagonal(j,n))*standard<T>(i,j,n,repeated);
    if(i)x+=subdiagonal<T>(i)*T(diagonal(j,n))*standard<T>(i-1,j,n,repeated);
    if(j)x+=T(diagonal(i,n))*conjugate(subdiagonal<T>(j))*standard<T>(i,j-1,n,repeated);
    if(i&&j)x+=subdiagonal<T>(i)*conjugate(subdiagonal<T>(j))*standard<T>(i-1,j-1,n,repeated);
    return x;
}
template<class T>T b(int i,int j,int n) {
    if(i==j)return T(diagonal(i,n)*diagonal(i,n))+(i?subdiagonal<T>(i)*conjugate(subdiagonal<T>(i)):T(0));
    if(i==j+1)return subdiagonal<T>(i)*T(diagonal(j,n));
    if(j==i+1)return T(diagonal(i,n))*conjugate(subdiagonal<T>(j));
    return T(0);
}
template<class T>std::complex<double> wide(T x) {
    if constexpr(std::is_arithmetic_v<T>)return {double(x),0};
    else return {double(x.real()),double(x.imag())};
}
template<class T>
void verify(const ddla::DdlaHandle_t& h,int n,int nev,const typename ddla::real_type<T>::type* w,
            const std::vector<T>& z,bool repeated=false) {
    using R=typename ddla::real_type<T>::type;
    const double eps=std::numeric_limits<R>::epsilon();
    double ev=0,residual=0,orth=0,norma=0,normb=0;
    for(int i=0;i<n;++i) {
        if(!std::isfinite(double(w[i])))ev=1e30;
        ev=std::max(ev,std::abs(double(w[i])-eigenvalue(i,repeated))/(eps*std::max(1,n)*std::max(1.,std::abs(eigenvalue(i,repeated)))));
        double sa=0,sb=0;
        for(int j=0;j<n;++j){sa+=std::abs(wide(a<T>(i,j,n,repeated)));sb+=std::abs(wide(b<T>(i,j,n)));}
        norma=std::max(norma,sa);normb=std::max(normb,sb);
    }
    std::vector<std::complex<double>> bz(std::size_t(n)*nev);
    for(int j=0;j<nev;++j) {
        double znorm=0,rnorm=0;
        for(int i=0;i<n;++i) {
            std::complex<double> az=0,bs=0;
            for(int k=0;k<n;++k) {
                az+=wide(a<T>(i,k,n,repeated))*wide(z[k+std::size_t(j)*n]);
                bs+=wide(b<T>(i,k,n))*wide(z[k+std::size_t(j)*n]);
            }
            bz[i+std::size_t(j)*n]=bs;
            const double r=std::abs(az-double(w[j])*bs),v=std::abs(wide(z[i+std::size_t(j)*n]));
            if(!std::isfinite(r)||!std::isfinite(v))residual=1e30;
            rnorm=std::max(rnorm,r);znorm=std::max(znorm,v);
        }
        residual=std::max(residual,rnorm/(eps*std::max(1,n)*(norma+std::abs(double(w[j]))*normb)*std::max(znorm,1e-30)));
    }
    for(int j=0;j<nev;++j)for(int i=0;i<nev;++i) {
        std::complex<double> dot=0;
        for(int k=0;k<n;++k)dot+=std::conj(wide(z[k+std::size_t(i)*n]))*bz[k+std::size_t(j)*n];
        const double error=std::abs(dot-(i==j?1.:0.))/(eps*std::max(1,n));
        if(!std::isfinite(error))orth=1e30;
        orth=std::max(orth,error);
    }
    api_grid_test::require_close(h,"generalized known eigenvalues (scaled)",ev,200);
    api_grid_test::require_close(h,"original A Z - B Z Lambda (scaled)",residual,200);
    api_grid_test::require_close(h,"Z^H B Z - I (scaled)",orth,200);
}
} // namespace generalized_test
