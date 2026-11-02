#ifndef DDLA_PUNMTR_IMPL_H
#define DDLA_PUNMTR_IMPL_H
#include <ddla/ddla.h>
#include <cstddef>
namespace ddla { namespace detail {
template<class T>void punmtr_check(const DdlaHandle_t&,DdlaBackend,char,char,char,
    int m,int n,int k,const T* V,int ldv,const T* tmat,int ldt,T* C,const int* descC);
template<class T>void punmtr_workspace_size(const DdlaHandle_t&,char side,int rows,int cols,int k,int ldw,
    bool reduce,std::size_t* bytes);
// Local views are already validated. No allocation, queries or setup. The
// explicit reduction flag lets independent CPU column strips skip communication.
template<DdlaBackend B,class T>void punmtr_run(const DdlaHandle_t&,char side,char uplo,char trans,
    int rows,int cols,int k,int ldv,int ldt,int ldc,int ldw,bool reduce,
    const T* V,const T* tmat,T* C,T* work,T* reduced_work);
// The grouped CPU driver keeps its failure-draining boundaries around these phases.
template<DdlaBackend B,class T>void punmtr_project_run(const DdlaHandle_t&,char side,
    int rows,int cols,int k,int ldv,int ldc,int ldw,const T* V,const T* C,T* work);
template<DdlaBackend B,class T>void punmtr_update_run(const DdlaHandle_t&,char side,
    int rows,int cols,int k,int ldv,int ldc,int ldw,const T* V,const T* product,T* C);
} }
#endif
