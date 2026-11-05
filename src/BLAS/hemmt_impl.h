#ifndef DDLA_HEMMT_IMPL_H
#define DDLA_HEMMT_IMPL_H
#include <ddla/ddla.h>
#include <cstddef>
namespace ddla { namespace blas_impl {
template <DdlaBackend Backend,typename T>
bool hemmt_check(const DdlaHandle_t& h,char side,char uploA,char uploC,int n,
    const T& alpha,const T* A,int lda,const T* B,int ldb,const T& beta,T* C,int ldc);
template <typename T>
void hemmt_workspace_size(const DdlaHandle_t& h,int n,bool product,std::size_t* bytes);
template <DdlaBackend Backend,typename T>
void hemmt_run(const DdlaHandle_t& h,char side,char uploA,char uploC,int n,
    const T& alpha,const T* A,int lda,const T* B,int ldb,const T& beta,T* C,int ldc,
    void* work,std::size_t bytes);
// Complete a stored diagonal block into scratch, preserving the source.
template <DdlaBackend Backend,typename T>
void diagonal_block_run(const DdlaHandle_t& h,bool upper,bool hermitian,int width,
    const T* A,int lda,T* output,int ldo);
}}
#endif
