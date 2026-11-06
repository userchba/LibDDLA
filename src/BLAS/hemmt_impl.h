#ifndef DDLA_HEMMT_IMPL_H
#define DDLA_HEMMT_IMPL_H
#include <ddla/ddla.h>
#include <cstddef>
namespace ddla { namespace blas_impl {
template <DdlaBackend Backend,typename T,typename TA=T,typename TB=T>
bool hemmt_check(const DdlaHandle_t& h,char side,char uploA,char uploC,int n,
    const T& alpha,const TA* A,int lda,const TB* B,int ldb,const T& beta,T* C,int ldc,
    const char* routine = "hemmt",int free_dimension = -1);
template <typename T,typename TA=T,typename TB=T>
void hemmt_workspace_size(const DdlaHandle_t& h,int n,bool product,std::size_t* bytes,int free_dimension = -1,char side = 'L',char uploC = 'U');
// The same panel product serves HEMMT and SYMMT. Symmetric reflection uses
// transpose and retains complex diagonal entries. Workspace geometry is shared.
template <DdlaBackend Backend,typename T,typename TA=T,typename TB=T>
void hemmt_run(const DdlaHandle_t& h,char side,char uploA,char uploC,int n,
    const T& alpha,const TA* A,int lda,const TB* B,int ldb,const T& beta,T* C,int ldc,
    void* work,std::size_t bytes,bool hermitian = true,int free_dimension = -1);
// n is the order of A. free_dimension defaults to n; F output permits a
// rectangular general B/C, shared with the mixed HEMM overloads.
// Complete a stored diagonal block into scratch, preserving the source.
template <DdlaBackend Backend,typename T>
void diagonal_block_run(const DdlaHandle_t& h,bool upper,bool hermitian,int width,
    const T* A,int lda,T* output,int ldo);
}}
#endif
