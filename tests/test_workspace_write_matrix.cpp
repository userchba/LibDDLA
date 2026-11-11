#include "mpi_test_init.h"
#include <ddla/ddla.h>
#include <ddla/ddla_vector.h>
#include "write_matrix.h"
#include "workspace_test_common.h"

#include <array>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <locale>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <type_traits>
#include <vector>
#include <unistd.h>

namespace {
using namespace ddla;
void require(bool value,const char* message)
{if(!value)throw std::runtime_error(message);}
void clean(const workspace_test::Snapshot& s,bool query)
{
    require(s.ddla_alloc==0&&s.ddla_free==0&&s.pinned_alloc==0&&s.raw_gpu_alloc==0&&s.host_new==0,
        "write_matrix component allocated/released managed storage");
    require(s.vendor_query==0&&s.collectives==0&&s.resource_setup==0,
        "write_matrix component queried vendor work or prepared a resource");
    if(query)require(s.copies==0&&s.launches==0&&s.syncs==0,"write_matrix query executed work");
}
class FixedOutput:public std::streambuf {
    std::array<char,2048> bytes_{};
public:
    FixedOutput(){reset();}
    void reset(){setp(bytes_.data(),bytes_.data()+bytes_.size());}
    std::string str()const{return {pbase(),pptr()};}
};
template<class T>T value(double r,double i=0)
{if constexpr(std::is_arithmetic_v<T>){(void)i;return T(r);}else return T(r,i);}
template<class T>std::string expected(int n)
{
    if constexpr(std::is_arithmetic_v<T>)
        return n==3?"0 -3.25 -5 \n2 4.5 6 \n":"0 -3.25 \n2 4.5 \n";
    else
        return n==3?"(0,0) (-3.25,3) (-5,5) \n(2,-2) (4.5,-4.5) (6,-6) \n":"(0,0) (-3.25,3) \n(2,-2) (4.5,-4.5) \n";
}
std::string read_file(const std::string& filename)
{
    std::ifstream input(filename);
    require(bool(input),"write_matrix public output could not be opened");
    return {std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()};
}
template<DdlaBackend B,class T>void cases(DdlaHandle_t h,const std::string& filename)
{
    const std::vector<T> initial={value<T>(5e-12,-5e-12),value<T>(2,-2),value<T>(-3.25,3),
        value<T>(4.5,-4.5),value<T>(-5,5),value<T>(6,-6)};
    ddla::vector<T> input(h,initial.size());
    DDLA_CHECK(ddlaMemcpy(h,input.data(),initial.data(),initial.size()*sizeof(T)));
    DDLA_CHECK(ddlaSynchronize(h));
    require(matrix_io_impl::write_matrix_check(input.data(),2,3,filename.c_str()),"write_matrix valid input rejected");
    workspace_test::Scope qs(true);
    std::size_t host_bytes=1;
    matrix_io_impl::write_matrix_workspace_size<B,T>(2,3,&host_bytes);
    clean(qs.finish(),true);
    require(host_bytes==(B==DdlaBackend::GPU?detail::padded_bytes<T>(initial.size()):0),"write_matrix staging length");
    // One exactly sized staging buffer between poisoned guards; reuse it for a
    // smaller shape without another query.
    constexpr std::size_t guard=64;constexpr unsigned char poison=0xA7;
    std::vector<unsigned char> staging(host_bytes+2*guard,poison);
    auto guards_intact=[&]{
        for(std::size_t i=0;i<guard;++i)if(staging[i]!=poison||staging[guard+host_bytes+i]!=poison)return false;
        return true;};
    FixedOutput buffer;std::ostream output(&buffer);output.imbue(std::locale::classic());
    output<<0.25<<' '<<float(0.5)<<'\n'; // prepare standard stream facets outside run
    for(int repeat=0;repeat<3;++repeat){
        const int n=repeat==1?2:3;buffer.reset();output.clear();
        workspace_test::Scope run;
        matrix_io_impl::write_matrix_run<B,T>(input.data(),2,n,output,staging.data()+guard,host_bytes);
        clean(run.finish(),false);
        require(buffer.str()==expected<T>(n),"write_matrix run formatting differs");
        require(guards_intact(),"write_matrix staging canary overwritten");
        write_matrix<B,T>(input.data(),2,n,filename.c_str());
        require(read_file(filename)==expected<T>(n),"checked write_matrix differs from run/reference output");
    }
    std::vector<T> after(initial.size());DDLA_CHECK(ddlaMemcpy(h,after.data(),input.data(),after.size()*sizeof(T)));
    DDLA_CHECK(ddlaSynchronize(h));require(after==initial,"write_matrix modified input");
    const auto saved=read_file(filename);
    workspace_test::Scope invalid;
    write_matrix<B,T>(nullptr,2,3,filename.c_str());
    write_matrix<B,T>(input.data(),0,3,filename.c_str());
    write_matrix<B,T>(input.data(),2,-1,filename.c_str());
    write_matrix<B,T>(input.data(),2,3,nullptr);
    clean(invalid.finish(),true);
    require(read_file(filename)==saved,"write_matrix invalid input changed an existing file");
    require(!matrix_io_impl::write_matrix_check<T>(nullptr,2,3,filename.c_str()),"write_matrix null input accepted");
    std::remove(filename.c_str());
}
template<DdlaBackend B>void backend(const std::string& filename)
{
    DdlaHandle_t h=nullptr;DDLA_CHECK(ddlaInit(h,B));DDLA_CHECK(ddlaSet(h,MPI_COMM_SELF));
    try{cases<B,float>(h,filename);cases<B,double>(h,filename);
        cases<B,std::complex<float>>(h,filename);cases<B,std::complex<double>>(h,filename);}
    catch(...){ddlaDestroy(h);throw;}
    DDLA_CHECK(ddlaDestroy(h));
}
}
int main(int argc,char**argv)
{
    ddla_test_mpi_init(&argc,&argv);int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);int failed=0;
    const auto filename="/tmp/libddla_workspace_write_matrix_"+std::to_string(::getpid())+"_"+std::to_string(rank)+".txt";
    try{
        backend<DdlaBackend::CPU>(filename);
#if DDLA_HAS_GPU
        backend<DdlaBackend::GPU>(filename);
#endif
        std::cout<<"workspace write_matrix PASS\n";
    }catch(const std::exception& e){std::cerr<<"workspace write_matrix FAIL: "<<e.what()<<'\n';failed=1;}
    std::remove(filename.c_str());MPI_Allreduce(MPI_IN_PLACE,&failed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    MPI_Finalize();return failed;
}
