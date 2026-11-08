#include "api_grid_test_common.h"
#include "grouped_stage5_integration_oracle.h"
using namespace api_grid_test;
namespace {
template<class T,ddla::DdlaBackend Backend=ddla::DdlaBackend::CPU>void run_type(const ddla::DdlaHandle_t&h){
    for(auto dims:{std::array<int,3>{9,4,4},{17,4,4},{41,4,2},{73,8,4}})for(bool upper:{false,true}){
        if(Backend==ddla::DdlaBackend::GPU&&upper)continue;
        const auto r=grouped_stage5_oracle::integration<T,Backend>(h,dims[0],dims[1],dims[2],upper);
        require_close(h,"grouped native producer dense elementary",r.apply.dense,50);
        require_close(h,"grouped native producer current stage5",r.apply.sequential,50);
        require_close(h,"grouped native producer reconstruction",r.reconstruction,50);
        require_close(h,"grouped native producer readonly/padding",r.apply.guards+r.guards,0);
    }
}
void body(const ddla::DdlaHandle_t&h,const Shape&){
#if DDLA_HAS_GPU
    if(ddlaGetBackend(h)==ddla::DdlaBackend::GPU) {
        run_type<float,ddla::DdlaBackend::GPU>(h);
        run_type<double,ddla::DdlaBackend::GPU>(h);
        run_type<std::complex<float>,ddla::DdlaBackend::GPU>(h);
        run_type<std::complex<double>,ddla::DdlaBackend::GPU>(h);
        return;
    }
#endif
    run_type<float>(h);run_type<double>(h);run_type<std::complex<float>>(h);run_type<std::complex<double>>(h);
}
}
int main(int argc,char**argv){return run_grid_test(argc,argv,"test_punmtr_he2hb_grouped_integration",body);}
