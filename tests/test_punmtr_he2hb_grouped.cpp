#include "api_grid_test_common.h"
#include "grouped_stage5_oracle.h"
using namespace api_grid_test;
namespace {
template<class T>void run_type(const ddla::DdlaHandle_t&h){
    for(auto shape:{std::array<int,3>{0,4,4},{1,4,4},{2,1,1},{9,4,4},{17,4,4},{22,4,2},{41,8,2}})
        for(int pattern:{0,1,2,3}){
            const auto r=grouped_stage5_oracle::direct<T>(h,shape[0],shape[1],shape[2],pattern,pattern?0:5,pattern==3?0:3);
            require_close(h,"grouped source QL vs dense elementary",r.dense,50);
            require_close(h,"grouped vs current stage5",r.sequential,50);
            require_close(h,"grouped readonly/parent/padding",r.guards,0);
        }
    for(auto shape:{std::array<int,3>{0,4,4},{1,4,4},{2,1,1},{9,4,4},{17,4,4},{22,4,2},{41,8,2}})
        for(int pad:{0,3})for(bool reverse:{false,true}){
            const auto r=grouped_stage5_oracle::shared_parent<T>(h,shape[0],shape[1],shape[2],pad,reverse);
            require_close(h,"shared-parent dense elementary",r.dense,50);
            require_close(h,"shared-parent current stage5",r.sequential,50);
            require_close(h,"shared-parent backup/readonly/padding/overlap rejection",r.guards,0);
        }
    require_close(h,"grouped local argument validation",grouped_stage5_oracle::validation<T>(h),0);
}
void body(const ddla::DdlaHandle_t&h,const Shape&){
    double policy=0;
    policy+=grouped_stage5_oracle::policy_queries<float>(h);
    policy+=grouped_stage5_oracle::policy_queries<double>(h);
    policy+=grouped_stage5_oracle::policy_queries<std::complex<float>>(h);
    policy+=grouped_stage5_oracle::policy_queries<std::complex<double>>(h);
    require_close(h,"stage5 shared policy and exact queries at4999/5000/5001",policy,0);
#if DDLA_HAS_GPU
    if(ddlaGetBackend(h)==ddla::DdlaBackend::GPU) {
        ddla::DdlaHandle_t cpu=nullptr;
        DDLA_CHECK(ddlaInit(cpu,ddla::DdlaBackend::CPU));
        DDLA_CHECK(ddlaSet(cpu,MPI_COMM_SELF,1,1));
        grouped_stage5_oracle::device_cases<float>(h,cpu);
        grouped_stage5_oracle::device_cases<double>(h,cpu);
        grouped_stage5_oracle::device_cases<std::complex<float>>(h,cpu);
        grouped_stage5_oracle::device_cases<std::complex<double>>(h,cpu);
        DDLA_CHECK(ddlaDestroy(cpu));
        return;
    }
#endif
    run_type<float>(h);run_type<double>(h);run_type<std::complex<float>>(h);run_type<std::complex<double>>(h);
}
}
int main(int argc,char**argv){return run_grid_test(argc,argv,"test_punmtr_he2hb_grouped",body);}
