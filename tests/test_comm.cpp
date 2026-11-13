#include "comm_test_common.h"
using namespace comm_test;
int main(int argc,char** argv)
{
    ddla_test_mpi_init(&argc,&argv);
    try {
        bool staged=false;
        for(int i=1;i<argc;++i) {
            if(std::string(argv[i])=="--host-staged")staged=true;
            else throw std::invalid_argument("usage: test_comm [--host-staged]");
        }
        int size,rank;MPI_Comm_size(MPI_COMM_WORLD,&size);MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        MPI_Comm reversed;MPI_Comm_split(MPI_COMM_WORLD,0,size-rank,&reversed);
        for(int pr=1;pr<=size;++pr)if(size%pr==0) {
            run<DdlaBackend::CPU>(reversed,pr,size/pr);
#if DDLA_HAS_GPU
            run<DdlaBackend::GPU>(reversed,pr,size/pr,staged);
#endif
        }
        MPI_Comm_free(&reversed);MPI_Finalize();return 0;
    }catch(const std::exception& e){std::cerr<<"COMM_HOST_BUFFERS_FAILURE "<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
