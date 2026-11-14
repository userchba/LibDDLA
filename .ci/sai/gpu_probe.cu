#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
int main() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count != 1) {
        std::fprintf(stderr, "GPU_MAPPING_ERROR count=%d error=%s\n", count,
                     cudaGetErrorString(status));
        return 1;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 2;
    char host[256] = {};
    gethostname(host, sizeof(host) - 1);
    std::printf("host=%s rank=%s local_rank=%s mask=%s device_count=%d name=%s uuid=",
                host, std::getenv("OMPI_COMM_WORLD_RANK"),
                std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"),
                std::getenv("CUDA_VISIBLE_DEVICES"), count, prop.name);
    for (unsigned char c : prop.uuid.bytes) std::printf("%02x", c);
    std::printf(" memory_bytes=%zu\n", prop.totalGlobalMem);
    return 0;
}
