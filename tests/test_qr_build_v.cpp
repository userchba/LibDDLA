#include "mpi_test_init.h"
#include <ddla/ddla.h>
#include "ddla_stream.h"
#include "QR/pgeqrf_kernels.h"
#include <algorithm>
#include <complex>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

template<class T> T value(int row, int column)
{
    using R = typename ddla::real_type<T>::type;
    const R real = R((row * 17 + column * 31) % 127 - 63);
    if constexpr (std::is_same_v<T, R>) return real;
    else return T(real, R((row * 7 + column * 19) % 101 - 50));
}
template<class T> std::uint64_t hash(const std::vector<T>& data)
{
    const auto* p = reinterpret_cast<const unsigned char*>(data.data());
    std::uint64_t result = 1469598103934665603ULL;
    for (std::size_t i = 0; i < data.size() * sizeof(T); ++i) result = (result ^ p[i]) * 1099511628211ULL;
    return result;
}
template<class T> void check(ddla::DdlaHandle_t h, int global_rows, int columns, int rev, int mode)
{
    constexpr int block = 73, start = 37;
    const int source_rank = h->nprocs > 1 ? 1 : 0;
    const int rank = h->myprow_, ranks = h->nprows_;
    const int first = ddla::num_loc(start, block, rank, source_rank, ranks);
    const int rows = ddla::num_loc(start + global_rows, block, rank, source_rank, ranks) - first;
    const int ld = rows + 7;
    const T guard = T(-12345);
    std::vector<T> output(5 + std::size_t(ld) * (columns + 2) + 7, guard);
    T* const v = output.data() + 5;
    const bool alias = mode == 0;
    const bool local = mode != 2 || h->myid != 0;
    const int source_rows = local ? rows : global_rows;
    const int source_ld = source_rows + 3;
    std::vector<T> input(alias ? 0 : std::size_t(source_ld) * columns, guard);
    for (int c = 0; c < columns; ++c) {
        if (alias) {
            for (int i = 0; i < rows; ++i)
                v[i + std::size_t(c) * ld] = value<T>(ddla::indxl2g(first + i, block, rank, source_rank, ranks), c);
        } else {
            for (int i = 0; i < source_rows; ++i) {
                const int g = local ? ddla::indxl2g(first + i, block, rank, source_rank, ranks) : start + i;
                input[i + std::size_t(c) * source_ld] = value<T>(g, c);
            }
        }
    }
    const auto input_hash = hash(input);
    ddla::detail::qr_build_v<ddla::DdlaBackend::CPU, T>(h, alias || local, rev, v, ld,
        alias ? v : input.data(), alias ? ld : source_ld, rows, columns, start,
        global_rows, first, block, rank, source_rank, ranks);
    if (hash(input) != input_hash) throw std::runtime_error("source modified");
    for (std::size_t q = 0; q < output.size(); ++q) {
        T expected = guard;
        if (q >= 5) {
            const std::size_t relative = q - 5;
            const int c = int(relative / ld), i = int(relative % ld);
            if (c < columns && i < rows) {
                const int g = ddla::indxl2g(first + i, block, rank, source_rank, ranks);
                const int pivot = rev ? start + global_rows - columns + c : start + c;
                expected = g == pivot ? T(1) : (rev ? g > pivot : g < pivot) ? T(0) : value<T>(g, c);
            }
        }
        if (output[q] != expected) throw std::runtime_error("reflector or padding mismatch");
    }
    ddla::detail::qr_build_v<ddla::DdlaBackend::CPU, T>(h, true, rev, nullptr, ld,
        nullptr, source_ld, 0, columns, start, global_rows, first, block, rank, source_rank, ranks);
    ddla::detail::qr_build_v<ddla::DdlaBackend::CPU, T>(h, true, rev, nullptr, ld,
        nullptr, source_ld, rows, 0, start, global_rows, first, block, rank, source_rank, ranks);
}
template<class T> void run(ddla::DdlaHandle_t h)
{
    for (int rev : {0, 1}) for (int mode : {0, 1, 2}) {
        check<T>(h, 71, 9, rev, mode);
        check<T>(h, 8195, 257, rev, mode);
    }
}
int main(int argc, char** argv)
{
    ddla_test_mpi_init(&argc, &argv);
    ddla::DdlaHandle_t h = nullptr;
    try {
        int ranks = 1; MPI_Comm_size(MPI_COMM_WORLD, &ranks);
        DDLA_CHECK(ddla::ddlaInit(h, ddla::DdlaBackend::CPU));
        DDLA_CHECK(ddla::ddlaSet(h, MPI_COMM_WORLD, ranks, 1));
        const int budget = h->host_threads;
        for (int threads : {1, 2, std::max(2, budget)}) {
            h->host_threads = threads;
            run<float>(h); run<double>(h); run<std::complex<float>>(h); run<std::complex<double>>(h);
        }
        h->host_threads = budget;
        if (h->myid == 0) std::cout << "QR_BUILD_V_THREADS_PASS ranks=" << ranks << " budget=" << budget << std::endl;
        DDLA_CHECK(ddla::ddlaDestroy(h));
        MPI_Finalize(); return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl; MPI_Abort(MPI_COMM_WORLD, 1); return 1;
    }
}
