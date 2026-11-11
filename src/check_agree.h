#ifndef DDLA_CHECK_AGREE_H
#define DDLA_CHECK_AGREE_H

// ---------------------------------------------------------------------------
// Collective argument check: the first component of every distributed
// routine's check / plan / run split.
//
// Each rank validates what it can see, then ONE MPI_Allreduce decides for the
// whole communicator: if any rank rejected its arguments, every rank throws
// and nobody is left waiting in a later collective.  The rank that found the
// problem rethrows its own exception, so the type and the message are the ones
// a local check would have produced; the other ranks throw
// std::invalid_argument("<routine>: invalid argument on another rank").
//
// Environment-driven choices (DDLA_LU_PANEL, DDLA_QR_PANEL, ...) are read
// inside the check and registered with Agree::mode; they ride the same
// Allreduce, and a choice that differs between ranks throws
// std::runtime_error("<name> differs between the ranks").  Nothing else in a
// routine reads the environment.
//
// A handle without a communicator, or with a single rank, is checked locally.
// ---------------------------------------------------------------------------

#include <ddla/ddla_handle_t.h>
#include <ddla/ddla_desc.h>
#include "ddla_connector.h"
#include "ddla_stream.h"

#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace ddla {
namespace detail {

struct Agree {
    static constexpr int kMaxModes = 6;

    const char* names[kMaxModes] = {};
    int values[kMaxModes] = {};
    int count = 0;

    // Register an environment-driven choice every rank has to make identically.
    void mode(const char* name, int value)
    {
        if (count >= kMaxModes) throw std::logic_error("ddla: too many agreed modes");
        names[count] = name;
        values[count] = value;
        ++count;
    }
};

// Global operation metadata must agree before any rank takes an empty-path
// return or starts a descriptor-dependent communication schedule. LLD and
// local pointers/coordinates deliberately do not belong in this signature.
template<std::size_t N>
void agree_signature(const DdlaHandle_t& h,const char* name,const int (&signature)[N])
{
    if(h==nullptr||h->nprocs<=1||h->comm==MPI_COMM_NULL)return;
    int smallest[N],largest[N];
    MPI_CHECK(MPI_Allreduce(signature,smallest,static_cast<int>(N),MPI_INT,MPI_MIN,h->comm));
    MPI_CHECK(MPI_Allreduce(signature,largest,static_cast<int>(N),MPI_INT,MPI_MAX,h->comm));
    for(std::size_t i=0;i<N;++i)if(smallest[i]!=largest[i])
        throw std::invalid_argument(std::string(name)+": global metadata differs between ranks");
}
inline void descriptor_signature(int* destination,const int* descriptor) noexcept
{
    constexpr int fields[]={DDLA_DTYPE_,DDLA_M_,DDLA_N_,DDLA_MB_,DDLA_NB_,DDLA_RSRC_,DDLA_CSRC_};
    for(std::size_t i=0;i<7;++i)destination[i]=descriptor[fields[i]];
}

struct Unit {};

template <typename Fn>
auto agree_impl(const DdlaHandle_t& h, const char* name, Fn& fn)
    -> decltype(fn(std::declval<Agree&>()))
{
    using Result = decltype(fn(std::declval<Agree&>()));

    Agree agree;
    if (h == nullptr || h->nprocs <= 1 || h->comm == MPI_COMM_NULL) return fn(agree);

    std::exception_ptr failure;
    Result result{};
    try {
        result = fn(agree);
    } catch (...) {
        failure = std::current_exception();
    }

    // [bad, then (value, -value) per mode]: MAX gives the maximum and, negated, the minimum.
    int reduced[1 + 2 * Agree::kMaxModes] = {};
    reduced[0] = failure ? 1 : 0;
    for (int i = 0; i < agree.count; ++i) {
        reduced[1 + 2 * i] = agree.values[i];
        reduced[2 + 2 * i] = -agree.values[i];
    }
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, reduced, 1 + 2 * Agree::kMaxModes, MPI_INT, MPI_MAX, h->comm));

    if (reduced[0]) {
        if (failure) std::rethrow_exception(failure);
        throw std::invalid_argument(std::string(name) + ": invalid argument on another rank");
    }
    for (int i = 0; i < agree.count; ++i) {
        if (reduced[1 + 2 * i] != -reduced[2 + 2 * i])
            throw std::runtime_error(std::string(agree.names[i]) + " differs between the ranks");
    }
    return result;
}

// Run fn(Agree&) on every rank and agree on the outcome.  fn does all of the
// routine's validation (it may throw) and returns what the later phases need
// (a check with nothing to return may return void).
template <typename Fn>
auto agree_check(const DdlaHandle_t& h, const char* name, Fn&& fn)
    -> decltype(std::declval<Fn&>()(std::declval<Agree&>()))
{
    using Result = decltype(std::declval<Fn&>()(std::declval<Agree&>()));
    if constexpr (std::is_void<Result>::value) {
        auto unit = [&](Agree& agree) { fn(agree); return Unit{}; };
        agree_impl(h, name, unit);
    } else {
        return agree_impl(h, name, fn);
    }
}

// ---------------------------------------------------------------------------
// Collective failure agreement for the steps after the check.
//
// Sizing the scratch, an allocation or starting a worker team can still fail
// on one rank only.  agree_local runs such a rank-local step and reduces its
// outcome, so every rank leaves together instead of waiting in a later
// collective: the failing rank rethrows its own exception, the others throw
// std::runtime_error("<what> failed on another rank").  agree_failure is the
// reduction alone, for a caller that caught the exception itself.
// ---------------------------------------------------------------------------
inline void agree_failure(MPI_Comm comm, const std::exception_ptr& failure, const char* what)
{
    int failed = failure ? 1 : 0;
    MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, comm));
    if (failed) {
        if (failure) std::rethrow_exception(failure);
        throw std::runtime_error(std::string(what) + " failed on another rank");
    }
}
inline void agree_failure(const DdlaHandle_t& h, const std::exception_ptr& failure, const char* what)
{
    agree_failure(ddlaGetCommunicator(h), failure, what);
}

template <typename Comm, typename Fn>
void agree_local(const Comm& comm_or_handle, const char* what, Fn&& fn)
{
    std::exception_ptr failure;
    try {
        fn();
    } catch (...) {
        failure = std::current_exception();
    }
    agree_failure(comm_or_handle, failure, what);
}

} // namespace detail
} // namespace ddla

#endif // DDLA_CHECK_AGREE_H
