#ifndef DDLA_WORKSPACE_H
#define DDLA_WORKSPACE_H

#include <ddla/ddla_vector.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>

namespace ddla {

// Borrowed contiguous storage. Copying a view never transfers ownership.
template <typename T> struct BufferView {
    T* ptr = nullptr;
    std::size_t count = 0;
    constexpr T* data() const noexcept { return ptr; }
    constexpr std::size_t size() const noexcept { return count; }
    constexpr std::size_t capacity() const noexcept { return count; }
    constexpr bool empty() const noexcept { return count == 0; }
    constexpr T& operator[](std::size_t i) const noexcept { return ptr[i]; }
    constexpr T* begin() const noexcept { return ptr; }
    constexpr T* end() const noexcept { return count == 0 ? ptr : ptr + count; }
};

// Borrowed capacity with a mutable active prefix. Metadata builders may reuse
// it like a vector, but growth beyond the caller's fixed span is forbidden.
// Elements are disposable arena objects; no element may require destruction.
template<class T>class WorkspaceArray {
    static_assert(std::is_trivially_destructible<T>::value,"workspace array elements must be disposable");
    BufferView<T> storage_{};std::size_t size_=0;
public:
    using value_type=T;
    WorkspaceArray()=default;
    explicit WorkspaceArray(DdlaHandle_t) noexcept{}
    void bind(BufferView<T> storage,std::size_t size) noexcept
    {assert(size<=storage.size());storage_=storage;size_=size;}
    T* data() const noexcept{return storage_.data();}
    std::size_t size() const noexcept{return size_;}
    std::size_t capacity() const noexcept{return storage_.size();}
    bool empty() const noexcept{return !size_;}
    T* begin() const noexcept{return data();}
    T* end() const noexcept{return data()?data()+size_:nullptr;}
    T& operator[](std::size_t i) const noexcept{assert(i<capacity());return data()[i];}
    T& front() const noexcept{assert(size_);return data()[0];}
    T& back() const noexcept{assert(size_);return data()[size_-1];}
    void reserve(std::size_t n) noexcept{assert(n<=capacity());}
    void clear() noexcept{size_=0;}
    void resize(std::size_t n) noexcept{assert(n<=capacity());size_=n;}
    void resize(std::size_t n,const T& value) noexcept
    {const auto old=size_;resize(n);for(auto i=old;i<n;++i)::new(static_cast<void*>(data()+i))T(value);}
    void assign(std::size_t n,const T& value) noexcept
    {resize(n);for(std::size_t i=0;i<n;++i)::new(static_cast<void*>(data()+i))T(value);}
    template<class It,std::enable_if_t<!std::is_integral<It>::value,int> = 0>
    void assign(It first,It last) noexcept{clear();for(;first!=last;++first)push_back(*first);}
    void push_back(const T& value) noexcept
    {assert(size_<capacity());::new(static_cast<void*>(data()+size_++))T(value);}
    void swap(WorkspaceArray& other) noexcept{std::swap(storage_,other.storage_);std::swap(size_,other.size_);}
};

struct WorkspaceSize {
    std::size_t host_bytes = 0;
    std::size_t backend_bytes = 0;
    std::size_t host_alignment = alignof(std::max_align_t);
    std::size_t backend_alignment = alignof(std::max_align_t);
};

struct Workspace {
    BufferView<std::byte> host;
    BufferView<std::byte> backend;
};

namespace detail {
inline WorkspaceSize workspace_max(const WorkspaceSize& a,const WorkspaceSize& b) noexcept
{
    return {std::max(a.host_bytes,b.host_bytes),std::max(a.backend_bytes,b.backend_bytes),
            std::max(a.host_alignment,b.host_alignment),std::max(a.backend_alignment,b.backend_alignment)};
}
inline std::size_t workspace_add(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::length_error("workspace: size overflow");
    return a + b;
}
inline std::size_t workspace_mul(std::size_t a, std::size_t b)
{
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::length_error("workspace: size overflow");
    return a * b;
}
inline std::size_t workspace_align(std::size_t n, std::size_t alignment)
{
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        throw std::invalid_argument("workspace: alignment must be a power of two");
    return workspace_add(n, alignment - 1) & ~(alignment - 1);
}

// Query-only layout builder: no storage, allocation, communication or data access.
class WorkspaceLayout {
    WorkspaceSize size_{};
    static std::size_t append(std::size_t& bytes, std::size_t& required_alignment,
                              std::size_t count, std::size_t element_bytes, std::size_t alignment)
    {
        if (count == 0) return 0;
        const auto offset = workspace_align(bytes, alignment);
        bytes = workspace_add(offset, workspace_mul(count, element_bytes));
        required_alignment = std::max(required_alignment, alignment);
        return offset;
    }
public:
    template <typename T> std::size_t host(std::size_t count, std::size_t alignment = alignof(T))
    { return append(size_.host_bytes, size_.host_alignment, count, sizeof(T), std::max(alignment, alignof(T))); }
    // Kernel scalar types can have stronger alignment than their host type
    // (notably thrust::complex versus std::complex). Keep mixed slices aligned.
    template <typename T> std::size_t backend(std::size_t count, std::size_t alignment = alignof(std::max_align_t))
    { return append(size_.backend_bytes, size_.backend_alignment, count, sizeof(T), std::max(alignment, alignof(T))); }
    WorkspaceSize size() const noexcept { return size_; }
};

template <typename T>
inline BufferView<T> workspace_at(BufferView<std::byte> bytes, std::size_t offset, std::size_t count) noexcept
{
    assert(offset <= bytes.size());
    assert(count <= (bytes.size() - offset) / sizeof(T));
    auto* p = bytes.data() == nullptr ? nullptr : bytes.data() + offset;
    assert(count == 0 || reinterpret_cast<std::uintptr_t>(p) % alignof(T) == 0);
    return {reinterpret_cast<T*>(p), count};
}
template <typename T>
inline void workspace_start_objects(BufferView<T> view) noexcept
{
    static_assert(std::is_trivially_destructible<T>::value, "workspace objects must not need destruction");
    // C++17 standard libraries do not consistently mark std::complex's
    // allocation-free default constructor noexcept. Callers use scalar/POD
    // scratch whose construction cannot allocate or throw.
    for (std::size_t i = 0; i < view.size(); ++i)
        ::new (static_cast<void*>(view.data() + i)) T;
}
inline std::byte* workspace_aligned(std::byte* p, std::size_t alignment) noexcept
{
    if (p == nullptr) return nullptr;
    auto n = reinterpret_cast<std::uintptr_t>(p);
    return reinterpret_cast<std::byte*>((n + alignment - 1) & ~(alignment - 1));
}
} // namespace detail

// Optional owner used by checked convenience calls and by callers that want
// reusable capacity. It is deliberately outside check/query/run.
class WorkspaceStorage {
    std::unique_ptr<std::max_align_t[]> host_;
    std::size_t host_capacity_ = 0;
    vector<std::max_align_t> backend_;
    WorkspaceSize size_{};
public:
    explicit WorkspaceStorage(DdlaHandle_t handle) : backend_(handle) {}
    WorkspaceStorage(const WorkspaceStorage&) = delete;
    WorkspaceStorage& operator=(const WorkspaceStorage&) = delete;
    WorkspaceStorage(WorkspaceStorage&&) noexcept = default;
    WorkspaceStorage& operator=(WorkspaceStorage&&) noexcept = default;

    void reserve(WorkspaceSize size)
    {
        // Validate both alignments before changing either owner.
        (void)detail::workspace_align(0, size.host_alignment);
        (void)detail::workspace_align(0, size.backend_alignment);
        const auto hb = size.host_bytes == 0 ? 0 : detail::workspace_add(size.host_bytes, size.host_alignment - 1);
        const auto db = size.backend_bytes == 0 ? 0 : detail::workspace_add(size.backend_bytes, size.backend_alignment - 1);
        constexpr auto unit = sizeof(std::max_align_t);
        const auto hn = hb / unit + (hb % unit != 0);
        const auto dn = db / unit + (db % unit != 0);
        if (hn > host_capacity_) {
            std::unique_ptr<std::max_align_t[]> next(new std::max_align_t[hn]);
            host_ = std::move(next);
            host_capacity_ = hn;
        }
        backend_.reserve(dn);
        size_ = size;
    }
    Workspace view() noexcept
    {
        return {{size_.host_bytes == 0 ? nullptr : detail::workspace_aligned(reinterpret_cast<std::byte*>(host_.get()), size_.host_alignment), size_.host_bytes},
                {size_.backend_bytes == 0 ? nullptr : detail::workspace_aligned(reinterpret_cast<std::byte*>(backend_.data()), size_.backend_alignment), size_.backend_bytes}};
    }
};

static_assert(std::is_trivially_copyable<WorkspaceSize>::value, "workspace sizes must not own resources");
static_assert(std::is_trivially_copyable<Workspace>::value, "workspace views must not own resources");
} // namespace ddla
#endif
