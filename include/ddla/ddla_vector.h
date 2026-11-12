#ifndef DDLA_VECTOR_H
#define DDLA_VECTOR_H

#include <ddla/ddla_handle_t.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ddla {

/// Contiguous, uninitialized storage in the handle's backend memory.
///
/// CPU handles allocate host memory; GPU handles allocate device memory. The
/// handle must outlive the vector. Allocation, copies on growth, and release
/// use the handle's queue, without an additional synchronization.
///
/// T must be trivially copyable. Unlike std::vector, new elements are not
/// initialized and there is no host element access: pass data() to DDLA
/// operations, or use ddlaMemcpy and synchronize before reading on the host.
/// Ownership is movable, but not copyable. Growing preserves existing values;
/// shrinking and clear() retain the allocation for reuse.
/// reset() releases storage with error checking; destruction never throws.
template <typename T>
class vector {
    static_assert(std::is_trivially_copyable<T>::value,
                  "ddla::vector requires a trivially copyable element type");
    static_assert(!std::is_const<T>::value && !std::is_volatile<T>::value,
                  "ddla::vector requires an unqualified element type");
    static_assert(alignof(T) <= alignof(std::max_align_t),
                  "ddla::vector does not support over-aligned element types");

    DdlaHandle_t handle_;
    T* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t capacity_ = 0;

public:
    explicit vector(DdlaHandle_t handle, std::size_t count = 0) : handle_(handle)
    {
        if (handle == nullptr) throw std::invalid_argument("ddla::vector: null handle");
        resize(count);
    }

    ~vector() noexcept
    {
        if (data_ != nullptr) (void)ddlaFree(handle_, data_);
    }

    vector(const vector&) = delete;
    vector& operator=(const vector&) = delete;

    vector(vector&& other) noexcept
        : handle_(other.handle_), data_(other.data_),
          size_(other.size_), capacity_(other.capacity_)
    {
        other.data_ = nullptr;
        other.size_ = other.capacity_ = 0;
    }

    vector& operator=(vector&& other) noexcept
    {
        if (this != &other) {
            vector next(std::move(other));
            swap(next);
        }
        return *this;
    }

    DdlaHandle_t handle() const noexcept { return handle_; }
    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return size_ == 0; }
    std::size_t max_size() const noexcept
    {
        return std::numeric_limits<std::size_t>::max() / sizeof(T);
    }

    void reserve(std::size_t count)
    {
        if (count > max_size()) throw std::length_error("ddla::vector: allocation size overflow");
        if (count <= capacity_) return;
        vector next(handle_);
        DDLA_CHECK(ddlaMalloc(handle_, &next.data_, count * sizeof(T)));
        if (size_ != 0)
            DDLA_CHECK(ddlaMemcpy(handle_, next.data_, data_, size_ * sizeof(T)));
        DDLA_CHECK(ddlaFree(handle_, data_));
        data_ = next.data_;
        next.data_ = nullptr;
        capacity_ = count;
    }

    void resize(std::size_t count)
    {
        reserve(count);
        size_ = count;
    }

    void clear() noexcept { size_ = 0; }

    void reset()
    {
        DDLA_CHECK(ddlaFree(handle_, data_));
        data_ = nullptr;
        size_ = capacity_ = 0;
    }

    void swap(vector& other) noexcept
    {
        using std::swap;
        swap(handle_, other.handle_);
        swap(data_, other.data_);
        swap(size_, other.size_);
        swap(capacity_, other.capacity_);
    }
};

} // namespace ddla

#endif // DDLA_VECTOR_H
