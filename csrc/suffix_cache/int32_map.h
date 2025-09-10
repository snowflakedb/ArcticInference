#pragma once
#include <cstdint>
#include <climits>
#include <new>
#include <utility>
#include <type_traits>
#include <iterator>
#include <cassert>

template <class T>
class Int32Map {
public:
    using iterator_value = std::pair<int, T&>;
    using const_iterator_value = std::pair<int, const T&>;

    Int32Map() = default;

    Int32Map(Int32Map&& o) noexcept
        : slots_(o.slots_), mask_(o.mask_), size_(o.size_), tombstones_(o.tombstones_) {
        o.slots_ = nullptr; o.mask_ = o.size_ = o.tombstones_ = 0;
    }
    Int32Map& operator=(Int32Map&& o) noexcept {
        if (this == &o) return *this;
        destroy_all_(); delete[] slots_;
        slots_ = o.slots_; o.slots_ = nullptr;
        mask_ = o.mask_;   o.mask_ = 0;
        size_ = o.size_;   o.size_ = 0;
        tombstones_ = o.tombstones_; o.tombstones_ = 0;
        return *this;
    }

    Int32Map(const Int32Map&) = delete;
    Int32Map& operator=(const Int32Map&) = delete;

    ~Int32Map() { destroy_all_(); delete[] slots_; }

    // ----- Minimal API -----
    bool empty() const noexcept { return size_ == 0; }
    uint32_t size() const noexcept { return size_; }

    bool contains(int key) const noexcept { return find_index_(key) != NPOS; }

    bool erase(int key) {
        if (!slots_) return false;
        uint32_t idx = find_index_(key);
        if (idx == NPOS) return false;
        value_ptr_(slots_[idx])->~T();
        slots_[idx].key = KEY_TOMBSTONE;
        --size_;
        ++tombstones_;
        maybe_cleanup_tombstones_();
        return true;
    }

    template <class... Args>
    T& emplace(int key, Args&&... args) {
        assert(key != KEY_EMPTY && key != KEY_TOMBSTONE && "reserved key");
        ensure_capacity_for_insert_();
        uint32_t idx;
        if (probe_insert_or_find_(key, idx)) {
            return *value_ptr_(slots_[idx]); // already present
        }
        slots_[idx].key = key;
        ::new (static_cast<void*>(&slots_[idx].storage)) T(std::forward<Args>(args)...);
        ++size_;
        return *value_ptr_(slots_[idx]);
    }

    // operator[]: default-construct if absent (T must be default-constructible)
    T& operator[](int key) {
        assert(key != KEY_EMPTY && key != KEY_TOMBSTONE && "reserved key");
        ensure_capacity_for_insert_();
        uint32_t idx;
        if (probe_insert_or_find_(key, idx)) {
            return *value_ptr_(slots_[idx]);
        }
        slots_[idx].key = key;
        ::new (static_cast<void*>(&slots_[idx].storage)) T();
        ++size_;
        return *value_ptr_(slots_[idx]);
    }

    // ----- Iteration (range-for) -----
    template <bool IsConst>
    class Iter {
        using Map  = std::conditional_t<IsConst, const Int32Map, Int32Map>;
        using ValR = std::conditional_t<IsConst, const T&, T&>;
        using Pair = std::pair<int, ValR>;
    public:
        using value_type        = Pair;
        using difference_type   = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        Iter() : m_(nullptr), i_(0) {}
        Iter(Map* m, uint32_t i) : m_(m), i_(i) { advance_(); }

        Pair operator*() const {
            auto& s = m_->slots_[i_];
            return { s.key, *m_->value_ptr_(s) };
        }
        Iter& operator++() { ++i_; advance_(); return *this; }
        bool operator==(const Iter& o) const { return m_ == o.m_ && i_ == o.i_; }
        bool operator!=(const Iter& o) const { return !(*this == o); }

    private:
        void advance_() {
            uint32_t c = m_->cap_();
            while (m_ && i_ < c && !m_->is_filled_(m_->slots_[i_].key)) ++i_;
        }
        Map* m_;
        uint32_t i_;
    };

    using iterator = Iter<false>;
    using const_iterator = Iter<true>;

    iterator begin() { return iterator(this, 0); }
    iterator end()   { return iterator(this, cap_()); }
    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end()   const { return const_iterator(this, cap_()); }
    const_iterator cbegin() const { return begin(); }
    const_iterator cend()   const { return end(); }

private:
    // ----- Constants (CAPITAL_STYLE) -----
    static constexpr int KEY_EMPTY     = INT_MIN;
    static constexpr int KEY_TOMBSTONE = INT_MIN + 1;

    static constexpr uint32_t MIN_CAPACITY = 1;   // power of two
    static constexpr uint32_t NPOS = 0xFFFFFFFFu;

    struct Slot {
        int key;
        typename std::aligned_storage<sizeof(T), alignof(T)>::type storage;
        Slot() noexcept : key(KEY_EMPTY) {}
    };

    // ----- Data (compact: 1 pointer + 3x uint32_t) -----
    Slot*    slots_       = nullptr; // 8 bytes on 64-bit
    uint32_t mask_        = 0;       // cap-1 (valid only if slots_ != nullptr)
    uint32_t size_        = 0;       // number of FILLED
    uint32_t tombstones_  = 0;       // number of TOMBSTONE

    // ----- Helpers -----
    uint32_t cap_() const noexcept { return slots_ ? (mask_ + 1u) : 0u; }

    static bool is_filled_(int k) noexcept { return k != KEY_EMPTY && k != KEY_TOMBSTONE; }

    static uint32_t next_pow2_(uint32_t x) {
        if (x <= 1) return 1;
        --x; x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
        return x + 1;
    }

    static uint32_t mix_hash_(int key) noexcept {
        // 32-bit mix (Murmur-inspired)
        uint32_t x = static_cast<uint32_t>(key);
        x ^= x >> 16; x *= 0x7feb352dU;
        x ^= x >> 15; x *= 0x846ca68bU;
        x ^= x >> 16;
        return x;
    }

    static T* value_ptr_(Slot& s) noexcept {
        return std::launder(reinterpret_cast<T*>(&s.storage));
    }
    static const T* value_ptr_(const Slot& s) noexcept {
        return std::launder(reinterpret_cast<const T*>(&s.storage));
    }

    void init_(uint32_t want) {
        uint32_t c = next_pow2_(want < MIN_CAPACITY ? MIN_CAPACITY : want);
        slots_ = new Slot[c];
        mask_ = c - 1;
        size_ = tombstones_ = 0;
    }

    void destroy_all_() noexcept {
        if (!slots_) return;
        uint32_t c = cap_();
        for (uint32_t i = 0; i < c; ++i) {
            if (is_filled_(slots_[i].key)) {
                value_ptr_(slots_[i])->~T();
                slots_[i].key = KEY_EMPTY;
            }
        }
        size_ = tombstones_ = 0;
    }

    uint32_t max_allowed_() const noexcept {
        // ~80% load: cap - cap/5
        uint32_t c = cap_();
        return c - (c / 5u);
    }

    void ensure_capacity_for_insert_() {
        if (!slots_) { init_(MIN_CAPACITY); return; }
        if (size_ + tombstones_ + 1 > max_allowed_()) {
            rehash_(cap_() * 2u);
        }
    }

    void maybe_cleanup_tombstones_() {
        uint32_t c = cap_();
        if (tombstones_ >= (c >> 2)) { // >= 25% tombstones
            rehash_(c);
        }
    }

    // Find exact key; returns index or NPOS.
    uint32_t find_index_(int key) const noexcept {
        if (!slots_) return NPOS;
        uint32_t c = cap_();
        uint32_t idx  = mix_hash_(key) & mask_;
        uint32_t step = 0;
        for (uint32_t probes = 0; probes < c; ++probes) {
            int k = slots_[idx].key;
            if (k == KEY_EMPTY) return NPOS;   // stop at first empty
            if (k == key) return idx;
            ++step;
            idx = (idx + step) & mask_;        // triangular probing
        }
        return NPOS;
    }

    // Either finds existing (true, idx_out set) or returns insert slot (false).
    bool probe_insert_or_find_(int key, uint32_t& idx_out) noexcept {
        uint32_t c = cap_();
        uint32_t idx  = mix_hash_(key) & mask_;
        uint32_t step = 0;
        uint32_t first_tomb = NPOS;

        for (uint32_t probes = 0; probes < c; ++probes) {
            int k = slots_[idx].key;
            if (k == KEY_EMPTY) {
                idx_out = (first_tomb != NPOS) ? first_tomb : idx;
                return false;
            }
            if (k == key) { idx_out = idx; return true; }
            if (k == KEY_TOMBSTONE && first_tomb == NPOS) first_tomb = idx;
            ++step;
            idx = (idx + step) & mask_;
        }
        idx_out = (first_tomb != NPOS) ? first_tomb : NPOS; // fallback
        return false;
    }

    template <class U>
    static void place_new_(Slot* arr, uint32_t cap, uint32_t mask, int key, U&& val) {
        uint32_t idx  = mix_hash_(key) & mask;
        uint32_t step = 0;
        for (uint32_t probes = 0; probes < cap; ++probes) {
            int k = arr[idx].key;
            if (k == KEY_EMPTY || k == KEY_TOMBSTONE) {
                arr[idx].key = key;
                ::new (static_cast<void*>(&arr[idx].storage)) T(std::forward<U>(val));
                return;
            }
            ++step;
            idx = (idx + step) & mask;
        }
        assert(false && "rehash placement failed");
    }

    void rehash_(uint32_t new_cap) {
        new_cap = next_pow2_(new_cap < MIN_CAPACITY ? MIN_CAPACITY : new_cap);
        Slot* fresh = new Slot[new_cap]; // keys default to KEY_EMPTY
        uint32_t new_mask = new_cap - 1;

        if (slots_) {
            uint32_t c = cap_();
            for (uint32_t i = 0; i < c; ++i) {
                auto& s = slots_[i];
                if (!is_filled_(s.key)) continue;
                int k = s.key;
                T* v  = value_ptr_(s);
                place_new_(fresh, new_cap, new_mask, k, std::move(*v));
                v->~T();
                s.key = KEY_EMPTY;
            }
            delete[] slots_;
        }

        slots_ = fresh;
        mask_ = new_mask;
        tombstones_ = 0; // cleaned
        // size_ unchanged
    }
};
