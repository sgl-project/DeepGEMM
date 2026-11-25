#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/legacy/layout/mqa_logits.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

// SM100 paged scheduler: metadata emits per-SM (q_token_idx, kv_split_idx) starts
// Device traversal walks chunk-outer / Q-block-inner tasks

namespace deep_gemm::legacy::sched {

// Per-request geometry accessor; this is where varlen and non-varlen diverge
template <uint32_t kNextN, bool kIsContextLens2D, bool kIsVarlen,
          uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t PAGE_KV>
struct RequestInfo {
    uint32_t q_token_start;       // request_q_token_start
    uint32_t num_q_tokens;        // request_num_q_tokens
    uint32_t num_q_blocks;        // request_num_q_blocks
    uint32_t num_kv_splits;       // request_num_kv_splits  = ceil(context_len / SPLIT_KV)
    uint32_t num_kv_pages;        // = ceil(context_len / PAGE_KV); page-level bound for the last partial split

    CUTLASS_DEVICE RequestInfo() = default;

    CUTLASS_DEVICE RequestInfo(const uint32_t& q_token_start, const uint32_t& num_q_tokens,
                               const uint32_t& context_len)
        : q_token_start(q_token_start), num_q_tokens(num_q_tokens),
          num_q_blocks(math::ceil_div(num_q_tokens, BLOCK_Q)),
          num_kv_splits(math::ceil_div(context_len, SPLIT_KV)),
          num_kv_pages(math::ceil_div(context_len, PAGE_KV)) {}

    // Resolve the request that starts at `q_token_idx`
    CUTLASS_DEVICE static RequestInfo from_q_token(const uint32_t& q_token_idx,
                                                   const uint32_t& num_q_tokens_total,
                                                   const uint32_t* context_lens,
                                                   const uint32_t* indices) {
        if constexpr (kIsVarlen) {
            // Varlen request = maximal run of equal `indices`
            const uint32_t request_id = indices[q_token_idx];
            uint32_t t = q_token_idx;
            while (t + 1 < num_q_tokens_total and indices[t + 1] == request_id)
                ++ t;
            return RequestInfo(q_token_idx, t - q_token_idx + 1, context_lens[t]);
        } else {
            // Regular grid: request = q_token_idx / next_n, next_n tokens each
            const uint32_t lens_idx = kIsContextLens2D ? q_token_idx + kNextN - 1 : q_token_idx / kNextN;
            return RequestInfo(q_token_idx, kNextN, context_lens[lens_idx]);
        }
    }

    // Average q-token partition across Q-blocks; returns both offset and count
    CUTLASS_DEVICE void get_q_block_span(const uint32_t& q, uint32_t& token_offset, uint32_t& num_tokens) const {
        const uint32_t base = num_q_tokens / num_q_blocks, rem = num_q_tokens % num_q_blocks;
        token_offset = q * base + (q < rem ? q : rem);
        num_tokens = base + (q < rem ? 1 : 0);
    }

    // block_table row for this request: request id for non-varlen, token row for varlen
    CUTLASS_DEVICE uint32_t get_block_table_row() const {
        if constexpr (kIsVarlen)
            return q_token_start;
        else
            return q_token_start / kNextN;
    }
};

inline constexpr uint32_t kNumMetadataThreads = 1024;
inline constexpr uint32_t kNumMetadataWarps = kNumMetadataThreads / 32;
DG_STATIC_ASSERT(kNumMetadataWarps <= 32 and kNumMetadataThreads % 32 == 0, "Invalid metadata thread count");

// Inclusive prefix sum
CUTLASS_DEVICE void metadata_prefix_scan(const uint32_t thread_idx, const uint32_t num_items,
                                         uint32_t* values, uint32_t* warp_sums) {
    const uint32_t num_items_per_thread = math::ceil_div(num_items, kNumMetadataThreads) | 1u;
    const uint32_t item_begin_idx = cute::min(thread_idx * num_items_per_thread, num_items);
    const uint32_t item_end_idx = cute::min(item_begin_idx + num_items_per_thread, num_items);
    uint32_t even_sum = 0, odd_sum = 0;
    uint32_t item_idx = item_begin_idx;
    for (; item_idx + 2 <= item_end_idx; item_idx += 2) {
        even_sum += values[item_idx];
        values[item_idx] = even_sum + odd_sum;
        odd_sum += values[item_idx + 1];
        values[item_idx + 1] = odd_sum + even_sum;
    }
    if (item_idx < item_end_idx) {
        even_sum += values[item_idx];
        values[item_idx] = even_sum + odd_sum;
    }
    const uint32_t thread_offset = math::cta_exclusive_sum<kNumMetadataThreads>(even_sum + odd_sum, warp_sums);
    for (item_idx = item_begin_idx; item_idx < item_end_idx; ++item_idx)
        values[item_idx] += thread_offset;
    __syncthreads();
}

// Balance work across SMs by request prefix sum
template <uint32_t kNextN, bool kIsContextLens2D, bool kIsVarlen,
          uint32_t SPLIT_KV, uint32_t kNumSMs>
CUTLASS_GLOBAL __launch_bounds__(kNumMetadataThreads, 1)
void sm100_paged_mqa_logits_metadata(const uint32_t num_requests,
                                     const uint32_t num_q_tokens_total,
                                     const uint32_t* context_lens,
                                     const uint32_t* indices,
                                     uint32_t* schedule_meta) {
    const uint32_t thread_idx = threadIdx.x;
    DG_DEVICE_ASSERT(blockDim.x == kNumMetadataThreads);
    cudaGridDependencySynchronize();  // wait for the primary kernel (CDP launch)

    extern __shared__ uint32_t smem[];
    const auto request_q_token_start_idx = smem;                            // [num_requests], varlen only
    const auto request_work_prefix = smem + (kIsVarlen ? num_requests : 0); // [num_requests]
    const auto warp_sums = request_work_prefix + num_requests;              // [kNumMetadataWarps]
    const auto num_logical_requests_shared = warp_sums + kNumMetadataWarps;

    uint32_t num_logical_requests;
    if constexpr (kIsVarlen) {
        // Extract request starts
        DG_DEVICE_ASSERT(reinterpret_cast<uintptr_t>(indices) % 8 == 0);
        const uint32_t num_tokens_per_thread = (math::ceil_div(num_q_tokens_total, kNumMetadataThreads * 2) | 1u) * 2;
        const uint32_t lo = cute::min(thread_idx * num_tokens_per_thread, num_q_tokens_total);
        const uint32_t hi = cute::min(lo + num_tokens_per_thread, num_q_tokens_total);
        uint32_t num_request_starts = 0;
        uint32_t prev_id = lo > 0 and lo < hi ? indices[lo - 1] : 0u;
        const auto indices_vec2 = reinterpret_cast<const uint2*>(indices);
        uint32_t token_idx = lo;
        for (; token_idx + 2 <= hi; token_idx += 2) {
            const uint2 ids = indices_vec2[token_idx / 2];
            const bool is_x_start = (token_idx == 0) or (ids.x != prev_id);
            const bool is_y_start = ids.y != ids.x;
            num_request_starts += is_x_start + is_y_start;
            prev_id = ids.y;
        }
        if (token_idx < hi) {
            const bool is_request_start = (token_idx == 0) or (indices[token_idx] != prev_id);
            num_request_starts += is_request_start;
        }
        uint32_t request_idx = math::cta_exclusive_sum<kNumMetadataThreads>(num_request_starts, warp_sums);
        prev_id = lo > 0 and lo < hi ? indices[lo - 1] : 0u;
        token_idx = lo;
        for (; token_idx + 2 <= hi; token_idx += 2) {
            const uint2 ids = indices_vec2[token_idx / 2];
            const bool is_x_start = (token_idx == 0) or (ids.x != prev_id);
            const bool is_y_start = ids.y != ids.x;
            if (is_x_start)
                request_q_token_start_idx[request_idx ++] = token_idx;
            if (is_y_start)
                request_q_token_start_idx[request_idx ++] = token_idx + 1;
            prev_id = ids.y;
        }
        if (token_idx < hi) {
            const bool is_request_start = (token_idx == 0) or (indices[token_idx] != prev_id);
            if (is_request_start)
                request_q_token_start_idx[request_idx ++] = token_idx;
        }
        if (thread_idx == kNumMetadataThreads - 1)
            *num_logical_requests_shared = request_idx;
        __syncthreads();
        num_logical_requests = *num_logical_requests_shared;
    } else {
        num_logical_requests = num_requests;
    }

    const auto get_request_info = [&](const uint32_t& request_idx,
                                      uint32_t& q_token_start_idx,
                                      uint32_t& num_q_tokens,
                                      uint32_t& context_len) {
        if constexpr (kIsVarlen) {
            q_token_start_idx = request_q_token_start_idx[request_idx];
            const uint32_t q_token_end_idx = request_idx + 1 < num_logical_requests ?
                                                 request_q_token_start_idx[request_idx + 1] : num_q_tokens_total;
            num_q_tokens = q_token_end_idx - q_token_start_idx;
            context_len = context_lens[q_token_end_idx - 1];
        } else {
            q_token_start_idx = request_idx * kNextN;
            num_q_tokens = kNextN;
            const uint32_t lens_idx = kIsContextLens2D ? request_idx * kNextN + kNextN - 1 : request_idx;
            context_len = context_lens[lens_idx];
        }
    };

    // Compute per-request work
    for (uint32_t request_idx = thread_idx; request_idx < num_logical_requests; request_idx += kNumMetadataThreads) {
        uint32_t q_token_start_idx, num_q_tokens, context_len;
        get_request_info(request_idx, q_token_start_idx, num_q_tokens, context_len);
        request_work_prefix[request_idx] = math::ceil_div(context_len, SPLIT_KV) * num_q_tokens;
    }
    __syncthreads();

    if (num_logical_requests > 0) {
        metadata_prefix_scan(thread_idx, num_logical_requests, request_work_prefix, warp_sums);
    }
    const uint32_t num_total_work = num_logical_requests > 0 ? request_work_prefix[num_logical_requests - 1] : 0u;

    // Partition work across SMs
    const uint32_t q = num_total_work / kNumSMs;
    const uint32_t rem = num_total_work % kNumSMs;
    for (uint32_t sm_idx = thread_idx; sm_idx <= kNumSMs; sm_idx += kNumMetadataThreads) {
        const uint32_t w = sm_idx * q + cute::min(sm_idx, rem);
        // Find the request containing `w`
        uint32_t lo = 0, hi = num_logical_requests;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if (request_work_prefix[mid] <= w)
                lo = mid + 1;
            else
                hi = mid;
        }
        const uint32_t request_idx = lo;
        uint32_t q_token_idx, kv_split_idx;
        if (request_idx < num_logical_requests) {
            const uint32_t w_in_request = w - (request_idx == 0 ? 0u : request_work_prefix[request_idx - 1]);
            uint32_t q_token_start_idx, num_q_tokens, context_len;
            get_request_info(request_idx, q_token_start_idx, num_q_tokens, context_len);
            // Align SM starts to request/split boundaries
            q_token_idx = q_token_start_idx;
            kv_split_idx = w_in_request / num_q_tokens;
        } else {
            // Tail sentinel: one-past-the-end
            q_token_idx = num_q_tokens_total;
            kv_split_idx = 0;
        }
        schedule_meta[sm_idx * 2] = q_token_idx;
        schedule_meta[sm_idx * 2 + 1] = kv_split_idx;
    }
}

// Localized schedule for a KV cache whose pages `[0, num_local_pages)` are homed in locality domain 0 and the rest in domain 1.
// A request belongs to the domain that holds most of its pages (of up to `kNumMaxPageSamples` evenly spaced ones), and each
// domain's work is balanced across its share of the slots. The schedule is static, i.e. the stock partition with slot `i`
// for CTA `i`, where claims would not pay off: below `min_num_localized_splits` KV splits, or when less than
// `kMinHomedWorkNum / kMinHomedWorkDen` of the work reads pages of its request's domain (pages from an allocator that
// ignores the domains). `min_num_localized_splits == 0` always claims.
template <uint32_t kNextN, uint32_t SPLIT_KV, uint32_t PAGE_KV, uint32_t kNumSlots>
CUTLASS_GLOBAL __launch_bounds__(kNumMetadataThreads, 1)
void sm100_paged_mqa_logits_localized_metadata(const uint32_t num_requests,
                                               const uint32_t min_num_localized_splits,
                                               const uint32_t num_local_pages,
                                               const uint32_t block_table_stride,
                                               const uint32_t* context_lens,
                                               const uint32_t* block_table,
                                               uint32_t* schedule_meta) {
    using Layout = layout::PagedMQALogitsLocalizedSchedule;
    constexpr uint32_t kNumMaxPageSamples = 32;
    constexpr uint32_t kMinHomedWorkNum = 7, kMinHomedWorkDen = 8;
    const uint32_t thread_idx = threadIdx.x;
    DG_DEVICE_ASSERT(blockDim.x == kNumMetadataThreads);
    cudaGridDependencySynchronize();

    extern __shared__ uint32_t smem[];
    const auto work_prefix = smem;                                  // [2][num_requests], per domain
    const auto request_domains = work_prefix + 2 * num_requests;    // [num_requests]
    const auto homed_work_prefix = request_domains + num_requests;  // [num_requests]
    const auto warp_sums = homed_work_prefix + num_requests;        // [kNumMetadataWarps]
    const uint32_t num_q_tokens_total = num_requests * kNextN;

    const auto get_context_len = [&](const uint32_t& request_idx) {
        return context_lens[request_idx * kNextN + kNextN - 1];
    };
    const auto get_work = [&](const uint32_t& request_idx) {
        return math::ceil_div(get_context_len(request_idx), SPLIT_KV) * kNextN;
    };

    // Total work decides whether claiming pays off; requests vote for their domain
    for (uint32_t request_idx = thread_idx; request_idx < num_requests; request_idx += kNumMetadataThreads) {
        const uint32_t work = get_work(request_idx);
        work_prefix[request_idx] = work;
        const uint32_t num_pages = math::ceil_div(get_context_len(request_idx), PAGE_KV);
        const uint32_t num_samples = cute::min(num_pages, kNumMaxPageSamples);
        const auto pages = block_table + request_idx * static_cast<uint64_t>(block_table_stride);
        uint32_t num_remote_samples = 0;
        for (uint32_t i = 0; i < num_samples; ++ i)
            num_remote_samples += pages[static_cast<uint64_t>(i) * num_pages / num_samples] >= num_local_pages;
        const uint32_t domain_idx = num_remote_samples * 2 > num_samples ? 1u : 0u;
        request_domains[request_idx] = domain_idx;
        // The share of the request's work that reads its domain's pages
        const uint32_t num_homed_samples = domain_idx == 1 ? num_remote_samples : num_samples - num_remote_samples;
        homed_work_prefix[request_idx] = num_samples == 0 ? work :
            static_cast<uint32_t>(static_cast<uint64_t>(work) * num_homed_samples / num_samples);
    }
    __syncthreads();
    if (num_requests > 0) {
        metadata_prefix_scan(thread_idx, num_requests, work_prefix, warp_sums);
        metadata_prefix_scan(thread_idx, num_requests, homed_work_prefix, warp_sums);
    }
    const uint32_t num_total_work = num_requests > 0 ? work_prefix[num_requests - 1] : 0u;
    const uint32_t num_homed_work = num_requests > 0 ? homed_work_prefix[num_requests - 1] : 0u;
    const bool is_static = min_num_localized_splits > 0 and
        (num_total_work / kNextN < min_num_localized_splits or
         static_cast<uint64_t>(num_homed_work) * kMinHomedWorkDen < static_cast<uint64_t>(num_total_work) * kMinHomedWorkNum);
    __syncthreads();

    // Per-domain work; a static schedule has all of it in domain 0
    for (uint32_t request_idx = thread_idx; request_idx < num_requests; request_idx += kNumMetadataThreads) {
        const uint32_t domain_idx = is_static ? 0u : request_domains[request_idx];
        const uint32_t work = get_work(request_idx);
        work_prefix[request_idx] = domain_idx == 0 ? work : 0u;
        work_prefix[num_requests + request_idx] = domain_idx == 1 ? work : 0u;
    }
    __syncthreads();
    uint32_t num_domain_work[2] = {0u, 0u};
    if (num_requests > 0) {
        metadata_prefix_scan(thread_idx, num_requests, work_prefix, warp_sums);
        metadata_prefix_scan(thread_idx, num_requests, work_prefix + num_requests, warp_sums);
        num_domain_work[0] = work_prefix[num_requests - 1];
        num_domain_work[1] = work_prefix[2 * num_requests - 1];
    }

    // Slots in proportion to the work, at least one for a domain with any
    uint32_t num_local_slots = kNumSlots;
    if (num_domain_work[1] > 0) {
        const uint64_t rounded = (static_cast<uint64_t>(num_domain_work[0]) * kNumSlots + num_total_work / 2) / num_total_work;
        num_local_slots = cute::min(cute::max(static_cast<uint32_t>(rounded), num_domain_work[0] > 0 ? 1u : 0u), kNumSlots - 1);
    }

    // (q_token_idx, kv_split_idx) of work unit `w` of a domain, as in the stock schedule
    const auto get_start = [&](const uint32_t* prefix, const uint32_t& w, uint32_t& q_token_idx, uint32_t& kv_split_idx) {
        uint32_t lo = 0, hi = num_requests;
        while (lo < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if (prefix[mid] <= w)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo < num_requests) {
            q_token_idx = lo * kNextN;
            kv_split_idx = (w - (lo == 0 ? 0u : prefix[lo - 1])) / kNextN;
        } else {
            // Tail sentinel: one-past-the-end
            q_token_idx = num_q_tokens_total;
            kv_split_idx = 0;
        }
    };
    for (uint32_t slot_idx = thread_idx; slot_idx <= kNumSlots; slot_idx += kNumMetadataThreads) {
        uint32_t words[4] = {num_q_tokens_total, 0u, num_q_tokens_total, 0u};
        if (slot_idx < kNumSlots) {
            const uint32_t domain_idx = slot_idx < num_local_slots ? 0u : 1u;
            const uint32_t num_slots = domain_idx == 0 ? num_local_slots : kNumSlots - num_local_slots;
            const uint32_t i = domain_idx == 0 ? slot_idx : slot_idx - num_local_slots;
            const uint32_t q = num_domain_work[domain_idx] / num_slots, rem = num_domain_work[domain_idx] % num_slots;
            const auto prefix = work_prefix + domain_idx * num_requests;
            get_start(prefix, i * q + cute::min(i, rem), words[0], words[1]);
            get_start(prefix, (i + 1) * q + cute::min(i + 1, rem), words[2], words[3]);
            DG_DEVICE_ASSERT(words[1] <= Layout::kSlotSplitMask);
            words[1] |= (is_static ? Layout::kSlotFlagStatic | Layout::kSlotFlagAllDomains : 0u) |
                        (domain_idx != 0 ? Layout::kSlotFlagDomain : 0u);
        }
        #pragma unroll
        for (uint32_t i = 0; i < 4; ++ i)
            schedule_meta[slot_idx * 4 + i] = words[i];
    }
    const auto request_domain_words = schedule_meta + Layout::get_request_domains_offset(kNumSlots);
    for (uint32_t word_idx = thread_idx; word_idx * 32 < num_requests; word_idx += kNumMetadataThreads) {
        uint32_t word = 0;
        for (uint32_t i = 0; i < 32 and word_idx * 32 + i < num_requests; ++ i)
            word |= (is_static ? 0u : request_domains[word_idx * 32 + i]) << i;
        request_domain_words[word_idx] = word;
    }
    if (thread_idx == 0) {
        schedule_meta[Layout::get_counter_offset(kNumSlots)] = 0;
        schedule_meta[Layout::get_counter_offset(kNumSlots) + 1] = 0;
        schedule_meta[Layout::get_num_local_slots_offset(kNumSlots)] = num_local_slots;
        schedule_meta[Layout::get_flags_offset(kNumSlots)] = is_static ? Layout::kFlagStatic : 0u;
    }
}

// Device scheduler walks this SM's schedule range and implements SchedulerConcept
// All specialized warps instantiate it and advance through the same task sequence
template <bool kHasIndices> struct SM100IndicesStorage { const uint32_t* indices; };
template <> struct SM100IndicesStorage<false> {};

// A localized walk only visits the requests of its slot's locality domain, or all of them in a static schedule
template <bool kIsLocalized> struct SM100RequestDomainStorage {
    static constexpr uint32_t kAllDomains = 2;
    const uint32_t* request_domain_words;
    uint32_t domain_idx;
};
template <> struct SM100RequestDomainStorage<false> {};

template <uint32_t kNextN, bool kIsContextLens2D, bool kIsVarlen,
          uint32_t kNumHeads, uint32_t SPLIT_KV, uint32_t PAGE_KV, uint32_t kSplitsPerChunk,
          bool kIsLocalized = false>
struct SM100PagedMQALogitsScheduler : SM100IndicesStorage<kIsVarlen>, SM100RequestDomainStorage<kIsLocalized> {
    // SchedulerConcept descriptors
    static constexpr bool kIsPaged = true;
    static constexpr bool kIsVarlenLayout = kIsVarlen;
    static constexpr bool kIsContextLens2DLayout = kIsContextLens2D;
    static constexpr uint32_t kTokensPerRequest = kNextN;
    static constexpr bool kHasPartialBlock = true;
    static constexpr uint32_t kPageKV = PAGE_KV;
    static constexpr uint32_t kNumPagesPerSplit = SPLIT_KV / PAGE_KV;
    static constexpr uint32_t BLOCK_Q = 128 / kNumHeads;

    using Info = RequestInfo<kNextN, kIsContextLens2D, kIsVarlen, BLOCK_Q, SPLIT_KV, PAGE_KV>;

    const uint32_t* context_lens;
    const uint32_t* block_table;
    uint32_t block_table_stride;
    uint32_t num_q_tokens_total;

    // Walk state
    Info cur;                           // current request geometry
    uint32_t cur_kv_split_base;         // current chunk start (request-internal split)
    uint32_t cur_q_block_in_request;    // current Q-block within the request
    uint32_t end_q_token_idx, end_kv_split_idx;
    bool done;

    // Geometry stashed by `next_q_block` for the accessors below
    uint32_t cur_block_table_row;       // request's block-table row
    uint32_t cur_q_block_token_base;    // global first-token row of this Q-block
    uint32_t cur_num_block_tokens;      // valid tokens in this Q-block
    uint32_t cur_request_num_kv_pages;  // ceil(context_len / PAGE_KV); bound for last partial split

    CUTLASS_DEVICE const uint32_t* get_indices() const {
        if constexpr (kIsVarlen)
            return this->indices;
        return nullptr;
    }

    CUTLASS_DEVICE SM100PagedMQALogitsScheduler(const uint32_t& sm_idx,
                                                const uint32_t* context_lens,
                                                const uint32_t* schedule_meta,
                                                const uint32_t* indices,
                                                const uint32_t* block_table,
                                                const uint32_t& block_table_stride,
                                                const uint32_t& num_q_tokens_total) {
        this->context_lens = context_lens;
        this->block_table = block_table;
        this->block_table_stride = block_table_stride;
        this->num_q_tokens_total = num_q_tokens_total;
        if constexpr (kIsVarlen)
            this->indices = indices;

        auto start = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx];
        const auto end = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx + 1];
        if constexpr (kIsLocalized) {
            // The start split of a localized slot carries the slot's flags
            // NOTES: the grid has one CTA per slot
            using Layout = layout::PagedMQALogitsLocalizedSchedule;
            this->request_domain_words = schedule_meta + Layout::get_request_domains_offset(gridDim.x);
            this->domain_idx = (start.y & Layout::kSlotFlagAllDomains) != 0 ? this->kAllDomains :
                               ((start.y & Layout::kSlotFlagDomain) != 0 ? 1u : 0u);
            start.y &= Layout::kSlotSplitMask;
        }
        end_q_token_idx = end.x;
        end_kv_split_idx = end.y;

        cur_kv_split_base = start.y;
        cur_q_block_in_request = 0;
        done = (start.x >= num_q_tokens_total) or
               (start.x == end_q_token_idx and start.y >= end_kv_split_idx);
        if (not done)
            cur = Info::from_q_token(start.x, num_q_tokens_total, context_lens, get_indices());

        cur_block_table_row = 0;
        cur_q_block_token_base = 0;
        cur_num_block_tokens = 1;
        cur_request_num_kv_pages = 0;
    }

    // First request at or after `q_token_idx` that this walk visits
    CUTLASS_DEVICE uint32_t get_next_request_q_token(uint32_t q_token_idx) const {
        if constexpr (kIsLocalized) {
            DG_STATIC_ASSERT(not kIsVarlen, "Localized schedules have a regular request grid");
            while (this->domain_idx != this->kAllDomains and q_token_idx < num_q_tokens_total) {
                const uint32_t request_idx = q_token_idx / kNextN;
                if (((this->request_domain_words[request_idx / 32] >> (request_idx % 32)) & 1u) == this->domain_idx)
                    break;
                q_token_idx += kNextN;
            }
        }
        return q_token_idx;
    }

    // Exclusive split bound for the current request, clamped at the next SM start
    CUTLASS_DEVICE uint32_t get_cur_kv_split_upper() const {
        return (cur.q_token_start == end_q_token_idx) ? end_kv_split_idx : cur.num_kv_splits;
    }

    // Emit the next (Q-block, chunk) task and stash its addressing geometry
    template <bool kIsCompressedLogits = false>
    CUTLASS_DEVICE bool next_q_block(uint32_t& q_block_idx, uint32_t& kv_split_base, uint32_t& num_kv_splits,
                                     uint32_t* = nullptr, uint32_t* = nullptr) {
        q_block_idx = 0;  // addressing uses stashed state
        if (done)
            return false;

        // Capture emitted task geometry before advancing state
        const uint32_t upper = get_cur_kv_split_upper();
        cur_block_table_row = cur.get_block_table_row();
        uint32_t q_block_token_offset, q_block_num_tokens;
        cur.get_q_block_span(cur_q_block_in_request, q_block_token_offset, q_block_num_tokens);
        cur_q_block_token_base = cur.q_token_start + q_block_token_offset;
        cur_num_block_tokens = q_block_num_tokens;
        cur_request_num_kv_pages = cur.num_kv_pages;
        kv_split_base = cur_kv_split_base;
        const uint32_t remaining = upper - cur_kv_split_base;   // upper > cur_kv_split_base (guarded by `done`)
        num_kv_splits = (cur.num_q_blocks == 1) ? remaining
                                               : (remaining < kSplitsPerChunk ? remaining : kSplitsPerChunk);

        // Advance in Q-block, chunk, request order
        ++ cur_q_block_in_request;
        if (cur_q_block_in_request == cur.num_q_blocks) {
            cur_q_block_in_request = 0;
            cur_kv_split_base += num_kv_splits;
            if (cur_kv_split_base >= upper) {
                if (cur.q_token_start == end_q_token_idx) {
                    // Reached the next SM's start
                    done = true;
                } else {
                    // Move to next request owned from split 0
                    const uint32_t next_q_token = get_next_request_q_token(cur.q_token_start + cur.num_q_tokens);
                    cur_kv_split_base = 0;
                    if (next_q_token >= num_q_tokens_total)
                        done = true;
                    else {
                        cur = Info::from_q_token(next_q_token, num_q_tokens_total, context_lens, get_indices());
                        // The new request may already be this SM's end
                        if (cur.q_token_start == end_q_token_idx and end_kv_split_idx == 0)
                            done = true;
                    }
                }
            }
        }
        return true;
    }

    CUTLASS_DEVICE uint32_t get_num_block_tokens(const uint32_t&) const {
        return cur_num_block_tokens;
    }

    CUTLASS_DEVICE uint32_t get_q_tma_token_base(const uint32_t&) const {
        return cur_q_block_token_base;
    }

    CUTLASS_DEVICE static uint32_t get_kv_tma_offset(const uint32_t& kv_split_base, const uint32_t& kv_split_idx) {
        return (kv_split_base + kv_split_idx) * SPLIT_KV;
    }

    CUTLASS_DEVICE uint32_t get_kv_page_coord_by_page_offset(const uint32_t& page_offset) const {
        if (page_offset >= cur_request_num_kv_pages)
            return 0;
        const auto block_table_offset = cur_block_table_row * static_cast<uint64_t>(block_table_stride);
        return block_table[block_table_offset + page_offset];
    }

    CUTLASS_DEVICE uint32_t get_logits_row(const uint32_t&, const uint32_t& token_idx) const {
        return cur_q_block_token_base + token_idx;
    }

    CUTLASS_DEVICE uint32_t get_logits_col(const uint32_t& kv_split_base,
                                           const uint32_t& kv_split_idx,
                                           const uint32_t& math_thread_idx) const {
        return (kv_split_base + kv_split_idx) * SPLIT_KV + math_thread_idx;
    }
};

// Walks the slots that this CTA claims from a localized schedule. Slots `[0, num_local_slots)` hold domain 0's work and
// the rest domain 1's; domain-0 CTAs claim from the front and domain-1 CTAs from the back of one counter, so a CTA
// reads its own domain's pages while there are any and the other domain's after that. The KV loader warp claims,
// since it runs ahead of the pipeline, and publishes the slot sequence in a shared ring for the other warps.
// Every (request, KV split) is still computed exactly once, by the unchanged main loop.
template <typename Inner, uint32_t kNumSlotsPerRing, uint32_t kClaimerWarpIdx>
struct SM100PagedMQALogitsClaimingScheduler : Inner {
    using Layout = layout::PagedMQALogitsLocalizedSchedule;
    // Shared words: [kNumSlotsPerRing] slots (-1 when none is left), the number published, the SM's domain
    static constexpr uint32_t kNumRingWords = kNumSlotsPerRing + 2;
    // `domain_idx` of a walk that claims nothing more
    static constexpr uint32_t kEnded = Inner::kAllDomains + 1;

    const uint32_t* schedule_meta;
    volatile int* ring;
    uint32_t num_acquired;

    // The walk starts exhausted
    // NOTES: slot `i` owns the pair of metadata entries starting at `2 * i`; this CTA has read slot `cta_idx` already
    CUTLASS_DEVICE SM100PagedMQALogitsClaimingScheduler(const uint32_t& cta_idx,
                                                        const uint32_t* context_lens,
                                                        const uint32_t* schedule_meta,
                                                        const uint32_t* block_table,
                                                        const uint32_t& block_table_stride,
                                                        const uint32_t& num_q_tokens_total,
                                                        volatile int* ring)
        : Inner(cta_idx * 2, context_lens, schedule_meta, nullptr,
                block_table, block_table_stride, num_q_tokens_total),
          schedule_meta(schedule_meta), ring(ring), num_acquired(0) {
        this->done = true;
    }

    // One claim on the two-ended counter (low word: domain 0's claims, high word: domain 1's)
    CUTLASS_DEVICE static int claim(const uint32_t* schedule_meta, const uint32_t& sm_domain_idx) {
        const uint32_t num_slots = gridDim.x, num_ctas = gridDim.x;
        const auto counter = reinterpret_cast<const uint64_t*>(schedule_meta + Layout::get_counter_offset(num_slots));
        const uint64_t count = ptx::atomic_add(counter, sm_domain_idx == 0 ? 1ull : (1ull << 32));
        const uint32_t num_front = static_cast<uint32_t>(count), num_back = static_cast<uint32_t>(count >> 32);
        // Every CTA fails exactly once, so the last claim of a launch leaves the counter ready for the next one
        if (num_front + num_back == num_slots + num_ctas - 1)
            *const_cast<uint64_t*>(counter) = 0;
        if (num_front + num_back >= num_slots)
            return -1;
        return static_cast<int>(sm_domain_idx == 0 ? num_front : num_slots - 1 - num_back);
    }

    // Claim the next slot with any work and publish it as slot `idx` of this CTA's sequence (-1 when none is left)
    CUTLASS_DEVICE static void claim_and_publish(const uint32_t* schedule_meta, volatile int* ring,
                                                 const uint32_t& idx, const uint32_t& num_q_tokens_total) {
        const uint32_t sm_domain_idx = static_cast<uint32_t>(ring[kNumSlotsPerRing + 1]);
        int slot_idx;
        while ((slot_idx = claim(schedule_meta, sm_domain_idx)) >= 0) {
            const auto start = reinterpret_cast<const uint2*>(schedule_meta)[slot_idx * 2];
            const auto end = reinterpret_cast<const uint2*>(schedule_meta)[slot_idx * 2 + 1];
            const bool empty = start.x >= num_q_tokens_total or
                               (start.x == end.x and (start.y & Layout::kSlotSplitMask) >= end.y);
            if (not empty)
                break;
        }
        ring[idx % kNumSlotsPerRing] = slot_idx;
        ring[kNumSlotsPerRing] = static_cast<int>(idx + 1);
    }

    // At kernel entry, before the first CTA-wide barrier: the first claim overlaps with the CTA's setup
    // NOTES: only once the grid dependency, which may be the schedule's writer, has completed
    CUTLASS_DEVICE static void init_ring(const uint32_t* schedule_meta, volatile int* ring,
                                         const uint32_t& sm_domain_idx, const uint32_t& num_q_tokens_total) {
        if (threadIdx.x == kClaimerWarpIdx * 32) {
            ring[kNumSlotsPerRing + 1] = static_cast<int>(sm_domain_idx);
            claim_and_publish(schedule_meta, ring, 0, num_q_tokens_total);
        }
    }

    // Slot `idx` of this CTA's sequence, or -1
    CUTLASS_DEVICE static int acquire(const uint32_t* schedule_meta, volatile int* ring,
                                      const uint32_t& idx, const uint32_t& num_q_tokens_total) {
        if (idx > 0 and threadIdx.x / 32 == kClaimerWarpIdx) {
            if (ptx::get_lane_idx() == 0)
                claim_and_publish(schedule_meta, ring, idx, num_q_tokens_total);
            __syncwarp();
        } else {
            while (ring[kNumSlotsPerRing] <= static_cast<int>(idx)) {}
        }
        return ring[idx % kNumSlotsPerRing];
    }

    template <bool kIsCompressedLogits = false>
    CUTLASS_DEVICE bool next_q_block(uint32_t& q_block_idx, uint32_t& kv_split_base, uint32_t& num_kv_splits,
                                     uint32_t* seq_k_start = nullptr, uint32_t* seq_k_end = nullptr) {
        while (true) {
            if (Inner::template next_q_block<kIsCompressedLogits>(q_block_idx, kv_split_base, num_kv_splits, seq_k_start, seq_k_end))
                return true;
            if (this->domain_idx == kEnded)
                return false;
            const int slot_idx = acquire(schedule_meta, ring, num_acquired ++, this->num_q_tokens_total);
            if (slot_idx < 0) {
                this->domain_idx = kEnded;
                return false;
            }
            static_cast<Inner&>(*this) = Inner(static_cast<uint32_t>(slot_idx) * 2, this->context_lens, schedule_meta, nullptr,
                                               this->block_table, this->block_table_stride, this->num_q_tokens_total);
        }
    }
};

} // namespace deep_gemm::legacy::sched
