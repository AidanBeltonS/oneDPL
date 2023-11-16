// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_parallel_backend_sycl_scan_H
#define _ONEDPL_parallel_backend_sycl_scan_H

#include <cstdint>
#include <sycl/sycl.hpp>

namespace oneapi::dpl::experimental::kt
{

inline namespace igpu {

constexpr ::std::size_t SUBGROUP_SIZE = 32;

template <typename _T>
struct scan_memory
{
    using _AtomicT = ::std::uint32_t;
    using _AtomicRefT = sycl::atomic_ref<_AtomicT, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>;
    using _AtomicCounterRefT = sycl::atomic_ref<_AtomicT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>;

    static constexpr _AtomicT NOT_READY = 0;
    static constexpr _AtomicT PARTIAL_MASK = 1;
    static constexpr _AtomicT FULL_MASK = 2;
    static constexpr _AtomicT OUT_OF_BOUNDS = 4;

    static constexpr ::std::size_t padding = SUBGROUP_SIZE;

    scan_memory(::std::size_t tile_id, char* scratch, ::std::size_t num_wgs)
        : atomic_flag(*(reinterpret_cast<_AtomicT*>(scratch) + tile_id + padding)), scratch(scratch),
          num_elements(get_num_elements(num_wgs))
    {
        ::std::size_t atomic_bytes = (num_elements + 1) * sizeof(_AtomicT);
        ::std::size_t align_extra_mem = alignof(_T) - (atomic_bytes % alignof(_T));

        flags = reinterpret_cast<_AtomicT*>(scratch);
        tile_values = reinterpret_cast<_T*>(scratch + atomic_bytes + align_extra_mem);

        scanned_partial_value = tile_values + tile_id + padding;
        scanned_full_value = tile_values + tile_id + padding + num_elements;
    }

    void
    set_partial(_T val)
    {
        (*scanned_partial_value) = val;
        atomic_flag.store(PARTIAL_MASK);
    }

    void
    set_full(_T val)
    {
        (*scanned_full_value) = val;
        atomic_flag.store(FULL_MASK);
    }

    _AtomicT
    load_flag(::std::size_t tile_id) const
    {
        _AtomicRefT flag(*(flags + tile_id + padding));
        return flag.load();
    }

    _T
    get_value(::std::size_t tile_id, _AtomicT flag) const
    {
        ::std::size_t offset = tile_id + padding + num_elements * is_full(flag);
        return tile_values[offset];
    }

    static char*
    allocate_memory(::std::size_t num_wgs, sycl::queue queue)
    {
        ::std::size_t num_elements = padding + num_wgs;
        ::std::size_t atomic_bytes = (num_elements + 1) * sizeof(_AtomicT);
        ::std::size_t align_extra_mem = alignof(_T) - (atomic_bytes % alignof(_T));
        ::std::size_t tile_sums_bytes = (2 * num_elements) * sizeof(_T);
        ::std::size_t memory_size = atomic_bytes + align_extra_mem + tile_sums_bytes;

        return sycl::malloc_device<char>(memory_size, queue);
    }

    static sycl::event
    async_free(char* scratch, sycl::queue& queue, sycl::event& dependencies)
    {
        return queue.submit(
            [=](sycl::handler& hdl)
            {
                hdl.depends_on(dependencies);
                hdl.host_task([=]() { sycl::free(scratch, queue); });
            });
    }

    static _AtomicT
    load_counter(::std::size_t num_wgs, char* scratch)
    {
        ::std::size_t num_elements = padding + num_wgs;
        _AtomicCounterRefT tile_counter(*(reinterpret_cast<_AtomicT*>(scratch) + num_elements));

        return tile_counter.fetch_add(1);
    }

    static ::std::size_t
    get_num_elements(::std::size_t num_wgs)
    {
        return padding + num_wgs;
    }

    static bool
    is_ready(_AtomicT flag)
    {
        return flag != NOT_READY;
    }

    static bool
    is_full(_AtomicT flag)
    {
        return flag == FULL_MASK;
    }

    static bool
    is_out_of_bounds(_AtomicT flag)
    {
        return flag == OUT_OF_BOUNDS;
    }

    _AtomicRefT atomic_flag;
    _T* scanned_partial_value;
    _T* scanned_full_value;

    ::std::size_t num_elements;
    char* scratch;
    _AtomicT* flags;
    _T* tile_values;
};

struct cooperative_lookback
{

    template <typename _T, typename _Subgroup, typename BinOp, template <typename> typename scan_memory>
    _T
    operator()(std::uint32_t tile_id, const _Subgroup& subgroup, BinOp bin_op, scan_memory<_T> memory)
    {
        using AtomicT = typename scan_memory<_T>::_AtomicT;

        _T sum = 0;
        int offset = -1;
        int i = 0;
        int local_id = subgroup.get_local_id();

        for (int tile = static_cast<int>(tile_id) + offset; tile >= 0; tile -= SUBGROUP_SIZE)
        {
            AtomicT flag;
            do
            {
                flag = memory.load_flag(tile - local_id);
            } while (!sycl::all_of_group(subgroup, scan_memory<_T>::is_ready(flag))); // Loop till all ready

            bool is_full = scan_memory<_T>::is_full(flag);
            auto is_full_ballot = sycl::ext::oneapi::group_ballot(subgroup, is_full);
            auto lowest_item_with_full = is_full_ballot.find_low();

            // TODO: Use identity_fn for out of bounds values
            _T contribution = local_id <= lowest_item_with_full && (!scan_memory<_T>::is_out_of_bounds(flag))
                                  ? memory.get_value(tile - local_id, flag)
                                  : _T{0};

            // Sum all of the partial results from the tiles found, as well as the full contribution from the closest tile (if any)
            sum += sycl::reduce_over_group(subgroup, contribution, bin_op);

            // If we found a full value, we can stop looking at previous tiles. Otherwise,
            // keep going through tiles until we either find a full tile or we've completely
            // recomputed the prefix using partial values
            if (is_full_ballot.any())
                break;

        }

        return sum;
    }
};

template <typename _KernelParam, bool _Inclusive, typename _InRange, typename _OutRange, typename _BinaryOp>
void
single_pass_scan_impl(sycl::queue __queue, _InRange&& __in_rng, _OutRange&& __out_rng, _BinaryOp __binary_op)
{
    using _Type = oneapi::dpl::__internal::__value_t<_InRange>;

    static_assert(_Inclusive, "Single-pass scan only available for inclusive scan");

    const ::std::size_t n = __in_rng.size();

    constexpr ::std::size_t wgsize = _KernelParam::workgroup_size;
    constexpr ::std::size_t elems_per_workitem = _KernelParam::elems_per_workitem;

    // Avoid non_uniform n by padding up to a multiple of wgsize
    ::std::uint32_t elems_in_tile = wgsize * elems_per_workitem;
    ::std::size_t num_wgs = oneapi::dpl::__internal::__dpl_ceiling_div(n, elems_in_tile);
    ::std::size_t num_workitems = num_wgs * wgsize;

    char* scratch = scan_memory<_Type>::allocate_memory(num_wgs, __queue);

    // FIX: can we get rid of this section, i.e. hide the dynamic_tile_id_counter, hide initialization details
    ::std::size_t status_flags_elems = scan_memory<_Type>::get_num_elements(num_wgs) + 1;
    ::std::size_t fill_num_wgs = oneapi::dpl::__internal::__dpl_ceiling_div(status_flags_elems, wgsize);
    using AtomicT = typename scan_memory<_Type>::_AtomicT;
    AtomicT* status_flags = reinterpret_cast<AtomicT*>(scratch);

    auto fill_event = __queue.submit(
        [&](sycl::handler& hdl)
        {
            hdl.parallel_for<class scan_kt_init>(sycl::nd_range<1>{fill_num_wgs * wgsize, wgsize},
                                                 [=](const sycl::nd_item<1>& item)
                                                 {
                                                     int id = item.get_global_linear_id();
                                                     if (id < status_flags_elems)
                                                         status_flags[id] = id < scan_memory<_Type>::padding
                                                                                ? scan_memory<_Type>::OUT_OF_BOUNDS
                                                                                : scan_memory<_Type>::NOT_READY;
                                                 });
        });

    auto event = __queue.submit([&](sycl::handler& hdl) {
        auto tile_id_lacc = sycl::local_accessor<std::uint32_t, 1>(sycl::range<1>{1}, hdl);
        auto tile_vals = sycl::local_accessor<_Type, 1>(sycl::range<1>{elems_in_tile}, hdl);
        hdl.depends_on(fill_event);

        oneapi::dpl::__ranges::__require_access(hdl, __in_rng, __out_rng);
        hdl.parallel_for<class scan_kt_main>(sycl::nd_range<1>(num_workitems, wgsize), [=](const sycl::nd_item<1>& item)  [[intel::reqd_sub_group_size(SUBGROUP_SIZE)]] {
            auto group = item.get_group();
            auto local_id = item.get_local_id(0);
            auto stride = item.get_local_range(0);
            auto subgroup = item.get_sub_group();

            // Obtain unique ID for this work-group that will be used in decoupled lookback
            if (group.leader())
            {
                tile_id_lacc[0] = scan_memory<_Type>::load_counter(num_wgs, scratch);
            }
            sycl::group_barrier(group);
            std::uint32_t tile_id = tile_id_lacc[0];

            // Global load into local
            auto wg_current_offset = (tile_id*elems_in_tile);
            auto wg_next_offset = ((tile_id+1)*elems_in_tile);
            size_t wg_local_memory_size = elems_in_tile;
            if (wg_current_offset >= n)
                return;
            if (wg_next_offset > n)
                wg_local_memory_size = n - wg_current_offset;

            if (wg_next_offset <= n) {
                _ONEDPL_PRAGMA_UNROLL
                for (std::uint32_t i = 0; i < elems_per_workitem; ++i)
                    tile_vals[local_id + stride * i] = __in_rng[wg_current_offset + local_id + stride * i];
            } else {
                for (std::uint32_t i = 0; i < elems_per_workitem; ++i) {
                    if (wg_current_offset + local_id + stride * i < n)
                        tile_vals[local_id + stride * i] = __in_rng[wg_current_offset + local_id + stride * i];
                }
            }
            sycl::group_barrier(group);

            auto in_begin = tile_vals.get_pointer();
            auto in_end = in_begin + wg_local_memory_size;
            auto out_begin = __out_rng.begin() + wg_current_offset;

            auto local_sum = sycl::joint_reduce(group, in_begin, in_end, __binary_op);
            _Type prev_sum = 0;

            // The first sub-group will query the previous tiles to find a prefix
            if (subgroup.get_group_id() == 0)
            {
                scan_memory<_Type> scan_mem(tile_id, scratch, num_wgs);

                if (group.leader())
                    scan_mem.set_partial(local_sum);

                // Find lowest work-item that has a full result (if any) and sum up subsequent partial results to obtain this tile's exclusive sum
                prev_sum = cooperative_lookback()(tile_id, subgroup, __binary_op, scan_mem);

                if (group.leader())
                    scan_mem.set_full(prev_sum + local_sum);
            }

            prev_sum = sycl::group_broadcast(group, prev_sum, 0);
            sycl::joint_inclusive_scan(group, in_begin, in_end, out_begin, __binary_op, prev_sum);
        });
    });

    scan_memory<_Type>::async_free(scratch, __queue, event);

    event.wait();
}

// The generic structure for configuring a kernel
template <std::uint16_t ElemsPerWorkItem, std::uint16_t WorkGroupSize, typename KernelName>
struct kernel_param
{
    static constexpr std::uint16_t elems_per_workitem = ElemsPerWorkItem;
    static constexpr std::uint16_t workgroup_size = WorkGroupSize;
    using kernel_name = KernelName;
};

template <typename _KernelParam, typename _InIterator, typename _OutIterator, typename _BinaryOp>
void
single_pass_inclusive_scan(sycl::queue __queue, _InIterator __in_begin, _InIterator __in_end, _OutIterator __out_begin, _BinaryOp __binary_op)
{
    auto __n = __in_end - __in_begin;

    auto __keep1 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::read, _InIterator>();
    auto __buf1 = __keep1(__in_begin, __in_end);
    auto __keep2 =
        oneapi::dpl::__ranges::__get_sycl_range<__par_backend_hetero::access_mode::write, _OutIterator>();
    auto __buf2 = __keep2(__out_begin, __out_begin + __n);

    single_pass_scan_impl<_KernelParam, true>(__queue, __buf1.all_view(), __buf2.all_view(), __binary_op);
}

} // inline namespace igpu

} // namespace oneapi::dpl::experimental::kt

#endif /* _ONEDPL_parallel_backend_sycl_scan_H */
