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
struct ScanMemory
{
    using _FlagT = ::std::uint32_t;
    using _AtomicFlagRefT = sycl::atomic_ref<_FlagT, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>;

    static constexpr _FlagT NOT_READY = 0;
    static constexpr _FlagT PARTIAL_MASK = 1;
    static constexpr _FlagT FULL_MASK = 2;
    static constexpr _FlagT OUT_OF_BOUNDS = 4;

    static constexpr ::std::size_t padding = SUBGROUP_SIZE;

    ScanMemory(::std::uint8_t* scan_memory_begin, ::std::size_t num_wgs) : num_elements(get_num_elements(num_wgs))
    {
        // In memory: [Partial Value, ..., Full Value, ..., Flag, ...]
        // Each section is num_wgs + padding
        _T* tile_values_begin = reinterpret_cast<_T*>(scan_memory_begin);
        partial_values_begin = tile_values_begin;
        full_values_begin = tile_values_begin + num_elements;

        // Aligned flags
        ::std::size_t tile_values_bytes = get_tile_values_bytes(num_elements);
        void* base_flags = reinterpret_cast<void*>(scan_memory_begin + tile_values_bytes);
        auto remainder = get_padded_flag_bytes(num_elements); // scan_memory_bytes - tile_values_bytes
        flags_begin = reinterpret_cast<_FlagT*>(
            ::std::align(::std::alignment_of_v<_FlagT>, get_flag_bytes(num_elements), base_flags, remainder));
    }

    void
    set_partial(::std::size_t tile_id, _T val)
    {
        _AtomicFlagRefT atomic_flag(*(flags_begin + tile_id + padding));

        partial_values_begin[tile_id + padding] = val;
        atomic_flag.store(PARTIAL_MASK);
    }

    void
    set_full(::std::size_t tile_id, _T val)
    {
        _AtomicFlagRefT atomic_flag(*(flags_begin + tile_id + padding));

        full_values_begin[tile_id + padding] = val;
        atomic_flag.store(FULL_MASK);
    }

    _FlagT
    load_flag(::std::size_t tile_id) const
    {
        _AtomicFlagRefT atomic_flag(*(flags_begin + tile_id + padding));

        return atomic_flag.load();
    }

    _T
    get_value(::std::size_t tile_id, _FlagT flag) const
    {
        ::std::size_t offset = tile_id + padding + num_elements * is_full(flag);
        return tile_values_begin[offset];
    }

    static ::std::size_t
    get_tile_values_bytes(::std::size_t num_elements)
    {
        return (2 * num_elements) * sizeof(_T);
    }

    static ::std::size_t
    get_flag_bytes(::std::size_t num_elements)
    {
        return num_elements * sizeof(_FlagT);
    }

    static ::std::size_t
    get_padded_flag_bytes(::std::size_t num_elements)
    {
        // sizeof(_FlagT) extra bytes for possible intenal alignment
        return get_flag_bytes(num_elements) + sizeof(_FlagT);
    }

    static ::std::size_t
    get_memory_size(::std::size_t num_wgs)
    {
        ::std::size_t num_elements = get_num_elements(num_wgs);
        // sizeof(_T) extra bytes are not needed because ScanMemory is going at the beginning of the scratch
        ::std::size_t tile_values_bytes = get_tile_values_bytes(num_elements);
        // Padding to provide room for aligment
        ::std::size_t flag_bytes = get_padded_flag_bytes(num_elements);

        std::cout << "get_memory_size " << std::endl;
        std::cout << "  num_elements " << num_elements << std::endl;
        std::cout << "  tile_values_bytes " << tile_values_bytes << std::endl;
        std::cout << "  flag_bytes " << flag_bytes << std::endl;
        std::cout << "  mem_size " << tile_values_bytes + flag_bytes << std::endl;

        return tile_values_bytes + flag_bytes;
    }

    static ::std::size_t
    get_num_elements(::std::size_t num_wgs)
    {
        return padding + num_wgs;
    }

    static bool
    is_ready(_FlagT flag)
    {
        return flag != NOT_READY;
    }

    static bool
    is_full(_FlagT flag)
    {
        return flag == FULL_MASK;
    }

    static bool
    is_out_of_bounds(_FlagT flag)
    {
        return flag == OUT_OF_BOUNDS;
    }

    ::std::size_t num_elements;
    _FlagT* flags_begin;
    _T* tile_values_begin;
    _T* partial_values_begin;
    _T* full_values_begin;
};

struct TileId
{
    using _TileIdT = ::std::uint32_t;
    using _AtomicTileRefT = sycl::atomic_ref<_TileIdT, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>;

    TileId(_TileIdT* tileid_memory) : tile_counter(*(tileid_memory)) {}

    constexpr static ::std::size_t
    get_padded_memory_size()
    {
        // extra sizeof(_TileIdT) for possible aligment issues
        return sizeof(_TileIdT) + sizeof(_TileIdT);
    }

    constexpr static ::std::size_t
    get_memory_size()
    {
        // extra sizeof(_TileIdT) for possible aligment issues
        return sizeof(_TileIdT);
    }

    _TileIdT
    fetch_inc()
    {
        return tile_counter.fetch_add(1);
    }

    _AtomicTileRefT tile_counter;
};

template <typename Type, template <typename> typename ScanMemory, typename TileId>
struct ScanScratchMemory
{
    using _TileIdT = typename TileId::_TileIdT;
    using _FlagT = typename ScanMemory<Type>::_FlagT;

    ScanScratchMemory(sycl::queue q) : q{q} {};

    ::std::uint8_t*
    scan_memory_ptr() noexcept
    {
        return scan_memory_begin;
    };

    _TileIdT*
    tile_id_ptr() noexcept
    {
        return tile_id_begin;
    };

    void
    allocate(::std::size_t num_wgs)
    {
        ::std::size_t scan_memory_size = ScanMemory<Type>::get_memory_size(num_wgs);
        constexpr ::std::size_t padded_tileid_size = TileId::get_padded_memory_size();
        constexpr ::std::size_t tileid_size = TileId::get_memory_size();

        mem_size_bytes = scan_memory_size + padded_tileid_size;

        scratch = reinterpret_cast<::std::uint8_t*>(sycl::malloc_device(mem_size_bytes, q));

        scan_memory_begin = scratch;

        void* base_tileid_ptr = reinterpret_cast<void*>(scan_memory_begin + scan_memory_size);
        size_t remainder = mem_size_bytes - scan_memory_size;

        tile_id_begin = reinterpret_cast<_TileIdT*>(
            ::std::align(::std::alignment_of_v<_TileIdT>, tileid_size, base_tileid_ptr, remainder));
    }

    void
    async_free(sycl::event event_dependency)
    {
        q.submit(
            [=](sycl::handler& hdl)
            {
                hdl.depends_on(event_dependency);
                hdl.host_task([=]() { sycl::free(scratch, q); });
            });
    }

    ::std::size_t mem_size_bytes;
    ::std::uint8_t* scratch = nullptr;

    ::std::uint8_t* scan_memory_begin = nullptr;
    _TileIdT* tile_id_begin = nullptr;

    sycl::queue q;
};

struct cooperative_lookback
{

    template <typename _T, typename _Subgroup, typename BinOp, template <typename> typename ScanMemory>
    _T
    operator()(std::uint32_t tile_id, const _Subgroup& subgroup, BinOp bin_op, ScanMemory<_T> memory)
    {
        using FlagT = typename ScanMemory<_T>::_FlagT;

        _T sum = 0;
        int offset = -1;
        int i = 0;
        int local_id = subgroup.get_local_id();

        for (int tile = static_cast<int>(tile_id) + offset; tile >= 0; tile -= SUBGROUP_SIZE)
        {
            FlagT flag;
            do
            {
                flag = memory.load_flag(tile - local_id);
            } while (!sycl::all_of_group(subgroup, ScanMemory<_T>::is_ready(flag))); // Loop till all ready

            bool is_full = ScanMemory<_T>::is_full(flag);
            auto is_full_ballot = sycl::ext::oneapi::group_ballot(subgroup, is_full);
            auto lowest_item_with_full = is_full_ballot.find_low();

            // TODO: Use identity_fn for out of bounds values
            _T contribution = local_id <= lowest_item_with_full && (!ScanMemory<_T>::is_out_of_bounds(flag))
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
    using _TileIdT = TileId::_TileIdT;
    using _FlagT = typename ScanMemory<_Type>::_FlagT;

    static_assert(_Inclusive, "Single-pass scan only available for inclusive scan");

    const ::std::size_t n = __in_rng.size();

    constexpr ::std::size_t wgsize = _KernelParam::workgroup_size;
    constexpr ::std::size_t elems_per_workitem = _KernelParam::elems_per_workitem;

    // Avoid non_uniform n by padding up to a multiple of wgsize
    ::std::uint32_t elems_in_tile = wgsize * elems_per_workitem;
    ::std::size_t num_wgs = oneapi::dpl::__internal::__dpl_ceiling_div(n, elems_in_tile);
    ::std::size_t num_workitems = num_wgs * wgsize;

    // TODO: Remove Dynamic Tile Counter from the middle of the scratch
    // Probably the best way is to have in memory: Values, Flags, Acc. Align memory.
    // Right Now -> Assumes TileId::TileIdT and ScanMemory<_Type>::FlagT are the same type
    ScanScratchMemory<_Type, ScanMemory, TileId> scan_scratch(__queue);
    scan_scratch.allocate(num_wgs);

    auto scan_memory_begin = scan_scratch.scan_memory_ptr();
    auto tile_id_begin = scan_scratch.tile_id_ptr();

    ::std::size_t num_elements = ScanMemory<_Type>::get_num_elements(num_wgs);
    ::std::size_t fill_num_wgs = oneapi::dpl::__internal::__dpl_ceiling_div(num_elements, wgsize);
    ::std::size_t tile_values_bytes = ScanMemory<_Type>::get_tile_values_bytes(num_elements);
    void* base_flags = reinterpret_cast<void*>(scan_memory_begin + tile_values_bytes);
    auto remainder = ScanMemory<_Type>::get_padded_flag_bytes(num_elements);
    auto status_flags_begin = reinterpret_cast<_FlagT*>(::std::align(
        ::std::alignment_of_v<_FlagT>, ScanMemory<_Type>::get_flag_bytes(num_elements), base_flags, remainder));

    std::cout << "OMG " << tile_values_bytes << std::endl << std::flush;

    auto fill_event = __queue.submit(
        [&](sycl::handler& hdl)
        {
            hdl.parallel_for<class scan_kt_init>(sycl::nd_range<1>{fill_num_wgs * wgsize, wgsize},
                                                 [=](const sycl::nd_item<1>& item)
                                                 {
                                                     int id = item.get_global_linear_id();
                                                     if (id < num_elements)
                                                         status_flags_begin[id] = id < ScanMemory<_Type>::padding
                                                                                      ? ScanMemory<_Type>::OUT_OF_BOUNDS
                                                                                      : ScanMemory<_Type>::NOT_READY;
                                                     if (id == num_elements)
                                                         tile_id_begin[0] = 0;
                                                 });
        });

    std::cout << "OMG" << std::endl << std::flush;

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
            TileId dynamic_tile_id(tile_id_begin);

            // Obtain unique ID for this work-group that will be used in decoupled lookback
            if (group.leader())
            {
                tile_id_lacc[0] = dynamic_tile_id.fetch_inc();
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

            auto in_begin = tile_vals.template get_multi_ptr<sycl::access::decorated::no>().get();
            auto in_end = in_begin + wg_local_memory_size;
            auto out_begin = __out_rng.begin() + wg_current_offset;

            auto local_sum = sycl::joint_reduce(group, in_begin, in_end, __binary_op);
            _Type prev_sum = 0;

            // The first sub-group will query the previous tiles to find a prefix
            if (subgroup.get_group_id() == 0)
            {
                ScanMemory<_Type> scan_mem(scan_memory_begin, num_wgs);

                if (group.leader())
                    scan_mem.set_partial(tile_id, local_sum);

                // Find lowest work-item that has a full result (if any) and sum up subsequent partial results to obtain this tile's exclusive sum
                prev_sum = cooperative_lookback()(tile_id, subgroup, __binary_op, scan_mem);

                if (group.leader())
                    scan_mem.set_full(tile_id, prev_sum + local_sum);
            }

            prev_sum = sycl::group_broadcast(group, prev_sum, 0);
            sycl::joint_inclusive_scan(group, in_begin, in_end, out_begin, __binary_op, prev_sum);
        });
    });

    scan_scratch.async_free(event);

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
