"""SM90 MegaMoE ring-buffer check.

Every shape runs four launches on a symm buffer of its capacity in the default layout (a lag-scheduled ring from 1024 tokens per
rank, the full pool below that): the full routing, a smaller token count with a skewed routing, the full routing again, and a
decode-sized one, so both launch-parity banks and the ring wrap are exercised on one allocation. The first launch must match the
reference of test_mega_moe_hopper.py within its tolerance and the third must repeat the first bit for bit. Where the default
layout is a ring, the launches of at most 896 tokens run again on a full-pool buffer (a capacity below 1024) and must match the
ring's output bit for bit: the same call under the wave order on the pool and the lag order on the ring. The first two-rank shape
is the configuration whose requested lag ring exceeds the full pool (the buffer is the pool itself under the lag schedule). A
further buffer of the default layout then repeats all four launches with each call's own ``num_tokens_bound`` instead of the buffer
capacity, which must not change a single bit either: on a lag ring that holds one wave of that bound, the decode-sized launch runs
the wave order under the bound where it runs the lag order without one, and on the top-6 shape the smaller launch runs a shorter lag
under its bound. A bound above the capacity must be rejected instead, and so must a buffer for an intermediate size the kernel's
tiles do not divide.

The bit-for-bit comparisons compare bit patterns, not BF16 values, and each rejection must come from its own host assertion. A
default layout that is a ring must be no larger than the full pool (a lag ring larger than the pool is clamped to it: on the first
two-rank shape, and on more shapes at a world size of 2 or 4), and the run fails unless at least one shape had a ring smaller than
its pool and one ran the cross-layout comparison.
"""

import argparse
import sys

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist
from deep_gemm.testing import calc_diff, get_arch_major

from test_mega_moe_hopper import _quantize_grouped_fp8_block_128_128, _reference_fused

ACTIVATION_CLAMP = 10.0
# tokens per rank from which the default layout is a lag ring (heuristics kSm90AutoLagMinTokens)
AUTO_LAG_MIN_TOKENS = 1024
# per-rank token bound up to which a call on a lag ring takes the wave schedule instead (heuristics kSm90CallWaveMaxTokens)
CALL_WAVE_MAX_TOKENS = 256
TOKEN_ALIGNMENT = 128
# the host assertions the two rejections must come from (csrc/apis/sm90_mega.hpp: the sizer, the launch)
SHAPE_ASSERTION = "hidden % 512 == 0 and intermediate_hidden % 256 == 0"
BOUND_ASSERTION = "num_tokens_bound <= num_max_tokens_per_rank"


def _shapes(num_ranks):
    out = [
        (f"ring.h1024.t{tokens}", dict(hidden=1024, intermediate_hidden=1024, num_experts=8 * num_ranks, num_topk=2, num_tokens=tokens))
        for tokens in (64, 256, 512, 1024, 2048)
    ]
    out.append(("ring.h4096.e288.t2048", dict(hidden=4096, intermediate_hidden=2048, num_experts=288, num_topk=8, num_tokens=2048)))
    # top-6 at a 5120-token capacity: the buffer's lag spans four 1024-token batches per local expert and the smaller launch's
    # own bound spans three, so here a bound shortens the lag instead of switching the order, which no other shape's bound does
    out.append(("ring.h1024.topk6.t5120", dict(hidden=1024, intermediate_hidden=1024, num_experts=8 * num_ranks, num_topk=6, num_tokens=5120)))
    # 64-token launches on a 65536-token capacity: more token stripes than the combine keeps bits for
    out.append(("ring.h1024.cap65536", dict(hidden=1024, intermediate_hidden=1024, num_experts=8 * num_ranks, num_topk=2, num_tokens=64, cap=65536)))
    # intermediate_hidden % 256 != 0: no 256-wide tile divides the L1 output into an even number of N blocks, so the buffer is refused
    out.append(("ring.h1024.ih384.reject", dict(hidden=1024, intermediate_hidden=384, num_experts=8 * num_ranks, num_topk=2, num_tokens=64,
                                                reject=True)))
    # the two-rank shapes come last (see test()): the sub-group's buffers must not be followed by buffers of the full group
    # two ranks, 8 local experts: the sizer asks for a 12800-token lag ring, the full pool is 9216 tokens -> clamped to the pool
    out.append(("ring.h1024.t2048.ep2", dict(hidden=1024, intermediate_hidden=1024, num_experts=16, num_topk=2, num_tokens=2048, world=2)))
    # two ranks x 256 local experts: the flag words of the workspace, (experts + local experts) x 4 bytes, fill whole KiB
    out.append(("ring.h1024.e512.ep2", dict(hidden=1024, intermediate_hidden=1024, num_experts=512, num_topk=2, num_tokens=256, world=2)))
    return out


def _align(x, a):
    return (x + a - 1) // a * a


def _bits_equal(a, b):
    # torch.equal compares BF16 values, so +0 and -0 would pass
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _full_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank):
    # layout::get_num_max_pool_tokens_sm90 (one partial block of kSM90MaxCandidateBlockM 128 per expert, kSM90LCMBlockM 128)
    return _align(num_ranks * num_max_tokens_per_rank * min(num_topk, num_experts_per_rank) + num_experts_per_rank * 127, 128)


def _routing(num_tokens, num_experts, num_topk, seed, num_hot_experts=0):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device="cuda", generator=gen)
    if num_hot_experts:
        # skewed routing: a few experts take most rows, the others a handful
        scores[:, :num_hot_experts] += 4.0
    topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    return topk_idx, topk_weights


def _run_shape(name, cfg, rank_idx, num_ranks, group, diff_tol):
    """Runs one shape on `group`; returns (ok, message, (ring below the pool, cross-layout compared)). No assertion and no collective
    outside `group`."""
    hidden, intermediate_hidden = cfg["hidden"], cfg["intermediate_hidden"]
    num_experts, num_topk, num_tokens = cfg["num_experts"], cfg["num_topk"], cfg["num_tokens"]
    num_experts_per_rank = num_experts // num_ranks
    num_max_tokens_per_rank = _align(cfg.get("cap", num_tokens), TOKEN_ALIGNMENT)

    def make_buffer(capacity):
        return deep_gemm.get_symm_buffer_for_sm90_mega_moe(group, num_experts, capacity, num_topk, hidden, intermediate_hidden)

    if cfg.get("reject"):
        # the host refuses the shape while sizing, before any collective
        try:
            buffer = make_buffer(num_max_tokens_per_rank)
        except RuntimeError as ex:
            matched = SHAPE_ASSERTION in str(ex)
            return matched, f"rejected{'' if matched else ' by another error'}: {str(ex).splitlines()[-1]}", (False, False)
        buffer.destroy()
        return False, "buffer created (expected the shape assertion)", (False, False)

    seed = rank_idx * 1000 + sum(map(ord, name))
    torch.manual_seed(seed)

    x_bf16 = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    l1_weights_bf16 = torch.randn((num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device="cuda") * 0.05
    l2_weights_bf16 = torch.randn((num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device="cuda") * 0.05
    x_fp8 = per_token_cast_to_fp8(x_bf16, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)
    l1_weights = _quantize_grouped_fp8_block_128_128(l1_weights_bf16)
    l2_weights = _quantize_grouped_fp8_block_128_128(l2_weights_bf16)
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(l1_weights, l2_weights)
    del l1_weights_bf16, l2_weights_bf16

    # launch 1 and 3: every token, the same routing (3 runs on the parity bank of launch 1 again); launch 2: fewer tokens, skewed
    # routing (the smaller call may select a different tile class on the same buffer; the checks do not depend on it)
    launches = [(num_tokens,) + _routing(num_tokens, num_experts, num_topk, seed)]
    num_tokens_2 = max(1, num_tokens * 3 // 4)
    launches.append((num_tokens_2,) + _routing(num_tokens_2, num_experts, num_topk, seed + 1, num_hot_experts=2))
    launches.append(launches[0])
    num_tokens_4 = min(CALL_WAVE_MAX_TOKENS, num_tokens)
    launches.append((num_tokens_4,) + _routing(num_tokens_4, num_experts, num_topk, seed + 2))

    # full pool -> the default layout is a lag ring (or the pool itself when the ring would exceed it), else the pool
    full_pool = _full_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank)
    default_is_ring = num_max_tokens_per_rank >= AUTO_LAG_MIN_TOKENS
    # the launches a full-pool buffer can also take (its capacity stays below the lag threshold)
    small = [i for i, (n, _, _) in enumerate(launches) if _align(n, TOKEN_ALIGNMENT) < AUTO_LAG_MIN_TOKENS] if default_is_ring else []

    def run_launches(buffer, bounds, indices=None):
        ys = []
        for (n, topk_idx, topk_weights), bound in zip([launches[i] for i in (indices or range(len(launches)))], bounds):
            buffer.x[:n].copy_(x_fp8[0][:n])
            buffer.x_sf[:n].copy_(x_fp8[1][:n])
            buffer.topk_idx[:n].copy_(topk_idx)
            buffer.topk_weights[:n].copy_(topk_weights)
            y = torch.empty((n, hidden), dtype=torch.bfloat16, device="cuda")
            deep_gemm.fp8_mega_moe(y, transformed_l1, transformed_l2, buffer, recipe=(128, 128, 128), activation="swiglu",
                                   activation_clamp=ACTIVATION_CLAMP, fast_math=True, num_tokens_bound=bound)
            torch.cuda.synchronize()
            ys.append(y)
        return ys

    # the default layout: data pools of the ring capacity, or of the full pool
    buffer = make_buffer(num_max_tokens_per_rank)
    num_pool_rows = buffer.l1_acts.shape[0]
    outputs = run_launches(buffer, [None] * len(launches))
    buffer.destroy()
    dist.barrier(group=group)

    # where that is a ring, the launches a full-pool buffer can take run again on one
    control_cap = max([_align(launches[i][0], TOKEN_ALIGNMENT) for i in small], default=0)
    control_rows, control_outputs = None, []
    if small:
        buffer = make_buffer(control_cap)
        control_rows = buffer.l1_acts.shape[0]
        control_outputs = run_launches(buffer, [None] * len(small), small)
        buffer.destroy()
        dist.barrier(group=group)

    # the default layout again, this time telling the kernel each call's own bound instead of letting it assume the capacity
    buffer = make_buffer(num_max_tokens_per_rank)
    bounds = [_align(n, TOKEN_ALIGNMENT) for n, _, _ in launches]
    bounded_outputs = run_launches(buffer, bounds)
    # a bound the buffer cannot receive is rejected on the host, before any rank launches
    over_cap_error = ""
    try:
        run_launches(buffer, [num_max_tokens_per_rank + TOKEN_ALIGNMENT] * len(launches))
    except RuntimeError as ex:
        over_cap_error = str(ex)
    rejected = BOUND_ASSERTION in over_cap_error
    buffer.destroy()
    dist.barrier(group=group)

    # data pools: the default layout is a lag ring of at most the pool, or the pool; the control buffer is its own full pool
    problems = []
    if default_is_ring and num_pool_rows > full_pool:
        problems.append(f"default layout ring={num_pool_rows} (expected a lag ring <= pool {full_pool})")
    elif not default_is_ring and num_pool_rows != full_pool:
        problems.append(f"default layout pool={num_pool_rows} (expected the full pool {full_pool} below {AUTO_LAG_MIN_TOKENS} tokens)")
    if small and control_rows != _full_pool_tokens(num_ranks, control_cap, num_topk, num_experts_per_rank):
        problems.append(f"full-pool control pool={control_rows} (expected the full pool of capacity {control_cap})")
    if not rejected:
        problems.append(f"over-capacity bound: {over_cap_error.splitlines()[-1] if over_cap_error else 'no error'}")

    # launch 3 bit-identical to launch 1; the full-pool control and the bounded calls bit-identical to the default layout
    replay = _bits_equal(outputs[0], outputs[2])
    cross_layout = all(_bits_equal(control_outputs[j], outputs[i]) for j, i in enumerate(small))
    bounded = all(_bits_equal(bounded_outputs[i], outputs[i]) for i in range(len(launches)))
    topk_idx_1, topk_weights_1 = launches[0][1], launches[0][2]
    y_ref = _reference_fused(
        x_fp8[0], x_fp8[1], topk_idx_1, topk_weights_1,
        l1_weights[0], l1_weights[1], l2_weights[0], l2_weights[1],
        rank_idx, num_ranks, group, num_experts, num_topk, hidden, intermediate_hidden, ACTIVATION_CLAMP)
    diff = calc_diff(outputs[0], y_ref)
    ok = replay and cross_layout and bounded and rejected and diff < diff_tol and not problems
    control = (f" control=pool{control_rows}@cap{control_cap}:launches{[i + 1 for i in small]}:{'OK' if cross_layout else 'FAIL'}"
               if small else "")
    msg = (f"default={'ring' if default_is_ring else 'pool'}{num_pool_rows} (pool {full_pool}){control} "
           f"replay={'OK' if replay else 'FAIL'} "
           f"bounds={bounds}<=cap{num_max_tokens_per_rank}:{'OK' if bounded else 'FAIL'} "
           f"over_cap_rejected={'OK' if rejected else 'FAIL'} diff={diff:.4f} (tol={diff_tol:.2f})"
           + (" " + "; ".join(problems) if problems else ""))
    return ok, msg, (default_is_ring and num_pool_rows < full_pool, bool(small))


def test(local_rank, num_local_ranks, args):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    if get_arch_major() != 9:
        dist_print(f"[SKIP] test_mega_moe_hopper_ring requires SM90; got SM{get_arch_major()}0", once_in_node=True)
        dist.destroy_process_group()
        return

    # the two-rank shapes run on ranks 0 and 1 (every rank takes part in creating the group)
    group_2 = dist.new_group(ranks=[0, 1]) if num_ranks > 2 else None

    failures, num_reduced_rings, num_cross_layout = [], 0, 0
    for name, cfg in _shapes(num_ranks):
        world = cfg.get("world", num_ranks)
        if world > num_ranks or cfg["num_experts"] % world != 0:
            dist_print(f"  [{name:<24}] SKIP ({cfg['num_experts']} experts on {world} ranks, {num_ranks} available)", once_in_node=True)
            continue
        covered = (False, False)
        try:
            if world == num_ranks:
                ok, msg, covered = _run_shape(name, cfg, rank_idx, num_ranks, group, args.diff_tol)
            elif rank_idx < world:
                ok, msg, covered = _run_shape(name, cfg, rank_idx, world, group_2, args.diff_tol)
            else:
                ok, msg = True, ""
        except Exception as ex:   # noqa: BLE001 -- a shape that raises on every rank is reported and the loop goes on; a raise on one rank alone still hangs on the collectives it skipped, as it would without the handler
            ok, msg = False, f"exception: {ex!r}"
        num_reduced_rings += covered[0]
        num_cross_layout += covered[1]
        # one verdict for all ranks before anyone prints or moves on: a rank-local failure must not skip a collective
        flag = torch.tensor([0 if ok else 1], dtype=torch.int32, device="cuda")
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        messages = [None] * num_ranks
        dist.all_gather_object(messages, msg if not ok else "")
        ok_all = flag.item() == 0
        failed_ranks = [r for r, m in enumerate(messages) if m]
        dist_print(f"  [{name:<24}] {msg if rank_idx == 0 and msg else ''} {'OK' if ok_all else 'FAIL'}"
                   + (f" ranks {failed_ranks}: {messages[failed_ranks[0]]}" if failed_ranks else ""), once_in_node=True)
        if not ok_all:
            failures.append(name)
    # run level: some shape must have had a ring below its pool and some the cross-layout comparison (rank 0 runs every shape)
    counts = torch.tensor([num_reduced_rings, num_cross_layout], dtype=torch.int32, device="cuda")
    dist.all_reduce(counts, op=dist.ReduceOp.MAX)
    covered_all = bool((counts > 0).all())
    dist_print(f"  [{'run':<24}] shapes with a ring below the pool: {counts[0].item()}, with the cross-layout check: {counts[1].item()} "
               f"{'OK' if covered_all else 'FAIL'}", once_in_node=True)
    if not covered_all:
        failures.append("run")
    dist.barrier()
    dist.destroy_process_group()
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SM90 MegaMoE ring-buffer bitwise check")
    parser.add_argument("--num-processes", type=int, default=8, help="Number of spawned processes, one per GPU")
    parser.add_argument("--diff-tol", type=float, default=0.07, help="calc_diff tolerance against the reference; default: 0.07")
    args = parser.parse_args()
    torch.multiprocessing.spawn(test, args=(args.num_processes, args), nprocs=args.num_processes)
