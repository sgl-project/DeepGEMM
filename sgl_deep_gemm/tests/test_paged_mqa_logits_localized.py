import math
import sys
import torch

import deep_gemm
from deep_gemm import locality_domain
from deep_gemm.testing import assert_bitwise_equal

NUM_HEADS, HEAD_DIM, PAGE_KV, SPLIT_KV = 32, 128, 64, 256
PAGE_SHAPE = (PAGE_KV, 1, HEAD_DIM + 4)
NUM_PAGES = 65536
STATIC_FLAG, SLOT_STATIC, SLOT_DOMAIN, SLOT_ALL_DOMAINS = 1, 1 << 31, 1 << 30, 1 << 29


def random_fp8(shape: tuple) -> torch.Tensor:
    # Any byte but the two NaN encodings of e4m3
    magnitude = torch.randint(0, 0x7f, shape, dtype=torch.uint8, device='cuda')
    return magnitude | (torch.randint(0, 2, shape, dtype=torch.uint8, device='cuda') << 7)


def fill_pages(kv_cache: torch.Tensor) -> None:
    pages = kv_cache.view(kv_cache.size(0), -1)
    pages[:, :PAGE_KV * HEAD_DIM] = random_fp8((kv_cache.size(0), PAGE_KV * HEAD_DIM))
    scales = torch.rand((kv_cache.size(0), PAGE_KV), device='cuda') * 0.01 + 0.005
    pages[:, PAGE_KV * HEAD_DIM:] = scales.view(torch.uint8).view(kv_cache.size(0), PAGE_KV * 4)


def make_caches() -> tuple:
    plain = torch.empty((NUM_PAGES, *PAGE_SHAPE), dtype=torch.uint8, device='cuda')
    fill_pages(plain)
    homed = locality_domain.empty_paged(NUM_PAGES, PAGE_SHAPE, torch.uint8)
    assert homed.shape == plain.shape and homed.dtype == plain.dtype and homed.is_contiguous()
    homed.copy_(plain)
    assert_bitwise_equal(homed, plain)
    return plain, homed


def make_block_table(context_lens: list, request_domains: list) -> torch.Tensor:
    # Request `i` takes distinct random pages of half `request_domains[i]` of the cache; `None` takes them anywhere
    max_pages = max(1, max(math.ceil(n / PAGE_KV) for n in context_lens))
    free = [torch.randperm(NUM_PAGES // 2, device='cuda') + d * (NUM_PAGES // 2) for d in range(2)]
    free.append(torch.randperm(NUM_PAGES, device='cuda'))
    used = [0, 0, 0]
    block_table = torch.zeros((len(context_lens), max_pages), dtype=torch.int32, device='cuda')
    for i, (n, d) in enumerate(zip(context_lens, request_domains)):
        d = 2 if d is None else d
        num_pages = math.ceil(n / PAGE_KV)
        block_table[i, :num_pages] = free[d][used[d]:used[d] + num_pages].int()
        used[d] += num_pages
    assert used[2] == 0 or used[0] == used[1] == 0, 'A case either homes all requests or none'
    return block_table


def run_case(caches: tuple, name: str, next_n: int, context_lens: list, request_domains: list,
             min_num_localized_splits=None, expect_static=None) -> None:
    plain, homed = caches
    batch_size, num_sms = len(context_lens), deep_gemm.get_num_sms()
    q = random_fp8((batch_size, next_n, NUM_HEADS, HEAD_DIM)).view(torch.float8_e4m3fn)
    weights = torch.randn((batch_size * next_n, NUM_HEADS), dtype=torch.float32, device='cuda')
    # Position `j` of a request sees `j` more tokens than position 0
    lens = torch.tensor(context_lens, dtype=torch.int32, device='cuda').unsqueeze(1)
    lens = (lens - (next_n - 1 - torch.arange(next_n, dtype=torch.int32, device='cuda'))).clamp(min=0).contiguous()
    lens[:, -1] = torch.tensor(context_lens, dtype=torch.int32, device='cuda')
    block_table = make_block_table(context_lens, request_domains)
    max_context_len = block_table.size(1) * PAGE_KV

    stock_meta = deep_gemm.get_paged_mqa_logits_metadata(lens, PAGE_KV, num_sms)
    ref = deep_gemm.fp8_paged_mqa_logits(q, plain, weights, lens, block_table, stock_meta, max_context_len)

    # Any cache that is not homed gets the stock schedule
    assert_bitwise_equal(deep_gemm.get_paged_mqa_logits_localized_metadata(lens, block_table, plain, num_sms), stock_meta)

    meta = deep_gemm.get_paged_mqa_logits_localized_metadata(lens, block_table, homed, num_sms, min_num_localized_splits)
    assert meta.dim() == 2 and meta.size(1) == 2 and meta.size(0) != num_sms + 1 and meta.dtype == torch.int32
    words = meta.view(-1).tolist()
    slots = [words[4 * i:4 * i + 4] for i in range(num_sms)]
    counter, num_local_slots, flags = words[4 * num_sms + 4:4 * num_sms + 6], words[4 * num_sms + 6], words[4 * num_sms + 7]
    domain_words = words[4 * num_sms + 8:]
    is_static = bool(flags & STATIC_FLAG)
    if expect_static is not None:
        assert is_static == expect_static, f'{name}: static={is_static}'
    assert counter == [0, 0] and words[4 * num_sms:4 * num_sms + 4] == [batch_size * next_n, 0, batch_size * next_n, 0]
    slots = [[word & 0xffffffff for word in slot] for slot in slots]
    assert all(bool(slot[1] & SLOT_STATIC) == bool(slot[1] & SLOT_ALL_DOMAINS) == is_static for slot in slots)

    # The slots tile every domain's work exactly once
    splits = [math.ceil(n / SPLIT_KV) for n in context_lens]
    domains = [(domain_words[i // 32] >> (i % 32)) & 1 for i in range(batch_size)]
    if is_static:
        assert num_local_slots == num_sms and not any(domains)
        stock = stock_meta.tolist()
        assert [[slot[0], slot[1] & (SLOT_ALL_DOMAINS - 1), slot[2], slot[3]] for slot in slots] == \
               [stock[i] + stock[i + 1] for i in range(num_sms)]
    else:
        for i, d in enumerate(request_domains):
            assert d is None or splits[i] == 0 or domains[i] == d, f'{name}: request {i} is in domain {domains[i]}'
    covered = [0] * batch_size
    for slot_idx, (start_q, start_split, end_q, end_split) in enumerate(slots):
        slot_domain = 1 if start_split & SLOT_DOMAIN else 0
        assert slot_domain == (0 if slot_idx < num_local_slots else 1)
        all_domains = bool(start_split & SLOT_ALL_DOMAINS)
        start_split &= SLOT_ALL_DOMAINS - 1
        request_idx, split_idx = start_q // next_n, start_split
        while request_idx < batch_size and (request_idx, split_idx) != (end_q // next_n, end_split):
            if (not all_domains and domains[request_idx] != slot_domain) or split_idx >= splits[request_idx]:
                request_idx, split_idx = request_idx + 1, 0
                continue
            covered[request_idx] += 1
            split_idx += 1
    assert covered == splits, f'{name}: slots cover {covered} of {splits} splits'

    # The last claim of a launch resets the counter, and a slot's static flag is only a hint
    flipped = meta.clone()
    flipped.view(-1)[1:4 * num_sms:4] ^= -(1 << 31)
    for schedule in (meta, meta, meta, flipped, flipped, meta):
        logits = deep_gemm.fp8_paged_mqa_logits(q, homed, weights, lens, block_table, schedule, max_context_len)
        torch.cuda.synchronize()
        assert schedule.view(-1)[4 * num_sms + 4:4 * num_sms + 6].tolist() == [0, 0], f'{name}: the claim counter was not reset'
        for i, n in enumerate(context_lens):
            num_computed = math.ceil(n / SPLIT_KV) * SPLIT_KV
            rows = slice(i * next_n, (i + 1) * next_n)
            assert_bitwise_equal(logits[rows, :num_computed], ref[rows, :num_computed])

    # A localized schedule only goes with a homed cache
    try:
        deep_gemm.fp8_paged_mqa_logits(q, plain, weights, lens, block_table, meta, max_context_len)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Expected a `RuntimeError`')
    print(f' > {name}: next_n={next_n}, {batch_size} requests, {sum(splits)} splits, '
          f'{"static" if is_static else f"{num_local_slots}/{num_sms - num_local_slots} slots"}')


def test_paged_mqa_logits_localized() -> None:
    print('Testing localized paged MQA logits:')
    caches = make_caches()
    alt = lambda n: [i % 2 for i in range(n)]
    for next_n in (1, 2, 5, 6):
        run_case(caches, 'even', next_n, [4096] * 16, alt(16), 0, False)
        run_case(caches, 'ragged', next_n, [1, 63, 64, 65, 255, 256, 257, 700, 4097, 12345, 31, 2049], alt(12), 0, False)
        run_case(caches, 'empty requests', next_n, [0, 513, 0, 0, 2048, 0, 77, 0], alt(8), 0, False)
        run_case(caches, 'single request', next_n, [20000], [1], 0, False)
        run_case(caches, 'tiny', next_n, [1], [0], 0, False)
        run_case(caches, 'fewer splits than SMs', next_n, [256] * 7, alt(7), 0, False)
        run_case(caches, 'domain 0 only', next_n, [3000] * 9, [0] * 9, 0, False)
        run_case(caches, 'domain 1 only', next_n, [3000] * 9, [1] * 9, 0, False)
        run_case(caches, 'skewed', next_n, [300, 300, 300, 30000], [0, 0, 0, 1], 0, False)
        run_case(caches, 'grouped', next_n, [5000] * 12, [0] * 6 + [1] * 6, 0, False)
        run_case(caches, 'mixed pages', next_n, [6000] * 10, [None] * 10, 0, False)
        run_case(caches, 'many requests', next_n, [200 + 37 * i for i in range(130)], alt(130), 0, False)
        run_case(caches, 'guard, small', next_n, [4096] * 32, alt(32), None, True)
        run_case(caches, 'guard, at the threshold', next_n, [SPLIT_KV * 128] * 64, alt(64), None, False)
        run_case(caches, 'guard, below the threshold', next_n, [SPLIT_KV * 128] * 63 + [SPLIT_KV * 127], alt(64), None, True)
        run_case(caches, 'guard, 16 requests', next_n, [SPLIT_KV * 640] * 16, alt(16), None, False)
        run_case(caches, 'guard, 15 requests', next_n, [SPLIT_KV * 640] * 15, alt(15), None, True)
        run_case(caches, 'guard, mixed pages', next_n, [SPLIT_KV * 128] * 64, [None] * 64, None, True)
        run_case(caches, 'guard, mixed pages, small threshold', next_n, [6000] * 20, [None] * 20, 1, True)
        run_case(caches, 'guard, homed pages, small threshold', next_n, [6000] * 20, alt(20), 1, False)
        run_case(caches, 'guard, one domain, small threshold', next_n, [6000] * 20, [0] * 20, 1, True)
        run_case(caches, 'guard, 5 of 20 in domain 1, small threshold', next_n, [6000] * 20, [1] * 5 + [0] * 15, 1, False)
        run_case(caches, 'guard, 4 of 20 in domain 1, small threshold', next_n, [6000] * 20, [1] * 4 + [0] * 16, 1, True)
        run_case(caches, 'static forced', next_n, [9000] * 20, alt(20), 2 ** 31 - 1, True)
    print()


if __name__ == '__main__':
    torch.manual_seed(0)

    if torch.cuda.get_device_capability()[0] != 10 or not deep_gemm.is_localization_available():
        print('Localized paged MQA logits need SM100 locality domains; skipping')
        sys.exit(0)
    test_paged_mqa_logits_localized()

    print('All tests passed')
