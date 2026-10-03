"""Paired SM100 producer: exact BF16 histograms and dynamic request schedules."""

import pytest
import torch

import deep_gemm as dg

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires SM100",
)


def histogram_reference(scores, lens):
    values = scores.float().contiguous()
    bits = values.view(torch.int32).long() & 0xFFFFFFFF
    mag = bits & 0x7FFFFFFF
    negative = ((bits >> 31) != 0) & (mag != 0)
    half = values.half().view(torch.int16).long() & 0xFFFF
    code = (half & 0x7FFF) >> 6
    adjusted = (mag - negative.long()).clamp(max=0x435F0000).int().view(torch.float32)
    wide = (adjusted.floor().long() + 288).clamp(min=304)
    code = torch.where(code >= 304, wide, code)
    bins = torch.where(negative, 512 + code, 511 - code)
    valid = torch.arange(scores.shape[1], device="cuda")[None, :] < lens
    valid &= ~values.isnan()
    out = torch.zeros((scores.shape[0], 1024), device="cuda", dtype=torch.int32)
    out.scatter_add_(1, bins.clamp(0, 1023), valid.int())
    return out


@pytest.mark.parametrize("n", range(1, 7))
@pytest.mark.parametrize("pdl", [False, True])
def test_exact_histogram_graph(n, pdl):
    torch.manual_seed(321 + n)
    rows, width, pages = 3 * n, 32768, 256
    q = torch.randint(0, 256, (rows, 1, 32, 64), device="cuda", dtype=torch.uint8).view(
        torch.int8
    )
    sf = torch.full((rows, 1, 32), 0x7D7D7D7D, device="cuda", dtype=torch.int32)
    cache = torch.randint(
        0, 256, (3 * pages, 128 * 68), device="cuda", dtype=torch.uint8
    )
    cache[:, 128 * 64 :] = 125
    cache = cache.view(-1, 128, 1, 68)
    weights = torch.randn((rows, 32), device="cuda", dtype=torch.bfloat16)
    weights[0].fill_(float("nan"))
    weights[-1].zero_()
    ids = torch.arange(rows, device="cuda", dtype=torch.int32) // n
    table = torch.randperm(3 * pages, device="cuda").int().view(3, pages)
    table = table.repeat_interleave(n, 0).contiguous()
    lens = torch.full((rows, 1), width, device="cuda", dtype=torch.int32)
    hist = torch.zeros((rows, 1024), device="cuda", dtype=torch.int32)

    def produce(histogram=None):
        schedule = dg.get_paged_mqa_logits_bf16_metadata(
            lens, 128, dg.get_num_sms(), indices=ids, tokens_per_request=n
        )
        return dg.fp4_paged_mqa_logits_bf16(
            (q, sf),
            cache,
            weights,
            lens,
            table,
            schedule,
            width,
            indices=ids,
            histogram=histogram,
            tokens_per_request=n,
        )

    old_pdl = dg.get_pdl()
    dg.set_pdl(pdl)
    try:
        produce(hist)
        plain = produce()
        assert torch.equal(hist, histogram_reference(plain, lens))
        produce(hist)
        assert torch.equal(hist, 2 * histogram_reference(plain, lens))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            hist.zero_()
            result = produce(hist)
        for length in (0, 1, 383, 384, 385, 513, 32767):
            lens.copy_(
                (length - n + 1 + torch.arange(rows, device="cuda").int() % n)
                .clamp_min(0)
                .view(-1, 1)
            )
            # Break same-ID runs without changing their page tables.
            ids.copy_(
                torch.arange(rows, device="cuda").int() // (n if length % 2 else 1)
            )
            graph.replay()
            plain = produce()
            mask = torch.arange(width, device="cuda")[None, :] < lens
            assert torch.equal(
                result.view(torch.int16).masked_fill(~mask, 0),
                plain.view(torch.int16).masked_fill(~mask, 0),
            )
            assert torch.equal(hist, histogram_reference(result, lens))
    finally:
        dg.set_pdl(old_pdl)


if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        print("SKIP: requires SM100")
    else:
        for use_pdl in (False, True):
            for next_n in range(1, 7):
                test_exact_histogram_graph(next_n, use_pdl)
        print("12 cases passed")
