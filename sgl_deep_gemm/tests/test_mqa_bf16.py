"""Paired SM100 producer: exact BF16 histograms and dynamic request schedules."""

import torch

import deep_gemm as dg
from utils import ref_coarse_histogram


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
        assert torch.equal(hist, ref_coarse_histogram(plain, lens))
        produce(hist)
        assert torch.equal(hist, 2 * ref_coarse_histogram(plain, lens))
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
            assert torch.equal(hist, ref_coarse_histogram(result, lens))
    finally:
        dg.set_pdl(old_pdl)


def test_schedule_row_limit():
    if torch.cuda.get_device_capability()[0] != 10:
        print("Skipping the SM100 BF16 schedule row limit on this architecture")
        return
    # One CTA builds the schedule, keeping two int32 per row in shared memory
    lens = torch.ones((16385, 1), device="cuda", dtype=torch.int32)
    dg.get_paged_mqa_logits_bf16_metadata(lens[:16384], 128, dg.get_num_sms())
    try:
        dg.get_paged_mqa_logits_bf16_metadata(lens, 128, dg.get_num_sms())
    except RuntimeError as error:
        assert "at most 16384 rows" in str(error), error
    else:
        raise AssertionError("a 16385-row schedule must be rejected")


if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        print("SKIP: requires SM100")
    else:
        for use_pdl in (False, True):
            for next_n in range(1, 7):
                test_exact_histogram_graph(next_n, use_pdl)
        test_schedule_row_limit()
        print("13 cases passed")
