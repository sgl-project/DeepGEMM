"""Focused GPU contracts for caller-owned MXFP4 expansion buffers."""

import unittest

import torch


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    'Requires SM90',
)
class MXFP4ExpansionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import deep_gemm

        cls.dg = deep_gemm

    def test_mask_refresh_on_graph_replay_and_empty_input(self):
        ids = torch.tensor(
            [0, 71, 72, 287, -1, 288], device='cuda', dtype=torch.int32
        ).repeat(100)
        mask = torch.empty(288, device='cuda', dtype=torch.int32)
        self.dg.build_mxfp4_mega_moe_expert_mask(ids, mask)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.dg.build_mxfp4_mega_moe_expert_mask(ids, mask)
        for selected in ([0, 71, 72, 287], [72]):
            if selected == [72]:
                ids.fill_(72)
            mask.fill_(9)
            graph.replay()
            expected = torch.zeros_like(mask)
            expected[selected] = 1
            torch.testing.assert_close(mask, expected)
        self.dg.build_mxfp4_mega_moe_expert_mask(ids[:0], mask)
        torch.testing.assert_close(mask, torch.zeros_like(mask))

    def make_weights(self):
        generator = torch.Generator(device='cuda').manual_seed(42)

        def weights(n, k):
            return (
                torch.randint(
                    -128,
                    128,
                    (3, n, k // 2),
                    generator=generator,
                    device='cuda',
                    dtype=torch.int8,
                ),
                torch.full((3, n, k // 32), 127, device='cuda', dtype=torch.uint8),
            )

        return self.dg.transform_weights_for_fp8_mxfp4_fused_mega_moe_sm90(
            weights(512, 512), weights(512, 256), layout='tiled'
        )

    def test_masked_expansion_graph_preserves_inactive_experts(self):
        l1, l2 = self.make_weights()
        reference = tuple(
            torch.empty((*w.shape[:3], 32768), device='cuda', dtype=torch.float8_e4m3fn)
            for w, _, _ in (l1, l2)
        )
        output = tuple(torch.empty_like(t) for t in reference)
        self.dg.expand_mxfp4_mega_moe_weights(l1, l2, reference)
        mask = torch.ones(8, device='cuda', dtype=torch.int32)

        def expand():
            self.dg.expand_mxfp4_mega_moe_weights(
                l1, l2, output, active_mask=mask, expert_offset=2
            )

        expand()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            expand()
        for selected in ([0, 2], [1], []):
            mask.zero_()
            for index in selected:
                mask[index + 2] = 1
            for tensor in output:
                tensor.view(torch.uint8).fill_(0x7F)
            graph.replay()
            for actual, expected in zip(output, reference):
                for expert in range(3):
                    if expert in selected:
                        torch.testing.assert_close(
                            actual[expert].float(),
                            expected[expert].float(),
                            atol=1e-6,
                            rtol=1e-6,
                        )
                    else:
                        self.assertTrue(
                            bool((actual[expert].view(torch.uint8) == 0x7F).all())
                        )

    def test_overlapping_buffers_are_rejected(self):
        ids = torch.arange(8, device='cuda', dtype=torch.int32)
        expected = ids.clone()
        for mask in (ids, ids[1:]):
            with (
                self.subTest(mask_size=mask.numel()),
                self.assertRaisesRegex(RuntimeError, 'single memory location'),
            ):
                self.dg.build_mxfp4_mega_moe_expert_mask(ids, mask)
            torch.testing.assert_close(ids, expected)
        l1, l2 = self.make_weights()
        output = tuple(
            torch.empty((*w.shape[:3], 32768), device='cuda', dtype=torch.float8_e4m3fn)
            for w, _, _ in (l1, l2)
        )
        for expanded in output:
            mask = expanded.view(torch.int32).flatten()[:8]
            with (
                self.subTest(shape=expanded.shape),
                self.assertRaisesRegex(RuntimeError, 'single memory location'),
            ):
                self.dg.expand_mxfp4_mega_moe_weights(
                    l1, l2, output, active_mask=mask, expert_offset=2
                )


if __name__ == '__main__':
    unittest.main()
