"""CPU-only regression checks for the fork's pybind argument adapters.

Extract only the adapters under test: no extension build or CUDA context needed.
Run with: python tests/test_python_wrapper_contracts.py
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().parents[1] / 'sgl_deep_gemm' / '__init__.py'
NAMES = {'fp8_einsum', 'k_grouped_fp8_gemm_tn_contiguous',
         'k_grouped_fp8_gemm_nt_contiguous'}


def load_adapters(native):
    tree = ast.parse(SOURCE.read_text())
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, ast.FunctionDef) and node.name in NAMES]
    assert {node.name for node in nodes} == NAMES
    namespace = {'_C': native, '_parse_tensor_or_tuple': lambda value: tuple(value)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class WrapperContracts(unittest.TestCase):
    def setUp(self):
        self.native = SimpleNamespace(**{name: Mock() for name in NAMES})
        self.adapters = load_adapters(self.native)
        self.a, self.b = (object(), object()), (object(), object())

    def test_einsum_preserves_plain_output(self):
        output = object()
        self.adapters['fp8_einsum']('bhr,hdr->bhd', self.a, self.b, output)
        self.native.fp8_einsum.assert_called_once_with(
            'bhr,hdr->bhd', self.a, self.b, output, c=None,
            recipe=(1, 128, 128), epilogue=None)

    def test_einsum_preserves_scaled_output_and_epilogue(self):
        output, sf, epilogue = object(), object(), object()
        self.adapters['fp8_einsum']('bhr,hdr->bhd', self.a, self.b,
                                    [output, sf], epilogue=epilogue)
        self.assertEqual(self.native.fp8_einsum.call_args.args[3], (output, sf))
        self.assertIs(self.native.fp8_einsum.call_args.kwargs['epilogue'], epilogue)

    def test_grouped_epilogue_is_not_a_bool(self):
        for name in NAMES - {'fp8_einsum'}:
            with self.subTest(name=name):
                output, layout, epilogue = object(), object(), object()
                self.adapters[name](self.a, self.b, output, [128], layout)
                op = getattr(self.native, name)
                self.assertIsNone(op.call_args.kwargs['epilogue'])
                self.assertEqual(len(op.call_args.args), 5)
                self.adapters[name](self.a, self.b, output, [128], layout,
                                    epilogue=epilogue)
                self.assertIs(op.call_args.kwargs['epilogue'], epilogue)
                with self.assertRaises(NotImplementedError):
                    self.adapters[name](self.a, self.b, output, [128], layout,
                                        use_padded_sf_layout=True)

    def test_native_typeerror_is_not_retried(self):
        for name in NAMES:
            with self.subTest(name=name):
                op = getattr(self.native, name)
                op.side_effect = TypeError('native error must survive')
                with self.assertRaisesRegex(TypeError, 'native error must survive'):
                    if name == 'fp8_einsum':
                        self.adapters[name]('bhr,hdr->bhd', self.a, self.b, object())
                    else:
                        self.adapters[name](self.a, self.b, object(), [128], object())
                self.assertEqual(op.call_count, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
