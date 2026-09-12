"""CPU coverage checks using the actual attention case enumerators, without Torch.

Run: python3 -m unittest discover -s sgl_deep_gemm -p test_release_test_profile.py
"""

import ast
import contextlib
import io
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


TESTS_DIR = Path(__file__).parent / 'tests'
sys.path.insert(0, str(TESTS_DIR))
ATTENTION_SOURCE = ast.parse((TESTS_DIR / 'test_attention.py').read_text())


def attention_function(name, arch_major=10):
    # Execute only the CPU enumerator/helper, not the GPU test module imports.
    node = next(node for node in ast.walk(ATTENTION_SOURCE)
                if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {
        'torch': SimpleNamespace(bfloat16='bf16', float='fp32'),
        'deep_gemm': SimpleNamespace(get_num_sms=lambda: 148),
        'get_arch_major': lambda: arch_major,
        'os': os, 'random': random, 'List': list,
    }
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(TESTS_DIR / 'test_attention.py'), 'exec'), namespace)
    return namespace[name]


def original_cases(name, arch_major=10):
    enumerator = {
        'prefill': 'enumerate_mqa_logits',
        'paged': 'enumerate_paged_mqa_logits',
        'sparse': 'enumerate_sparse_mqa_logits',
    }[name]
    return list(attention_function(enumerator, arch_major)())


def sample(name, cases, profile=None, num_cases=None):
    env = {}
    if profile is not None:
        env['DG_TEST_PROFILE'] = profile
    if num_cases is not None:
        env['DG_MQA_NUM_CASES'] = str(num_cases)
    with patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(io.StringIO()):
        return attention_function('sample_mqa_cases')(name, cases)


class ReleaseTestProfileTests(unittest.TestCase):
    def test_default_full_profile_preserves_all_cases_and_order(self):
        for name in ('prefill', 'paged', 'sparse'):
            cases = original_cases(name)
            for profile in (None, 'full'):
                self.assertIs(sample(name, cases, profile), cases)

    def test_legacy_random_sampling_is_unchanged(self):
        for name, seed in [('prefill', 0), ('paged', 100000), ('sparse', 200000)]:
            cases = original_cases(name)
            self.assertEqual(sample(name, cases, num_cases=7), random.Random(seed).sample(cases, 7))
            self.assertEqual(sample(name, cases, num_cases=0), [])
            self.assertEqual(len(sample(name, cases, num_cases=len(cases) + 1)), len(cases))

    def test_release_is_bounded_and_reuses_original_case_objects(self):
        expected = {9: (8, 8), 10: (36, 72), 12: (16, 16)}
        for arch, counts in expected.items():
            for name, count in zip(('prefill', 'paged'), counts):
                with self.subTest(arch=arch, name=name):
                    cases = original_cases(name, arch)
                    selected = sample(name, cases, 'release')
                    self.assertEqual(len(selected), count)
                    self.assertEqual(len(set(selected)), count)
                    self.assertTrue(all(any(case is source for source in cases) for case in selected))
                    self.assertEqual(sample(name, cases, 'release'), selected)

    def test_dense_preserves_dtype_layout_and_shape_families(self):
        for arch in (9, 10, 12):
            cases = original_cases('prefill', arch)
            selected = sample('prefill', cases, 'release')
            family = lambda c: (*c[:5], c[9])
            self.assertEqual({family(c) for c in selected}, {family(c) for c in cases})
            for fmt in {c[0] for c in cases}:
                actual = [c for c in selected if c[0] == fmt]
                source = [c for c in cases if c[0] == fmt]
                for index in (5, 6, 7, 8):
                    self.assertEqual({c[index] for c in actual}, {c[index] for c in source})
                # Large Q and long KV each remain represented without combining both costs.
                self.assertFalse(any(c[5] == 8192 and c[6] == 65536 for c in actual))

    def test_paged_preserves_dtype_layout_and_token_families(self):
        for arch in (9, 10, 12):
            cases = original_cases('paged', arch)
            selected = sample('paged', cases, 'release')
            self.assertEqual({c[:7] for c in selected}, {c[:7] for c in cases})
            for family in {c[:4] for c in cases}:
                actual = [c for c in selected if c[:4] == family]
                source = [c for c in cases if c[:4] == family]
                for index in (8, 9, 11):
                    self.assertEqual({c[index] for c in actual}, {c[index] for c in source})
            for fmt in {c[1] for c in cases}:
                actual = [c for c in selected if c[1] == fmt]
                self.assertEqual({c[12] for c in actual}, {8192, 65536})
            self.assertEqual({c[7] for c in selected}, {256})

    def test_sparse_covers_existing_small_and_long_context_edges(self):
        cases = original_cases('sparse')
        selected = sample('sparse', cases, 'release')
        self.assertEqual(len(selected), 12)
        self.assertEqual(len(set(selected)), 12)
        self.assertTrue(all(any(c is source for source in cases) for c in selected))
        for fmt in ('mxfp4', 'mxfp8'):
            actual = [c for c in selected if c[0] == fmt]
            self.assertEqual({c[1] for c in actual}, {False, True})
            contiguous = [c for c in actual if not c[1]]
            self.assertEqual({(c[4], c[6]) for c in contiguous},
                             {(8, False), (8, True), (16, False), (16, True)})
        self.assertTrue(any(c[3] == 0 for c in selected))
        self.assertTrue(any(c[1] and c[3] == 1024 * 1024 for c in selected))
        self.assertLessEqual(max(c[2] for c in selected), 9)
        self.assertEqual(sample('sparse', cases, 'release'), selected)

    def test_release_rejects_random_cap_and_unknown_profile(self):
        cases = original_cases('prefill')
        with self.assertRaises(ValueError):
            sample('prefill', cases, 'release', num_cases=1)
        with self.assertRaises(ValueError):
            sample('prefill', cases, 'typo')


class ReleaseRunnerTests(unittest.TestCase):
    def run_runner(self, *, release, arch=10, fail_sanitizer=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tests_dir = root / 'sgl_deep_gemm' / 'tests'
            tests_dir.mkdir(parents=True)
            for filename in ('test_attention.py', 'test_sanitizer.py'):
                (tests_dir / filename).touch()
            # GPU and Python are process boundaries: exercise the real shell
            # dispatch without importing a wheel or executing GPU test bodies.
            nvidia_smi = root / 'nvidia-smi'
            nvidia_smi.write_text(
                '#!/bin/sh\nif [ "$1" = "-L" ]; then echo GPU; '
                f'else echo {arch}.0; fi\n')
            nvidia_smi.chmod(0o755)
            python = root / 'test-python'
            python.write_text(
                f'#!{sys.executable}\n'
                'import json, os, sys\n'
                'if "-c" in sys.argv:\n'
                '    print("/installed/deep_gemm/__init__.py")\n'
                'else:\n'
                '    print("DISPATCH " + json.dumps([os.getenv("DG_TEST_PROFILE"), sys.argv[1:]]))\n'
                '    if "test_sanitizer.py" in sys.argv and os.getenv("FAIL_SANITIZER") == "1":\n'
                '        sys.exit(7)\n')
            python.chmod(0o755)
            env = os.environ.copy()
            env.pop('DG_MQA_NUM_CASES', None)
            env.update(PATH=f'{root}{os.pathsep}{env["PATH"]}', PYTHON=str(python),
                       FAIL_SANITIZER=str(int(fail_sanitizer)))
            cmd = ['bash', str(TESTS_DIR.parent / 'run_tests.sh'), str(root), '--skip-mega-moe']
            if release:
                cmd.append('--release')
            result = subprocess.run(cmd, env=env, text=True, capture_output=True)
            return result

    def test_release_runs_explicit_sanitizer_functions_with_both_tools(self):
        import json

        for arch in (9, 10):
            result = self.run_runner(release=True, arch=arch)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = [json.loads(line.removeprefix('DISPATCH '))
                     for line in result.stdout.splitlines() if line.startswith('DISPATCH ')]
            self.assertTrue(calls)
            self.assertTrue(all(profile == 'release' for profile, _ in calls))
            sanitizer = next(args for _, args in calls if 'test_sanitizer.py' in args)
            expected = 'test_attention.test_gemm_skip_head_mid'
            if arch == 10:
                expected += ',test_clean_logits_bounds.test_clean_logits_bounds'
            self.assertEqual(sanitizer, ['-u', 'test_sanitizer.py', '--funcs', expected,
                                         '--tools', 'memcheck,synccheck'])

    def test_default_runner_retains_full_sanitizer_discovery(self):
        result = self.run_runner(release=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('DISPATCH ["full", ["-u", "test_sanitizer.py"]]', result.stdout)

    def test_release_propagates_sanitizer_failure(self):
        result = self.run_runner(release=True, fail_sanitizer=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('FAIL test_sanitizer.py (exit 7)', result.stdout)


if __name__ == '__main__':
    unittest.main()
