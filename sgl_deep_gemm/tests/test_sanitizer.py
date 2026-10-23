import argparse
import importlib
import inspect
import os
import subprocess
import sys

import deep_gemm


# Single test template
script_dir = os.path.dirname(os.path.abspath(__file__))
test_template = """
import random
import sys
import torch

# Necessary for `generators.py`
sys.path.append('{script_dir}')

torch.manual_seed(0)
random.seed(0)

from {module_name} import {func_name}
{func_name}()
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--funcs', type=str, default='all')
    parser.add_argument('--tools', type=str, default='memcheck,synccheck')
    args = parser.parse_args()

    if args.funcs != 'all':
        funcs = []
        for name in [x.strip() for x in args.funcs.split(',')]:
            module_name, func_name = name.split('.')
            funcs.append((module_name, func_name))
    else:
        # Distributed entrypoints need rank arguments and their own launcher.
        # Keep this runner scoped to callable single-process tests and report
        # every excluded distributed entrypoint instead of calling it as test().
        funcs = []
        files = sorted(f for f in os.listdir(script_dir)
                       if f.startswith('test_') and f.endswith('.py') and f != 'test_sanitizer.py')
        arch_major = deep_gemm.testing.get_arch_major()
        for filename in files:
            if ((arch_major == 12 and filename in
                 ('test_hyperconnection.py', 'test_layout.py', 'test_legacy.py'))
                    or (arch_major != 10 and filename in ('test_mega_gate.py', 'test_mega_mhc.py'))):
                print(f' > Skipped {filename}: unsupported on architecture {arch_major}')
                continue
            module_name = os.path.splitext(filename)[0]
            for name, obj in inspect.getmembers(importlib.import_module(module_name)):
                if not (inspect.isfunction(obj) and name.startswith('test')
                        and obj.__module__ == module_name):
                    continue
                try:
                    inspect.signature(obj).bind()
                except TypeError:
                    print(f' > Skipped {module_name}.{name}: requires a distributed launcher')
                    continue
                funcs.append((module_name, name))
    tools = [x.strip() for x in args.tools.split(',')]

    env = os.environ.copy()
    env['CUDA_LAUNCH_BLOCKING'] = '1'
    env['DG_JIT_CHECK_NO_SPILLS'] = '1'
    env['DG_USE_NVIDIA_TOOLS'] = '1'
    env['DG_USE_TEMP_CUBLASLT_WORKSPACE'] = '1'  # Avoid holding CUDA tensor that crashes during shutdown
    env['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
    env['TORCH_SHOW_CPP_STACKTRACES'] = '1'

    print(f'Library path: {deep_gemm.__path__}')
    for module_name, func_name in funcs:
        for tool in tools:
            run_env = env.copy()
            if (module_name, func_name) == ('test_mega_mhc', 'test_mega_mhc_graph_api_contract'):
                # Capture allocations require PyTorch's graph-private caching pool.
                # Keep every non-graph API and bounds check on the uncached allocator.
                run_env['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '0'
                print(' > CUDA caching enabled for mHC graph capture; all sanitizer checks remain enabled')
            cmd = [
                os.environ.get('COMPUTE_SANITIZER', '/usr/local/cuda/bin/compute-sanitizer'),
                f'--tool={tool}',
                '--error-exitcode=1',
                '--target-processes=application-only',
                '--destroy-on-device-error=context',
                '--force-blocking-launches',
                '--check-api-memory-access=no',
                '--kernel-name-exclude', 'kns=nvjet',
                sys.executable,
                '-c',
                test_template.format(module_name=module_name, func_name=func_name, script_dir=script_dir)
            ]
            print(f'\n{"=" * 60}')
            print(f'Running {module_name}.{func_name} with compute-sanitizer {tool}')
            result = subprocess.run(cmd, env=run_env)
            if result.returncode != 0:
                sys.exit(result.returncode)
