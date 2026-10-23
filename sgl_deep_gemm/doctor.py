"""`python -m deep_gemm.doctor` -- one-shot health check for this install.

Prints everything needed to diagnose "why doesn't DeepGEMM work here" in one
place, and exits non-zero if anything is actually broken. Deliberately does not
print a hash of ``_C.so``: the binary embeds its build-time source path, so the
hash differs between byte-identical builds and only invites false alarms.
"""
from __future__ import annotations

import os
import sys


def _line(label: str, value: str, ok: bool | None = None) -> None:
    mark = '' if ok is None else ('  [ok]' if ok else '  [!!]')
    print(f'{label:<22}: {value}{mark}')


def main() -> int:
    problems: list[str] = []
    warnings: list[str] = []

    print('DeepGEMM doctor')
    print('=' * 60)

    # --- package ----------------------------------------------------------
    try:
        import deep_gemm
    except Exception as exc:  # noqa: BLE001 - this is the diagnostic path
        print(f'\nimport deep_gemm FAILED: {type(exc).__name__}: {exc}')
        return 1

    pkg_dir = os.path.dirname(os.path.abspath(deep_gemm.__file__))
    _line('version', getattr(deep_gemm, '__version__', '(unknown)'))
    _line('package', pkg_dir)
    lib_path = os.path.join(pkg_dir, '_C.so')
    if os.path.exists(lib_path):
        size = os.path.getsize(lib_path)
        import datetime
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(lib_path)).strftime('%Y-%m-%d %H:%M')
        _line('_C.so', f'{lib_path} ({size:,} bytes, built {mtime})', True)
    else:
        _line('_C.so', 'MISSING', False)
        problems.append('_C.so is missing from the installed package')

    # --- CUDA install -----------------------------------------------------
    from .cuda_helpers import cuda_diagnostics
    diag = cuda_diagnostics()
    if diag['error']:
        _line('CUDA_HOME', diag['error'], False)
        problems.append('CUDA install not found')
    else:
        _line('CUDA_HOME', diag['cuda_home'], True)
        _line('nvcc', diag['nvcc'] or 'MISSING', bool(diag['nvcc']))
        if not diag['nvcc']:
            problems.append('nvcc not found under CUDA_HOME/bin')
        want_headers = diag['cuda_home'] and os.path.join(diag['cuda_home'], 'include', 'curand_kernel.h')
        _line('device headers', diag['device_headers'] or 'MISSING (JIT will fail later)', bool(diag['device_headers']))
        if not diag['device_headers']:
            warnings.append('CUDA device headers missing; the runtime JIT needs curand_kernel.h')

    # --- torch / driver ---------------------------------------------------
    try:
        import torch
        _line('torch', f'{torch.__version__} (cuda {torch.version.cuda})', True)
    except Exception as exc:  # noqa: BLE001
        _line('torch', f'import failed: {exc}', False)
        problems.append('torch import failed')
        torch = None

    # --- architecture capability (no CUDA context needed) -----------------
    try:
        majors = deep_gemm.supported_arch_majors()
        _line('supported arch majors', ', '.join(f'sm{m}' for m in majors), True)
    except Exception as exc:  # noqa: BLE001
        majors = None
        _line('supported arch majors', f'unavailable: {exc}', False)
        problems.append('capability query unavailable (stale _C.so?)')

    # --- device -----------------------------------------------------------
    cap = None
    if torch is not None:
        try:
            if not torch.cuda.is_available():
                _line('device', 'no CUDA device visible', False)
                problems.append('torch.cuda.is_available() is False')
            else:
                props = torch.cuda.get_device_properties(0)
                cap = torch.cuda.get_device_capability(0)
                _line('device', f'{props.name}, sm_{cap[0]}{cap[1]} '
                                 f'(compute capability {cap[0]}.{cap[1]}), {props.multi_processor_count} SMs', True)
        except Exception as exc:  # noqa: BLE001
            _line('device', f'query failed: {exc}', False)
            problems.append('could not query the CUDA device')

    # --- does this build accept this device? ------------------------------
    if cap is not None and majors is not None:
        wants = cap[0]
        if wants in majors:
            _line('arch check', f'sm{wants} is supported by this build', True)
        else:
            _line('arch check', f'sm{wants} is NOT in {list(majors)} -- dispatch will refuse', False)
            problems.append(f'this _C.so does not accept sm{wants}; rebuild the host layer')

    # --- JIT cache --------------------------------------------------------
    cache_env = os.environ.get('DG_JIT_CACHE_DIR')
    if cache_env:
        roots = cache_env.split(':')
        _line('JIT cache', f'{cache_env}  (writable root: {roots[0]}, '
                           f'{len(roots) - 1} read-only lookup root(s))')
    else:
        _line('JIT cache', os.path.join(os.path.expanduser('~'), '.dj') + '  (default; set DG_JIT_CACHE_DIR to move)')
    if majors and cap is not None and cap[0] in majors:
        for root in ([cache_env.split(':')[0]] if cache_env else [os.path.join(os.path.expanduser('~'), '.dj')]):
            cache_dir = os.path.join(root, 'cache')
            try:
                count = len(os.listdir(cache_dir)) if os.path.isdir(cache_dir) else 0
            except OSError:
                count = 0
            _line('JIT cached kernels', f'{count} entries in {cache_dir}')

    # --- live dispatch: one tiny GEMM -------------------------------------
    if torch is not None and cap is not None and majors is not None and cap[0] in majors:
        try:
            print('\nrunning a 128x128x128 bf16 GEMM through the dispatcher ...')
            a = torch.randn(128, 128, dtype=torch.bfloat16, device='cuda')
            b = torch.randn(128, 128, dtype=torch.bfloat16, device='cuda')
            d = torch.empty(128, 128, dtype=torch.bfloat16, device='cuda')
            deep_gemm.bf16_gemm_nt(a, b, d)
            torch.cuda.synchronize()
            ref = (a.float() @ b.float().t()).to(torch.bfloat16)
            max_diff = (d.float() - ref.float()).abs().max().item()
            finite = bool(torch.isfinite(d).all().item())
            ok = finite and max_diff < 1.0
            _line('dispatch', f'bf16_gemm_nt ran, max|diff| vs fp32 ref = {max_diff:.4f}', ok)
            if not ok:
                problems.append('GEMM produced non-finite or clearly wrong output')
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).splitlines()[0]
            _line('dispatch', f'FAILED: {msg}', False)
            problems.append(f'live GEMM failed: {msg}')

    # --- verdict ----------------------------------------------------------
    print('=' * 60)
    for w in warnings:
        print(f'warning: {w}')
    if problems:
        for p in problems:
            print(f'PROBLEM: {p}')
        print('\nRESULT: FAIL')
        return 1
    if warnings:
        print('\nRESULT: OK (with warnings)')
        return 0
    print('\nRESULT: OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
