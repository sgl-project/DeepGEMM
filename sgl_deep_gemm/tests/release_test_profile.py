"""Deterministic release coverage selected from the original MQA enumerators.

This module deliberately has no Torch dependency. It selects existing tuple
objects; input generation, numerical checks, and repeated-call checks stay in
test_attention.py. The default profile preserves the complete enumeration.
"""

from collections import defaultdict
import random


def _groups(cases, key):
    groups = defaultdict(list)
    for case in cases:
        groups[key(case)].append(case)
    return groups.values()


def _pick(cases, fields):
    for case in cases:
        if all(case[index] == value for index, value in fields.items()):
            return case
    raise ValueError(f'Release MQA representative is absent from the original enumeration: {fields}')


def _prefill_cases(cases):
    selected = []
    format_groups = defaultdict(int)
    # Four representatives per format/dtype pair: compressed/full x CP/no CP.
    for family in _groups(cases, lambda c: c[:3]):
        fmt = family[0][0]
        group_index = format_groups[fmt]
        format_groups[fmt] += 1
        heads = sorted({c[7] for c in family})
        dims = sorted({c[8] for c in family})
        for slot, layout in enumerate(_groups(family, lambda c: (*c[3:5], c[9]))):
            fields = {
                5: min(c[5] for c in layout),
                6: min(c[6] for c in layout),
                7: heads[(4 * group_index + slot) % len(heads)],
                8: dims[(group_index + slot) % len(dims)],
            }
            # One long-KV and one large-Q case per format; keep them separate.
            if group_index == 0 and slot == 0:
                fields[6] = max(c[6] for c in layout)
            elif group_index == 0 and slot == 1:
                fields[5] = max(c[5] for c in layout)
            selected.append(_pick(layout, fields))
    return selected


def _paged_cases(cases):
    selected = []
    format_groups = defaultdict(int)
    # Four representatives per normal/varlen + format/dtype pair. Cover every
    # page size and both NextN (normal) or max tokens per request (varlen) values.
    for family in _groups(cases, lambda c: c[:4]):
        is_varlen, fmt = family[0][:2]
        group_index = format_groups[fmt]
        format_groups[fmt] += 1
        blocks = sorted({c[4] for c in family})
        heads = sorted({c[10] for c in family})
        dims = sorted({c[11] for c in family})
        token_index = 9 if is_varlen else 8
        tokens = sorted({c[token_index] for c in family})
        for slot in range(4):
            fields = {
                4: blocks[slot % len(blocks)],
                7: min(c[7] for c in family),
                token_index: tokens[slot % len(tokens)],
                10: heads[(2 * group_index + slot // 2) % len(heads)],
                11: dims[(group_index + slot) % len(dims)],
                12: min(c[12] for c in family),
            }
            # Long KV uses the smallest batch and one token, once per format.
            if group_index == 0 and slot == 0:
                fields[12] = max(c[12] for c in family)
            selected.append(_pick(family, fields))
    return selected


def _sparse_cases(cases):
    # Existing small-Q cases retain both formats, KV layouts, block sizes,
    # alignment modes, empty context, and a long paged context. Selecting by the
    # complete tuple fails loudly if an upstream enumerator removes a case.
    representatives = [
        (fmt, False, 9, split_kv, 8, 128, unaligned)
        for fmt, split_kv in [('mxfp4', 640), ('mxfp8', 512)]
        for unaligned in (False, True)
    ] + [
        (fmt, False, 3, 640, 16, 1024, unaligned)
        for fmt in ('mxfp4', 'mxfp8')
        for unaligned in (False, True)
    ] + [
        ('mxfp4', False, 2, 0, 16, 4, False),
        ('mxfp4', True, 3, 64, 16, 8, False),
        ('mxfp8', True, 3, 64, 16, 8, False),
        ('mxfp4', True, 3, 1024 * 1024, 8, 2048, False),
    ]
    return [_pick(cases, dict(enumerate(case))) for case in representatives]


def select_mqa_cases(name, cases, *, profile='full', num_cases=None):
    if profile not in ('full', 'release'):
        raise ValueError(f'Unknown DG_TEST_PROFILE: {profile!r}; expected full or release')
    if profile == 'release':
        if num_cases is not None:
            raise ValueError('DG_MQA_NUM_CASES cannot be combined with DG_TEST_PROFILE=release')
        return {'prefill': _prefill_cases, 'paged': _paged_cases, 'sparse': _sparse_cases}[name](cases)
    if num_cases is None:
        return cases
    rng = random.Random({'prefill': 0, 'paged': 100000, 'sparse': 200000}[name])
    return rng.sample(cases, min(int(num_cases), len(cases)))
