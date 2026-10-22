import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


class TestMHCReferenceImport(unittest.TestCase):
    def load_reference(self):
        third_party = Path(__file__).resolve().parents[2] / 'third-party'
        with patch.object(sys, 'path', [str(third_party), *sys.path]):
            from tilelang_ops import ref_mhc
        return ref_mhc

    def test_release_dependency_check(self):
        self.assertTrue(self.load_reference().has_baseline())

    def test_missing_reference_dependency(self):
        ref_mhc = self.load_reference()
        self.assertTrue(ref_mhc.has_baseline())
        output = io.StringIO()
        with patch.dict(sys.modules, {'deep_gemm.testing': None}):
            with contextlib.redirect_stdout(output):
                self.assertFalse(ref_mhc.has_baseline())
        self.assertIn('deep_gemm.testing', output.getvalue())


if __name__ == '__main__':
    unittest.main()
