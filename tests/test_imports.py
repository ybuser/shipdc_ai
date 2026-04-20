from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class ImportTest(TestCase):
    def test_import_public_modules(self) -> None:
        import shipdc_ai.cheap_filter
        import shipdc_ai.detector_runtime
        import shipdc_ai.event_store
        import shipdc_ai.manifest
        import shipdc_ai.nvr_emulator
        import shipdc_ai.paths
        import shipdc_ai.temporal_verifier

        self.assertTrue(shipdc_ai.manifest.MANIFEST_HEADER)
