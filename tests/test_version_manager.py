from __future__ import annotations

import unittest
from pathlib import Path

from app.rag.version_manager import VersionManager


class VersionManagerTests(unittest.TestCase):
    def test_detect_version_from_nested_parent_path(self) -> None:
        manager = VersionManager()
        path = Path("/docs/ocp-4.20-openshift-docs/installing/aws/install-config.md")
        self.assertEqual(manager.detect_version_from_path(path), "4.20")

    def test_detect_version_from_file_name_fallback(self) -> None:
        manager = VersionManager()
        path = Path("/docs/misc/OpenShift_Container_Platform-4.20-Architecture-en-US.md")
        self.assertEqual(manager.detect_version_from_path(path), "4.20")


if __name__ == "__main__":
    unittest.main()
