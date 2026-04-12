from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from scripts.import_openshift_docs import (
    build_output_path,
    collect_adoc_files,
    convert_adoc_to_markdown,
    import_openshift_docs,
    import_output_dir,
)


class ImportOpenshiftDocsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("tests/.tmp_import_openshift_docs")
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "repo" / "installing").mkdir(parents=True, exist_ok=True)
        (self.root / "repo" / "networking" / "ovn_k").mkdir(parents=True, exist_ok=True)
        self.adoc_a = self.root / "repo" / "installing" / "install-config.adoc"
        self.adoc_b = self.root / "repo" / "networking" / "ovn_k" / "about.adoc"
        self.adoc_a.write_text("= Install config\n\nBody\n", encoding="utf-8")
        self.adoc_b.write_text("= About OVN-Kubernetes\n\nBody\n", encoding="utf-8")

    def tearDown(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def test_collect_adoc_files_reads_selected_sections(self) -> None:
        files = collect_adoc_files(self.root / "repo", ["installing"])
        self.assertEqual(files, [self.adoc_a.resolve()])

    def test_convert_adoc_to_markdown_cleans_directives_and_preserves_structure(self) -> None:
        adoc = """= Install guide
:context: install
toc::[]

NOTE: Keep this in mind.

== Prerequisites

* first
* second
. third
. fourth

[source,yaml]
----
kind: Pod
metadata:
  name: demo
----

|===
| Name | Value
| Pod | demo
|===
"""
        markdown = convert_adoc_to_markdown(adoc)
        self.assertNotIn(":context:", markdown)
        self.assertNotIn("toc::[]", markdown)
        self.assertIn("# Install guide", markdown)
        self.assertIn("## Prerequisites", markdown)
        self.assertIn("> NOTE: Keep this in mind.", markdown)
        self.assertIn("- first", markdown)
        self.assertIn("1. third", markdown)
        self.assertIn("```yaml", markdown)
        self.assertIn("| Name | Value |", markdown)

    def test_import_openshift_docs_writes_markdown_tree(self) -> None:
        output_dir = import_output_dir(self.root, "4.20")
        created = import_openshift_docs(self.root / "repo", output_dir, ["installing", "networking"], clean=True)
        self.assertEqual(len(created), 2)
        self.assertTrue((output_dir / "installing" / "install-config.md").exists())
        self.assertTrue((output_dir / "networking" / "ovn_k" / "about.md").exists())

    def test_build_output_path_preserves_relative_tree(self) -> None:
        output = build_output_path(self.root / "repo", self.adoc_b, self.root / "out")
        self.assertEqual(output, self.root / "out" / "networking" / "ovn_k" / "about.md")


if __name__ == "__main__":
    unittest.main()
