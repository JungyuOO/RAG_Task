from __future__ import annotations

import unittest
from pathlib import Path

from apps.api.core.runtime_persistence import load_runtime_persistence_profile


class RuntimePersistenceProfileTests(unittest.TestCase):
    def test_default_profile_uses_development_semantics(self) -> None:
        profile = load_runtime_persistence_profile(
            cwd=Path("C:/workspace/rag-task"),
            environ={},
        )

        self.assertEqual(profile.mode, "development")
        self.assertTrue(profile.legacy_migration_enabled)
        self.assertFalse(profile.require_managed_secret_backend)
        self.assertEqual(profile.state_root, Path("C:/workspace/rag-task/data/runtime_state"))
        self.assertEqual(profile.state_db_path, Path("C:/workspace/rag-task/data/runtime_state/runtime_state.sqlite3"))
        self.assertIsNotNone(profile.legacy_json_path("connection_profiles.json"))

    def test_production_profile_disables_legacy_migration_and_prefers_env_key(self) -> None:
        profile = load_runtime_persistence_profile(
            cwd=Path("C:/workspace/rag-task"),
            environ={
                "RAG_TASK_RUNTIME_PERSISTENCE_MODE": "production",
                "RAG_TASK_SECRET_MASTER_KEY": "unit-test-master-key",
            },
        )

        self.assertEqual(profile.mode, "production")
        self.assertFalse(profile.legacy_migration_enabled)
        self.assertTrue(profile.require_managed_secret_backend)
        self.assertEqual(profile.secret_backend, "env_key")
        self.assertEqual(profile.secret_backend_source, "managed-default")
        self.assertIsNone(profile.legacy_json_path("connection_profiles.json"))

    def test_production_profile_requires_managed_secret_backend(self) -> None:
        with self.assertRaises(RuntimeError):
            load_runtime_persistence_profile(
                cwd=Path("C:/workspace/rag-task"),
                environ={"RAG_TASK_RUNTIME_PERSISTENCE_MODE": "production"},
            )

    def test_production_profile_rejects_unmanaged_secret_backend(self) -> None:
        with self.assertRaises(RuntimeError):
            load_runtime_persistence_profile(
                cwd=Path("C:/workspace/rag-task"),
                environ={
                    "RAG_TASK_RUNTIME_PERSISTENCE_MODE": "production",
                    "RAG_TASK_SECRET_BACKEND": "dpapi",
                },
            )

    def test_explicit_paths_override_defaults(self) -> None:
        profile = load_runtime_persistence_profile(
            cwd=Path("C:/workspace/rag-task"),
            environ={
                "RAG_TASK_RUNTIME_STATE_DIR": "D:/rag/runtime",
                "RAG_TASK_RUNTIME_DB_PATH": "D:/rag/runtime/ops.sqlite3",
                "RAG_TASK_RUNTIME_SECRET_PATH": "D:/rag/runtime/secrets.protected.json",
                "RAG_TASK_RUNTIME_SECRET_REFS_PATH": "D:/rag/runtime/secret_refs.json",
            },
        )

        self.assertEqual(profile.state_root, Path("D:/rag/runtime"))
        self.assertEqual(profile.state_db_path, Path("D:/rag/runtime/ops.sqlite3"))
        self.assertEqual(profile.secret_storage_path, Path("D:/rag/runtime/secrets.protected.json"))
        self.assertEqual(profile.secret_refs_path, Path("D:/rag/runtime/secret_refs.json"))


if __name__ == "__main__":
    unittest.main()


