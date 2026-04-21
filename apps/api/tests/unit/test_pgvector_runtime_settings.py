from __future__ import annotations

import unittest
from pathlib import Path

from apps.api.core.pgvector_settings import PgvectorRuntimeSettings


class PgvectorRuntimeSettingsTests(unittest.TestCase):
    def test_db_dsn_and_paths_are_derived_in_apps_api_settings(self) -> None:
        settings = PgvectorRuntimeSettings.model_validate(
            {
                "rag_source_dir": str(Path("C:/repo/data/source")),
                "embedding_backend": "ollama",
                "ollama_base_url": "http://localhost:11434",
                "ollama_embedding_model": "bge-m3",
                "ollama_timeout": 120.0,
                "db_host": "localhost",
                "db_port": 5432,
                "db_name": "rag",
                "db_user": "postgres",
                "db_password": "secret",
            }
        )

        self.assertEqual(settings.db_dsn, "host=localhost port=5432 dbname=rag user=postgres password=secret")
        self.assertEqual(settings.rag_source_dir, Path("C:/repo/data/source"))


if __name__ == "__main__":
    unittest.main()


