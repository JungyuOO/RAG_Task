import json


class VersionStore:
    """문서 버전 태그 관리"""

    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS document_versions (
        version_id SERIAL PRIMARY KEY,
        source_path TEXT NOT NULL,
        file_name TEXT NOT NULL,
        version_tag TEXT NOT NULL,
        product TEXT DEFAULT 'OCP',
        uploaded_at TIMESTAMP DEFAULT NOW(),
        metadata_json JSONB DEFAULT '{}'
    )
    """

    def __init__(self, pool):
        self.pool = pool

    async def ensure_table(self):
        async with self.pool.acquire() as conn:
            await conn.execute(self.CREATE_TABLE_SQL)

    async def create_version(self, source_path, file_name, version_tag, product="OCP", metadata=None):
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO document_versions (source_path, file_name, version_tag, product, metadata_json)
                   VALUES ($1, $2, $3, $4, $5) RETURNING version_id""",
                source_path, file_name, version_tag, product,
                json.dumps(metadata or {})
            )
            return dict(row)

    async def list_versions(self, product=None):
        async with self.pool.acquire() as conn:
            if product:
                rows = await conn.fetch(
                    "SELECT * FROM document_versions WHERE product=$1 ORDER BY uploaded_at DESC", product
                )
            else:
                rows = await conn.fetch("SELECT * FROM document_versions ORDER BY uploaded_at DESC")
            return [dict(r) for r in rows]

    async def get_by_tag(self, version_tag, product="OCP"):
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM document_versions WHERE version_tag=$1 AND product=$2",
                version_tag, product
            )
            return [dict(r) for r in rows]

    async def delete_version(self, version_id):
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM document_versions WHERE version_id=$1", version_id)
