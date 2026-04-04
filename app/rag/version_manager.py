import re
from pathlib import Path


class VersionManager:
    VERSION_PATTERN = re.compile(r'(\d+\.\d+)')
    OCP_PATTERN = re.compile(r'(?:ocp|openshift)[- _]?(\d+\.\d+)', re.IGNORECASE)
    FOLDER_PATTERN = re.compile(r'ocp-(\d+\.\d+)')

    def detect_version(self, filename: str) -> str | None:
        match = self.OCP_PATTERN.search(filename)
        if match:
            return match.group(1)
        return None

    def detect_version_from_path(self, path: Path) -> str | None:
        """폴더명(ocp-4.15) 또는 파일명에서 버전 감지."""
        folder_match = self.FOLDER_PATTERN.match(path.parent.name)
        if folder_match:
            return folder_match.group(1)
        return self.detect_version(path.name)

    def detect_version_from_content(self, content: str) -> str | None:
        match = re.search(r'(?:OpenShift|OCP)\s+(?:Container\s+Platform\s+)?(\d+\.\d+)', content)
        if match:
            return match.group(1)
        return None

    def filter_by_version(self, chunks: list, target_versions: list, version_map: dict) -> list:
        if not target_versions:
            return chunks
        target_ids = {vid for vid, vtag in version_map.items() if vtag in target_versions}
        return [c for c in chunks if c.get("version_id") in target_ids]
