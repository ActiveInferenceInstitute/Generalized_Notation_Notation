#!/usr/bin/env python3
"""
LLM Response Cache Module

Content-hash caching for LLM responses. Eliminates redundant API calls
when running the pipeline on unchanged GNN files.

Cache key: sha256(file_content + model_name + prompt_template)
Storage: output/13_llm_output/.cache/<hash>.json

Publications, rollback and eviction serialize per key, never across provider
requests. Permanent lock files survive cache clearing.
Each publication has a distinct rollback identity, separate from request/evidence
identity, so an identical newer response cannot be retracted by an older request.
"""

import hashlib
import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast
from uuid import uuid4

from gnn.pipeline._io import atomic_write_text

logger = logging.getLogger(__name__)


class LLMCache:
    """Content-addressed cache for LLM prompt responses."""

    def __init__(
        self, cache_dir: Optional[Path] = None, base_output_dir: Optional[Path] = None
    ) -> None:
        """
        Initialize cache.

        Args:
            cache_dir: Explicit directory to store cached responses. Takes
                precedence over ``base_output_dir``.
            base_output_dir: If provided (and ``cache_dir`` is None), the
                cache root is resolved via
                ``pipeline.config.get_output_dir_for_script("13_llm.py",
                base_output_dir) / ".cache"``. This allows multi-workspace /
                multi-pipeline runs to avoid sharing one on-disk cache.
                When both are None, uses ``output/13_llm_output/.cache``
                relative to CWD.
        """
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
        elif base_output_dir is not None:
            try:
                from gnn.pipeline.config import get_output_dir_for_script

                self.cache_dir = (
                    get_output_dir_for_script("13_llm.py", Path(base_output_dir))
                    / ".cache"
                )
            except ImportError:
                self.cache_dir = Path(base_output_dir) / "13_llm_output" / ".cache"
        else:
            self.cache_dir = Path("output/13_llm_output/.cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.writes = 0

    @staticmethod
    def _make_key(file_content: str, model_name: str, prompt_template: str) -> str:
        """Generate a deterministic cache key from content + model + prompt."""
        payload = json.dumps(
            [file_content, model_name, prompt_template], ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        """Return the file path for a given cache key."""
        return self.cache_dir / f"{key}.json"

    @contextmanager
    def _key_lock(self, key: str) -> Iterator[None]:
        """Serialize one key's mutations across threads, instances and processes."""
        # Each operation opens its own handle: native locks then conflict even
        # within one process. Never unlink this inode, including during clear().
        with (self.cache_dir / f".{key}.lock").open("a+b") as handle:
            if os.name == "posix":
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            else:
                import msvcrt

                handle.seek(0)
                windows_lock = cast(Any, msvcrt)
                windows_lock.locking(handle.fileno(), windows_lock.LK_LOCK, 1)
            # Closing this independent handle releases the lock on every exit.
            yield

    def get(
        self, file_content: str, model_name: str, prompt_template: str
    ) -> Optional[str]:
        """
        Look up a cached response.

        Args:
            file_content: Raw GNN file content.
            model_name: Ollama / provider model name.
            prompt_template: The prompt text sent to the LLM.

        Returns:
            Cached response string, or None on miss.
        """
        key = self._make_key(file_content, model_name, prompt_template)
        path = self._cache_path(key)

        try:
            with self._key_lock(key):
                if path.exists():
                    try:
                        with open(path) as f:
                            entry = json.load(f)
                        if not isinstance(entry, dict):
                            raise ValueError("Cache record is not an object")
                        response = entry["response"]
                        if (
                            entry.get("schema_version") != 2
                            or entry.get("key") != key
                            or entry.get("model") != model_name
                            or not isinstance(response, str)
                            or not response.strip()
                            or entry.get("response_sha256")
                            != hashlib.sha256(response.encode("utf-8")).hexdigest()
                        ):
                            raise ValueError(
                                "Cache response identity or digest is invalid"
                            )
                        self.hits += 1
                        logger.debug(f"Cache HIT: {key[:12]}…")
                        return response
                    except (OSError, ValueError, KeyError) as exc:
                        logger.warning(f"Corrupt cache entry {key[:12]}…: {exc}")
                        try:
                            path.unlink(missing_ok=True)
                        except OSError:
                            logger.debug("Cache entry could not be removed")
        except OSError as exc:
            logger.debug("Cache entry could not be locked: %s", exc)

        self.misses += 1
        logger.debug(f"Cache MISS: {key[:12]}…")
        return None

    def put(
        self,
        file_content: str,
        model_name: str,
        prompt_template: str,
        response: str,
    ) -> Optional[str]:
        """
        Store a response in the cache.

        Args:
            file_content: Raw GNN file content.
            model_name: Ollama / provider model name.
            prompt_template: The prompt text sent to the LLM.
            response: The LLM response to cache.

        Returns:
            Exact publication text for conditional rollback, or None on failure.
        """
        key = self._make_key(file_content, model_name, prompt_template)
        path = self._cache_path(key)

        entry: dict[str, Any] = {
            "schema_version": 2,
            "key": key,
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "publication_id": uuid4().hex,
            "content_length": len(file_content),
            "prompt_length": len(prompt_template),
            "response": response,
            "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
        }

        publication = json.dumps(entry, indent=2)
        try:
            with self._key_lock(key):
                atomic_write_text(path, publication)
            self.writes += 1
            logger.debug(f"Cache WRITE: {key[:12]}…")
            return publication
        except OSError as exc:
            logger.warning(f"Failed to write cache entry {key[:12]}…: {exc}")
        return None

    def get_verified(
        self, file_content: str, model_name: str, prompt_template: str
    ) -> dict[str, Any] | None:
        """Read a response with its digest-bound native admission evidence."""
        key = self._make_key(file_content, model_name, prompt_template)
        path = self._cache_path(key)
        try:
            with self._key_lock(key):
                try:
                    entry = json.loads(path.read_text()) if path.exists() else None
                    if entry is not None:
                        response, metadata = entry["response"], entry["metadata"]
                        if (
                            entry.get("schema_version") != 3
                            or entry.get("key") != key
                            or entry.get("model") != model_name
                            or not isinstance(response, str)
                            or not response.strip()
                            or not isinstance(metadata, dict)
                            or entry.get("response_sha256")
                            != hashlib.sha256(response.encode()).hexdigest()
                            or entry.get("metadata_sha256")
                            != hashlib.sha256(
                                json.dumps(
                                    metadata,
                                    sort_keys=True,
                                    ensure_ascii=False,
                                    allow_nan=False,
                                ).encode()
                            ).hexdigest()
                        ):
                            raise ValueError("Verified cache evidence differs")
                        self.hits += 1
                        return {"response": response, "metadata": metadata}
                except (OSError, ValueError, KeyError, TypeError):
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        logger.debug(
                            "Invalid verified cache entry could not be removed"
                        )
        except OSError as exc:
            logger.debug("Verified cache entry could not be locked: %s", exc)
        self.misses += 1
        return None

    def put_verified(
        self,
        file_content: str,
        model_name: str,
        prompt_template: str,
        response: str,
        metadata: dict[str, Any],
    ) -> Optional[str]:
        """Store evidence atomically; return its text for conditional rollback."""
        if not isinstance(response, str) or not response.strip():
            raise ValueError("Verified cache response must be nonempty")
        encoded = json.dumps(
            metadata, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        key = self._make_key(file_content, model_name, prompt_template)
        entry = {
            "schema_version": 3,
            "key": key,
            "publication_id": uuid4().hex,
            "model": model_name,
            "response": response,
            "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
            "metadata": metadata,
            "metadata_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        }
        publication = json.dumps(entry, indent=2, allow_nan=False)
        try:
            with self._key_lock(key):
                atomic_write_text(self._cache_path(key), publication)
            self.writes += 1
            return publication
        except OSError as exc:
            logger.warning("Failed to write verified cache entry %s: %s", key[:12], exc)
        return None

    def discard_publication(
        self,
        file_content: str,
        model_name: str,
        prompt_template: str,
        publication: str,
    ) -> None:
        """Remove only this request's unchanged, just-published record."""
        key = self._make_key(file_content, model_name, prompt_template)
        path = self._cache_path(key)
        try:
            with self._key_lock(key):
                if path.read_bytes() == publication.encode("utf-8"):
                    path.unlink()
        except FileNotFoundError:
            pass

    def summary(self) -> dict[str, Any]:
        """Return cache statistics for logging."""
        total = self.hits + self.misses
        ratio = (self.hits / total * 100) if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "hit_ratio_pct": round(ratio, 1),
            "cache_dir": str(self.cache_dir),
            "entries_on_disk": sum(1 for _ in self.cache_dir.glob("*.json")),
        }

    def clear(self) -> int:
        """Remove all cached entries. Returns count of entries removed."""
        count = 0
        for p in self.cache_dir.glob("*.json"):
            with self._key_lock(p.stem):
                try:
                    p.unlink()
                except FileNotFoundError:
                    continue
                count += 1
        logger.info(f"Cache cleared: {count} entries removed")
        return count
