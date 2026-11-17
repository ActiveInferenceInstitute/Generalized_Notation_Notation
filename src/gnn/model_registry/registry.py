"""
Model Registry

This module provides a centralized registry for GNN models with versioning,
metadata management, and model lifecycle tracking.
"""

import datetime
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ModelVersion:
    """Represents a specific version of a model."""

    def __init__(
        self, version: str, file_path: Path, created_at: Optional[str] = None
    ) -> None:
        """
        Initialize a model version.

        Args:
            version: Version string (e.g., "1.0.0")
            file_path: Path to the model file
            created_at: Creation timestamp (ISO format)
        """
        self.version = version
        self.file_path = file_path
        self.created_at = created_at or datetime.datetime.now().isoformat()
        self.metadata: Dict[str, Any] = {}
        self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute a hash of the model file content."""
        try:
            with open(self.file_path, "rb") as f:
                content = f.read()
            return hashlib.sha256(content).hexdigest()
        except (OSError, TypeError, ValueError) as exc:
            logger.debug("Failed to hash model file %s: %s", self.file_path, exc)
            return ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "file_path": str(self.file_path),
            "created_at": self.created_at,
            "metadata": self.metadata,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create a ModelVersion from a dictionary."""
        version = cls(
            version=data["version"],
            file_path=Path(data["file_path"]),
            created_at=data["created_at"],
        )
        version.metadata = data.get("metadata", {})
        version.hash = data.get("hash", "")
        return version


class ModelEntry:
    """Represents a model entry in the registry."""

    def __init__(self, model_id: str, name: Optional[str] = None) -> None:
        """
        Initialize a model entry.

        Args:
            model_id: Unique identifier for the model
            name: Human-readable name for the model
        """
        self.model_id = model_id
        self.name = name or model_id
        self.description = ""
        self.created_at = datetime.datetime.now().isoformat()
        self.updated_at = self.created_at
        self.versions: Dict[str, ModelVersion] = {}
        self.tags: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.current_version: Optional[str] = None

    def add_version(self, version: ModelVersion) -> None:
        """
        Add a new version to the model.

        Args:
            version: ModelVersion object
        """
        self.versions[version.version] = version
        self.current_version = version.version
        self.updated_at = datetime.datetime.now().isoformat()

    def get_version(self, version: Optional[str] = None) -> Optional[ModelVersion]:
        """
        Get a specific version of the model.

        Args:
            version: Version string (if None, returns current version)

        Returns:
            ModelVersion object if found, None otherwise
        """
        if version is None:
            if self.current_version:
                return self.versions.get(self.current_version)
            return None
        return self.versions.get(version)

    def add_tag(self, tag: str) -> None:
        """
        Add a tag to the model.

        Args:
            tag: Tag string
        """
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.datetime.now().isoformat()

    def remove_tag(self, tag: str) -> None:
        """
        Remove a tag from the model.

        Args:
            tag: Tag string
        """
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.datetime.now().isoformat()

    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Update model metadata.

        Args:
            metadata: Dictionary of metadata
        """
        self.metadata.update(metadata)
        self.updated_at = datetime.datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "versions": {v: ver.to_dict() for v, ver in self.versions.items()},
            "tags": self.tags,
            "metadata": self.metadata,
            "current_version": self.current_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelEntry":
        """Create a ModelEntry from a dictionary."""
        entry = cls(model_id=data["model_id"], name=data["name"])
        entry.description = data.get("description", "")
        entry.created_at = data.get("created_at", entry.created_at)
        entry.updated_at = data.get("updated_at", entry.updated_at)
        entry.tags = data.get("tags", [])
        entry.metadata = data.get("metadata", {})
        entry.current_version = data.get("current_version")

        # Load versions
        for v, ver_data in data.get("versions", {}).items():
            entry.versions[v] = ModelVersion.from_dict(ver_data)

        return entry


class ModelRegistry:
    """Centralized registry for GNN models."""

    def __init__(self, registry_path: Union[str, Path]) -> None:
        """
        Initialize the model registry.

        Args:
            registry_path: Path to the registry file
        """
        self.registry_path = Path(registry_path)
        self.models: Dict[str, ModelEntry] = {}
        self.load()

    def load(self) -> None:
        """Load the registry from file."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    data = json.load(f)

                for model_id, model_data in data.get("models", {}).items():
                    self.models[model_id] = ModelEntry.from_dict(model_data)
            except Exception as e:
                print(f"Error loading registry: {e}")

    def save(self) -> None:
        """Save the registry to file."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "version": "1.0.0",
            "updated_at": datetime.datetime.now().isoformat(),
            "models": {
                model_id: model.to_dict() for model_id, model in self.models.items()
            },
        }

        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def register_model(self, model_path: Path) -> bool:
        """
        Register a model in the registry.

        Args:
            model_path: Path to the model file

        Returns:
            True if registration succeeded, False otherwise
        """
        try:
            # Extract model metadata
            with open(model_path, encoding="utf-8") as f:
                content = f.read()

            # Extract model ID and name
            model_id = model_path.stem
            model_name = self._extract_model_name(content) or model_id

            # Create or update model entry
            if model_id in self.models:
                model = self.models[model_id]
                model.name = model_name  # Update name
            else:
                model = ModelEntry(model_id, model_name)
                self.models[model_id] = model

            # Extract version
            version_str = self._extract_version(content) or "1.0.0"

            # Create version
            model_version = ModelVersion(version_str, model_path)

            # Extract additional metadata
            metadata = self._extract_metadata(content)
            model_version.metadata = metadata

            # Also store metadata at the model level for direct access
            # Merge with existing model metadata (version-specific metadata takes precedence for updates)
            for key, value in metadata.items():
                if key not in model.metadata or model.metadata[key] != value:
                    model.metadata[key] = value

            # Add version to model
            model.add_version(model_version)

            # Extract tags
            tags = self._extract_tags(content)
            for tag in tags:
                model.add_tag(tag)

            # Update model description
            description = self._extract_description(content)
            if description:
                model.description = description

            return True

        except Exception as e:
            print(f"Error registering model {model_path}: {e}")
            return False

    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """
        Get a model by ID.

        Args:
            model_id: Model ID

        Returns:
            ModelEntry if found, None otherwise
        """
        return self.models.get(model_id)

    def search_models(self, query: str) -> List[ModelEntry]:
        """
        Search models by name, description, or tags.

        Args:
            query: Search query

        Returns:
            List of matching ModelEntry objects
        """
        query = query.lower()
        results: list[Any] = []

        for model in self.models.values():
            if (
                query in model.name.lower()
                or query in model.description.lower()
                or any(query in tag.lower() for tag in model.tags)
            ):
                results.append(model)

        return results

    def list_models(self) -> List[ModelEntry]:
        """
        List all models in the registry.

        Returns:
            List of all ModelEntry objects
        """
        return list(self.models.values())

    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the registry.

        Args:
            model_id: Model ID

        Returns:
            True if deletion succeeded, False otherwise
        """
        if model_id in self.models:
            del self.models[model_id]
            return True
        return False

    def _extract_model_name(self, content: str) -> Optional[str]:
        """Extract model name from content."""
        # Try to find ModelName: <name>
        model_name_match = re.search(r"ModelName:\s*([^\n]+)", content)
        if model_name_match:
            return model_name_match.group(1).strip()

        # Try to find # <name>
        title_match = re.search(r"^#\s+([^\n]+)", content)
        if title_match:
            return title_match.group(1).strip()

        return None

    def _extract_version(self, content: str) -> Optional[str]:
        """Extract version from content."""
        version_match = re.search(r"Version:\s*([^\n]+)", content)
        if version_match:
            return version_match.group(1).strip()
        return None

    def _extract_tags(self, content: str) -> List[str]:
        """Extract tags from content."""
        tags_match = re.search(r"Tags:\s*([^\n]+)", content)
        if tags_match:
            tags_str = tags_match.group(1).strip()
            return [tag.strip() for tag in tags_str.split(",")]
        return []

    def _extract_description(self, content: str) -> Optional[str]:
        """Extract description from content."""
        # Try to find Description: <description>
        desc_match = re.search(r"Description:\s*([^\n]+)", content)
        if desc_match:
            return desc_match.group(1).strip()

        # Try to find the first paragraph after the title
        lines = content.split("\n")
        in_paragraph = False
        paragraph_lines: list[Any] = []

        for line in lines:
            if line.startswith("#"):
                continue

            if not line.strip():
                if in_paragraph:
                    break
                continue

            in_paragraph = True
            paragraph_lines.append(line.strip())

        if paragraph_lines:
            return " ".join(paragraph_lines)

        return None

    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from content."""
        metadata: dict[Any, Any] = {}

        # Extract author
        author_match = re.search(r"Author:\s*([^\n]+)", content)
        if author_match:
            metadata["author"] = author_match.group(1).strip()

        # Extract date
        date_match = re.search(r"Date:\s*([^\n]+)", content)
        if date_match:
            metadata["date"] = date_match.group(1).strip()

        # Extract license
        license_match = re.search(r"License:\s*([^\n]+)", content)
        if license_match:
            metadata["license"] = license_match.group(1).strip()

        return metadata


def process_model_registry(
    target_dir: Path, output_dir: Path, **kwargs: Any
) -> Dict[str, Any]:
    """
    Process model registry for GNN files in the target directory.

    Args:
        target_dir: Directory containing GNN files
        output_dir: Directory to save registry results
        **kwargs: Additional arguments

    Returns:
        Dictionary with processing results
    """
    registry_path_arg = kwargs.get("registry_path")
    registry_path = (
        Path(registry_path_arg)
        if registry_path_arg is not None
        else output_dir / "model_registry.json"
    )
    registry = ModelRegistry(registry_path)

    processed_files = 0
    successful_registrations = 0

    # Find all GNN files
    gnn_extensions: list[Any] = [".md", ".gnn", ".json", ".yaml", ".yml"]
    gnn_files: list[Any] = []

    for ext in gnn_extensions:
        gnn_files.extend(target_dir.glob(f"**/*{ext}"))

    for gnn_file in gnn_files:
        processed_files += 1
        if registry.register_model(gnn_file):
            successful_registrations += 1

    # Filter models by ontology term if requested
    query_ontology = kwargs.get("query_ontology")
    matching_models = list(registry.models.keys())
    if query_ontology and isinstance(query_ontology, str):
        query_term = query_ontology.lower()
        filtered: dict[str, Any] = {}
        for mid, mdata in registry.models.items():
            content_str = str(mdata).lower()
            if query_term in content_str:
                filtered[mid] = mdata
        registry.models = filtered
        matching_models = list(filtered.keys())

    # Save registry
    registry.save()

    # Create summary
    results: dict[str, Any] = {
        "success": True,
        "processed_files": processed_files,
        "successful_registrations": successful_registrations,
        "registry_path": str(registry_path),
        "total_models": len(registry.models),
        "matching_models": matching_models,
    }

    return results
