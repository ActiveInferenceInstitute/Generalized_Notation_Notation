"""Compatibility re-exports for the maintained template library.

The implementation moved below the CLI, HTTP API and MCP adapters to
:mod:`gnn.operations.templates`; import from there in new code. The template
data (``template_index.json`` and ``template_assets/``) still ships in this
package directory, so published ``package://gnn.cli/...`` sources are
unchanged.
"""

from __future__ import annotations

from gnn.operations.templates import PACKAGE_NAME as PACKAGE_NAME
from gnn.operations.templates import REPO_ROOT as REPO_ROOT
from gnn.operations.templates import TEMPLATE_INDEX as TEMPLATE_INDEX
from gnn.operations.templates import (
    TEMPLATE_INDEX_RESOURCE as TEMPLATE_INDEX_RESOURCE,
)
from gnn.operations.templates import TemplateRecord as TemplateRecord
from gnn.operations.templates import _load_template_index as _load_template_index
from gnn.operations.templates import _sha256 as _sha256
from gnn.operations.templates import _sha256_chunks as _sha256_chunks
from gnn.operations.templates import _sha256_resource as _sha256_resource
from gnn.operations.templates import (
    _template_record_from_raw as _template_record_from_raw,
)
from gnn.operations.templates import (
    _validate_template_record as _validate_template_record,
)
from gnn.operations.templates import list_templates as list_templates
from gnn.operations.templates import pull_template as pull_template
from gnn.operations.templates import show_template as show_template

__all__ = [
    "PACKAGE_NAME",
    "REPO_ROOT",
    "TEMPLATE_INDEX",
    "TEMPLATE_INDEX_RESOURCE",
    "TemplateRecord",
    "list_templates",
    "pull_template",
    "show_template",
]
