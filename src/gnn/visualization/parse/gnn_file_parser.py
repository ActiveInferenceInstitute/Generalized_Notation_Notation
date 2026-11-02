"""
GNN Parser Module

This module provides functionality to parse GNN files and convert them into a structured format
for visualization and analysis.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict

from gnn.parsers.common import canonical_section_heading

logger = logging.getLogger(__name__)


class GNNParser:
    """Parser for GNN files in various formats."""

    def __init__(self) -> None:
        """Initialize the GNN parser."""
        self.sections: dict[str, Any] = {}

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a GNN file and return the structured content.

        Args:
            file_path: Path to the GNN file to parse

        Returns:
            Dictionary containing parsed GNN sections
        """
        path = Path(file_path)
        content = path.read_text()

        # Determine file format based on file content
        if "GNNSection," in content:
            return self._parse_csv_format(file_path, content)
        else:
            return self._parse_markdown_format(content)

    def _parse_csv_format(self, file_path: str, content: str) -> Dict[str, Any]:
        """Parse GNN file in CSV format."""
        sections: dict[Any, Any] = {}

        # Extract header comments
        header_lines: list[Any] = []
        content_lines = content.splitlines()
        line_index = 0

        while line_index < len(content_lines) and (
            content_lines[line_index].startswith("#")
            or not content_lines[line_index].strip()
        ):
            if content_lines[line_index].startswith("#"):
                header_lines.append(content_lines[line_index])
            line_index += 1

        # Add header information to sections
        if header_lines:
            sections["ModelName"] = header_lines[0].replace("#", "").strip()
            if len(header_lines) > 1:
                sections["ModelAnnotation"] = "\n".join(
                    [h.replace("#", "").strip() for h in header_lines[1:]]
                )

        # Re-parse the raw file content to extract all sections
        self._extract_gnn_csv_sections(content, sections)

        # Process state space and connections
        logger.debug(
            f"Before processing, StateSpaceBlock length: {len(sections.get('StateSpaceBlock', ''))}"
        )
        self._process_state_space(sections)
        self._process_connections(sections)

        return sections

    def _extract_gnn_csv_sections(self, content: str, sections: Dict[str, Any]) -> None:
        """Extract sections from GNN CSV format, handling quoted multiline values."""
        # Extract each section with its quoted content
        section_pattern = r'(\w+),(?:"([^"]*)"|(.*?)(?=\n\w+,|\Z))'

        for match in re.finditer(section_pattern, content, re.DOTALL):
            section_name = match.group(1)
            # Get the content from either the quoted group or unquoted group
            section_content = (
                match.group(2) if match.group(2) is not None else match.group(3)
            )

            if section_content is not None:
                # Clean up the section content
                if section_name in [
                    "StateSpaceBlock",
                    "Connections",
                    "InitialParameterization",
                    "Equations",
                    "Time",
                    "ActInfOntologyAnnotation",
                ]:
                    # Remove leading markdown header if present
                    if section_content.startswith("## "):
                        section_content = (
                            section_content.split("\n", 1)[1]
                            if "\n" in section_content
                            else ""
                        )

                sections[section_name] = section_content.strip()

    def _parse_markdown_format(self, content: str) -> Dict[str, Any]:
        """Parse GNN file in Markdown format."""
        sections: Dict[str, Any] = {}

        # Regex to find all section headers (## SectionName). The whole heading
        # text is captured so a spaced spelling still names its canonical
        # section instead of truncating at the first space.
        section_headers = list(
            re.finditer(r"^##[ \t]+(.+?)[ \t]*$", content, re.MULTILINE)
        )

        # Preserve header comments (lines before the first section header)
        if section_headers:
            first_header_start = section_headers[0].start()
            header_comments_content = content[:first_header_start].strip()
            if header_comments_content:
                sections["_HeaderComments"] = header_comments_content
        else:
            # If no sections, the whole file might be header comments or unstructured
            header_comments_content = content.strip()
            if header_comments_content:
                sections["_HeaderComments"] = header_comments_content

        for i, header_match in enumerate(section_headers):
            section_name = canonical_section_heading(header_match.group(1).strip())
            start_pos = header_match.end()

            # The end of the section is the start of the next section, or the end of the file
            end_pos = (
                section_headers[i + 1].start()
                if i + 1 < len(section_headers)
                else len(content)
            )

            section_content = content[start_pos:end_pos].strip()

            sections[section_name] = section_content

        # Process specific GNN sections for structured data
        self._process_state_space(sections)
        self._process_connections(sections)

        # Attempt to populate ModelName from header if not found as a section
        if "ModelName" not in sections and "_HeaderComments" in sections:
            header_lines = sections["_HeaderComments"].split("\n")
            if header_lines and header_lines[0].startswith("# GNN Example: "):
                sections["ModelName"] = (
                    header_lines[0].replace("# GNN Example: ", "").strip()
                )
            elif header_lines and header_lines[0].startswith("# "):
                sections["ModelName"] = header_lines[0][2:].strip()

        return sections

    def _process_state_space(self, sections: Dict[str, Any]) -> None:
        """Process the StateSpaceBlock to extract variables and their dimensions."""
        if "StateSpaceBlock" not in sections:
            logger.debug("StateSpaceBlock section not found")
            return

        state_space_content = sections["StateSpaceBlock"]
        variables: dict[Any, Any] = {}

        logger.debug("Processing state space block...")

        lines = state_space_content.split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith(
                "##"
            ):  # Ignore empty lines or sub-headers within content
                continue

            # First, check if there's a comment in the line
            comment = ""
            hash_index = line.find("#")
            if hash_index != -1:
                comment = line[hash_index + 1 :].strip()
                line = line[
                    :hash_index
                ].strip()  # Remove the comment part for cleaner regex matching

            # Match the variable definition
            match = re.match(r"(\w+(?:\^\w+)?(?:_\w+)?)\s*\[([^\]]+)\]", line)
            if not match:
                match = re.match(r"(\w+(?:\^\w+)?(?:_\w+)?)\s*([\w,=\[\]]+)", line)

            if match:
                var_name = match.group(1)
                dimensions_str = match.group(2)

                logger.debug(
                    f"Found variable: {var_name}, dimensions: {dimensions_str}, comment: {comment}"
                )

                dimensions: list[Any] = []
                var_type = None

                # Handle simple dimension strings like "len(π)" or "[2]"
                if re.fullmatch(r"len\(\w+\)", dimensions_str) or re.fullmatch(
                    r"\[\d+\]", dimensions_str
                ):
                    dimensions.append(dimensions_str)  # Keep as string for such cases
                else:
                    for dim_part in dimensions_str.split(","):
                        dim_part = dim_part.strip()
                        if "type=" in dim_part:
                            var_type = dim_part.split("=")[1]
                        else:
                            try:
                                dimensions.append(int(dim_part))
                            except ValueError:
                                dimensions.append(dim_part)  # Keep as string if not int

                # Ensure var_type is never None for safety
                if var_type is None:
                    var_type = "unknown"

                variables[var_name] = {
                    "dimensions": dimensions,
                    "type": var_type,
                    "comment": comment if comment else "",
                }

        if variables:
            logger.debug(
                f"Extracted {len(variables)} variables with comments: {[(k, v.get('comment', '')) for k, v in variables.items()]}"
            )
            sections["Variables"] = variables
        else:
            logger.debug("No variables could be extracted from state space")

    def _process_connections(self, sections: Dict[str, Any]) -> None:
        """Process the Connections section to extract graph structure."""
        if "Connections" not in sections:
            return

        connections_content = sections["Connections"]
        edges: list[Any] = []

        pattern = r"(\w+(?:\^\w+)?(?:_\w+)?(?:\+\d+)?)\s*([>\-])\s*(\w+(?:\^\w+)?(?:_\w+)?(?:\+\d+)?)\s*(?:=\s*([^#]*))?(?:#(.*))?"

        lines = connections_content.split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("##"):  # Ignore empty lines or sub-headers
                continue

            match = re.match(pattern, line)
            if match:
                source = match.group(1)
                edge_type = match.group(2)
                target = match.group(3)
                constraint = match.group(4).strip() if match.group(4) else None
                comment = match.group(5).strip() if match.group(5) else None

                logger.debug(f"Found edge: {source} {edge_type} {target}")

                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "directed": edge_type == ">",
                        "constraint": constraint,
                        "comment": comment,
                    }
                )

        if edges:
            sections["Edges"] = edges

    def extract_sections(self, file_path: str) -> Dict[str, str]:
        """
        Extract all sections from a GNN file without detailed parsing.

        Args:
            file_path: Path to the GNN file

        Returns:
            Dictionary with section names as keys and their raw content as values
        """
        return self.parse_file(file_path)
