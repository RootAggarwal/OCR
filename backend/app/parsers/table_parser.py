"""
app/parsers/table_parser.py — HTML table parser with colspan/rowspan expansion.

Used by the /export-table and /export-all endpoints to turn raw HTML page
content into a structured 2-D grid before writing to Excel.
"""

from typing import Any, Dict, List

from bs4 import BeautifulSoup


class TableParser:
    """
    Parse all HTML tables in a string, correctly expanding colspan and rowspan
    into a fully rectangular grid.

    Usage::

        parser = TableParser()
        parser.feed(html_string)
        # parser.tables is a list of grids:
        #   each grid is List[List[cell_dict]]
        #   cell_dict keys: text, header, colspan, rowspan, is_top_left

    Only ``is_top_left`` cells should be written to Excel; span-occupying
    cells are present in the grid for coordinate bookkeeping only.
    """

    def __init__(self) -> None:
        self.tables: List[List[List[Dict[str, Any]]]] = []

    def feed(self, content: str) -> None:
        """Parse *content* and populate ``self.tables``."""
        self.tables = []
        soup = BeautifulSoup(content, "html.parser")

        for table in soup.find_all("table"):
            grid: Dict[tuple, Dict[str, Any]] = {}
            current_row = 0

            for tr in table.find_all("tr"):
                current_col = 0

                for cell in tr.find_all(["td", "th"]):
                    # Skip cells already claimed by an earlier rowspan/colspan.
                    while (current_row, current_col) in grid:
                        current_col += 1

                    # Normalize cell text (collapse blank lines).
                    lines = [
                        line.strip()
                        for line in cell.get_text(separator="\n").split("\n")
                    ]
                    text = "\n".join(line for line in lines if line)

                    # Determine header status.
                    is_header = (
                        cell.name == "th"
                        or (
                            cell.get("class") and "header" in cell.get("class")
                        )
                        or cell.parent.name == "thead"
                    )
                    if not is_header and cell.find(["strong", "b"]):
                        is_header = True

                    colspan = int(cell.get("colspan", 1))
                    rowspan = int(cell.get("rowspan", 1))

                    # Fill every spanned (row, col) position in the grid.
                    for r_offset in range(rowspan):
                        for c_offset in range(colspan):
                            grid[(current_row + r_offset, current_col + c_offset)] = {
                                "text": text,
                                "header": is_header,
                                "colspan": colspan,
                                "rowspan": rowspan,
                                "is_top_left": (r_offset == 0 and c_offset == 0),
                            }

                    current_col += colspan

                current_row += 1

            if not grid:
                continue

            max_row = max(r for r, _ in grid) + 1
            max_col = max(c for _, c in grid) + 1

            _EMPTY_CELL: Dict[str, Any] = {
                "text": "",
                "header": False,
                "colspan": 1,
                "rowspan": 1,
                "is_top_left": True,
            }
            result_grid = [
                [grid.get((r, c), _EMPTY_CELL) for c in range(max_col)]
                for r in range(max_row)
            ]
            self.tables.append(result_grid)
