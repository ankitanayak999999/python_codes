import re

pattern = rf'(?<!_)({re.escape(table)})(?!_)'
tbl_matching_lines = [
    str(i + 1) for i, line in enumerate(lines)
    if re.search(pattern, col_pattern = rf'(?<!_)({re.escape(column)})(?!_)'
col_matching_lines = [
    str(i + 1) for i, line in enumerate(lines)
    if re.search(col_pattern, line.upper())
]


