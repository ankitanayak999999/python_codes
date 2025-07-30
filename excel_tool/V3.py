escaped_table = re.escape(table)
unqualified_table = re.escape(table.split('.')[-1])

# Patterns
full_pattern = rf'(?<!\w){escaped_table}(?!\w)'
partial_pattern = rf'(?<!\w)(?:\w+\.)*{unqualified_table}(?!\w)'

# Match collector
tbl_matching_lines = []
tbl_match_types = []
tbl_count = 0

for i, line in enumerate(lines):
    if re.search(full_pattern, line, re.IGNORECASE):
        tbl_matching_lines.append(str(i + 1))
        tbl_match_types.append('full')
        tbl_count += 1
    elif re.search(partial_pattern, line, re.IGNORECASE):
        tbl_matching_lines.append(str(i + 1))
        tbl_match_types.append('partial')
        tbl_count += 1

# Column pattern as before
col_pattern = rf'(?<!\w){re.escape(column)}(?!\w)'
col_matching_lines = [str(i + 1) for i, line in enumerate(lines) if re.search(col_pattern, line, re.IGNORECASE)]
col_count = len(col_matching_lines)
