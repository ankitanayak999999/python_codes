from collections import defaultdict
import re

# Escape and prepare patterns
escaped_table = re.escape(table)
unqualified_table = re.escape(table.split('.')[-1])

full_pattern = rf'(?<!\w){escaped_table}(?!\w)'
partial_pattern = rf'(?<!\w)(?:\w+\.)*{unqualified_table}(?!\w)'

# Group line numbers by match type
tbl_match_groups = defaultdict(list)

for i, line in enumerate(lines):
    line_num = str(i + 1)
    if re.search(full_pattern, line, re.IGNORECASE):
        tbl_match_groups['full'].append(line_num)
    elif re.search(partial_pattern, line, re.IGNORECASE):
        tbl_match_groups['partial'].append(line_num)

# Column pattern (no match-type splitting for now)
col_pattern = rf'(?<!\w){re.escape(column)}(?!\w)'
col_matching_lines = [str(i + 1) for i, line in enumerate(lines)
                      if re.search(col_pattern, line, re.IGNORECASE)]
col_count = len(col_matching_lines)

# Build result — 1 row for each table match type
if col_count > 0:
    for match_type in ['full', 'partial']:
        if tbl_match_groups[match_type]:  # Only if that type has matches
            results.append({
                'SEARCH_KEY_CONCAT': f"{table}.{column}",
                'SEARCH_KEY_1': table,
                'SEARCH_KEY_2': column,
                'VIEW_NAMES': view_name,
                'Table Found (Y/N)': 'Y',
                'Table_Match_Type': match_type,
                'no_of_time_search_key_1_found': len(tbl_match_groups[match_type]),
                'search_key_1_found_line_numbers': ','.join(tbl_match_groups[match_type]),
                'Column Found (Y/N)': 'Y',
                'no_of_time_search_key_2_found': col_count,
                'search_key_2_found_line_numbers': ','.join(col_matching_lines)
            })
