import difflib
import sys
import html

def generate_split_pane_diff(file1_path, file2_path, output_path):
    """
    Generates an interactive, side-by-side, synchronized-scrolling HTML diff report.
    """
    try:
        with open(file1_path, 'r', encoding='utf-8') as f:
            file1_lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{file1_path}' was not found.")
        sys.exit(1)

    try:
        with open(file2_path, 'r', encoding='utf-8') as f:
            file2_lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file '{file2_path}' was not found.")
        sys.exit(1)

    # Use SequenceMatcher to find differences
    matcher = difflib.SequenceMatcher(None, file1_lines, file2_lines)

    # Prepare content for the two panes
    left_pane_content = []
    right_pane_content = []
    
    # Keep track of line numbers
    left_num = 1
    right_num = 1

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for i in range(i1, i2):
                line_content = html.escape(file1_lines[i])
                left_pane_content.append(f'<div class="line same"><span class="ln">{left_num}</span>{line_content}</div>')
                right_pane_content.append(f'<div class="line same"><span class="ln">{right_num}</span>{line_content}</div>')
                left_num += 1
                right_num += 1
        else:
            if tag == 'replace' or tag == 'delete':
                for i in range(i1, i2):
                    line_content = html.escape(file1_lines[i])
                    left_pane_content.append(f'<div class="line deleted"><span class="ln">{left_num}</span>{line_content}</div>')
                    left_num += 1
                # Add placeholders on the right side to keep alignment
                for _ in range(i1, i2):
                     right_pane_content.append('<div class="line placeholder">&nbsp;</div>')

            if tag == 'replace' or tag == 'insert':
                for i in range(j1, j2):
                    line_content = html.escape(file2_lines[i])
                    right_pane_content.append(f'<div class="line added"><span class="ln">{right_num}</span>{line_content}</div>')
                    right_num += 1
                # Add placeholders on the left side to keep alignment
                for _ in range(j1, j2):
                     left_pane_content.append('<div class="line placeholder">&nbsp;</div>')


    # Combine into final HTML structure
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>File Comparison</title>
        <style>
            body {{ font-family: 'Courier New', Courier, monospace; margin: 0; padding: 0; }}
            .container {{ display: flex; height: 100vh; }}
            .pane {{ 
                width: 50%; 
                overflow-y: scroll; 
                border-right: 1px solid #ccc;
                box-sizing: border-box;
                padding: 10px;
                background-color: #fdfdfd;
            }}
            .pane:last-child {{ border-right: none; }}
            .header {{
                padding: 10px;
                background: #eee;
                border-bottom: 1px solid #ccc;
                font-weight: bold;
                text-align: center;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }}
            .line {{ white-space: pre; min-height: 1.2em; }}
            .ln {{ 
                display: inline-block;
                width: 40px;
                color: #888;
                text-align: right;
                margin-right: 10px;
                -webkit-user-select: none; user-select: none;
            }}
            .same {{ background-color: #fff; }}
            .added {{ background-color: #e6ffed; }}
            .deleted {{ background-color: #ffeef0; }}
            .placeholder {{ background-color: #f8f8f8; }}
            .hidden {{ display: none; }}
            .controls {{
                position: fixed; top: 10px; right: 20px;
                background: #fff; border: 1px solid #ccc; border-radius: 5px;
                padding: 8px; z-index: 1000; box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }}
            .controls button {{
                padding: 8px 12px; margin: 0 4px; cursor: pointer;
                border: 1px solid #aaa; border-radius: 3px; background-color: #e7e7e7;
            }}
            .controls button.active {{ background-color: #007bff; color: white; border-color: #007bff; }}
        </style>
    </head>
    <body>
        <div class="controls">
            <button id="showAllBtn" class="active" onclick="showFilter('all')">Show All</button>
            <button id="showDiffBtn" onclick="showFilter('diff')">Show Differences</button>
            <button id="showSameBtn" onclick="showFilter('same')">Show Same</button>
        </div>

        <div class="container">
            <div id="left-pane" class="pane">
                <div class="header" title="{html.escape(file1_path)}">{html.escape(file1_path)}</div>
                {''.join(left_pane_content)}
            </div>
            <div id="right-pane" class="pane">
                <div class="header" title="{html.escape(file2_path)}">{html.escape(file2_path)}</div>
                {''.join(right_pane_content)}
            </div>
        </div>

        <script>
            const leftPane = document.getElementById('left-pane');
            const rightPane = document.getElementById('right-pane');
            let isSyncing = false;

            leftPane.addEventListener('scroll', () => {{
                if (!isSyncing) {{
                    isSyncing = true;
                    rightPane.scrollTop = leftPane.scrollTop;
                }}
                isSyncing = false;
            }});

            rightPane.addEventListener('scroll', () => {{
                if (!isSyncing) {{
                    isSyncing = true;
                    leftPane.scrollTop = rightPane.scrollTop;
                }}
                isSyncing = false;
            }});

            function showFilter(filter) {{
                const allLines = document.querySelectorAll('.line');
                allLines.forEach(line => {{
                    line.classList.remove('hidden'); // Reset first
                    const isSame = line.classList.contains('same');
                    if (filter === 'diff' && isSame) {{
                        line.classList.add('hidden');
                    }} else if (filter === 'same' && !isSame) {{
                        line.classList.add('hidden');
                    }}
                }});

                document.querySelectorAll('.controls button').forEach(b => b.classList.remove('active'));
                document.getElementById(
                    filter === 'all' ? 'showAllBtn' : (filter === 'diff' ? 'showDiffBtn' : 'showSameBtn')
                ).classList.add('active');
            }}
        </script>
    </body>
    </html>
    """
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print("Successfully generated split-pane HTML diff report.")
        print(f"Output saved to: '{output_path}'")
    except IOError as e:
        print(f"Error writing to file '{output_path}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    # --- EDIT THE FILE PATHS BELOW ---

    # Hardcode the path to your first input file
    file1 = r"C:\path\to\your\file_1.sql"

    # Hardcode the path to your second input file
    file2 = r"C:\path\to\your\file_2.sql"

    # Hardcode the path for the output HTML report
    output_file = r"C:\path\to\save\split_pane_comparison.html"

    # --- NO MORE EDITS NEEDED BELOW THIS LINE ---

    generate_split_pane_diff(file1, file2, output_file)
