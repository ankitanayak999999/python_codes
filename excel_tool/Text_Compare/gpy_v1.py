import difflib
import sys
import html

def create_diff_report(file1_path, file2_path, output_path, template_path="template.html"):
    """
    Generates a side-by-side HTML diff report by populating a template file.

    Args:
        file1_path (str): Path to the first file.
        file2_path (str): Path to the second file.
        output_path (str): Path to save the final HTML report.
        template_path (str): Path to the HTML template file.
    """
    print("Reading input files...")
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

    print("Processing differences...")
    matcher = difflib.SequenceMatcher(None, file1_lines, file2_lines)
    
    left_pane_content = []
    right_pane_content = []
    left_num, right_num = 1, 1

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for line in file1_lines[i1:i2]:
                escaped_line = html.escape(line)
                left_pane_content.append(f'<div class="line same"><span class="ln">{left_num}</span>{escaped_line}</div>')
                right_pane_content.append(f'<div class="line same"><span class="ln">{right_num}</span>{escaped_line}</div>')
                left_num += 1
                right_num += 1
        else:
            if tag in ('replace', 'delete'):
                for line in file1_lines[i1:i2]:
                    left_pane_content.append(f'<div class="line deleted"><span class="ln">{left_num}</span>{html.escape(line)}</div>')
                    right_pane_content.append('<div class="line placeholder">&nbsp;</div>')
                    left_num += 1
            
            if tag in ('replace', 'insert'):
                for line in file2_lines[j1:j2]:
                    right_pane_content.append(f'<div class="line added"><span class="ln">{right_num}</span>{html.escape(line)}</div>')
                    left_pane_content.append('<div class="line placeholder">&nbsp;</div>')
                    right_num += 1

    print(f"Reading HTML template from '{template_path}'...")
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
    except FileNotFoundError:
        print(f"Error: Template file '{template_path}' not found.")
        print("Please ensure 'template.html' is in the same directory as the script.")
        sys.exit(1)

    # Replace placeholders in the template with actual content
    final_html = template_content.replace('{{FILE1_PATH}}', html.escape(file1_path))
    final_html = final_html.replace('{{FILE2_PATH}}', html.escape(file2_path))
    final_html = final_html.replace('{{LEFT_PANE_CONTENT}}', '\n'.join(left_pane_content))
    final_html = final_html.replace('{{RIGHT_PANE_CONTENT}}', '\n'.join(right_pane_content))

    print(f"Writing final report to '{output_path}'...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
        print("\nSuccessfully generated interactive HTML diff report!")
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
    output_file = r"C:\path\to\save\split_pane_report.html"

    # --- NO MORE EDITS NEEDED BELOW THIS LINE ---

    # The script assumes 'template.html' is in the same directory
    create_diff_report(file1, file2, output_file)
