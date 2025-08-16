def load_html_template(template_file: str, diff_table: str) -> str:
    with open(template_file, "r", encoding="utf-8") as f:
        tpl = f.read()
    return tpl.replace("{{ diff_table }}", diff_table)

def compare_text_files(file1: str, file2: str, project_path: str, template_file: str = None):

html_maker = difflib.HtmlDiff(wrapcolumn=120)

# Build only the diff TABLE (not a full HTML page)
table_html = html_maker.make_table(
    a_lines, b_lines,
    fromdesc=from_desc, todesc=to_desc,
    context=False,   # show all lines; your template adds filters
    numlines=0
)

# Default template path if none provided
if template_file is None:
    from pathlib import Path
    template_file = str(Path(__file__).with_name("text_compare_template.html"))

# Load template and inject the table
final_html = load_html_template(template_file, table_html)

# Write the final HTML
html_path.write_text(final_html, encoding="utf-8")


hp, ud = compare_text_files(file_1, file_2, project_path)
# or
hp, ud = compare_text_files(file_1, file_2, project_path, template_file=r"C:\path\to\text_compare_template.html")

pyinstaller your_spec_or_cmd --add-data "text_compare_template.html;."
