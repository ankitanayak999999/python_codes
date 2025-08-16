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



pyinstaller your_spec_or_cmd --add-data "text_compare_template.html;."

pyinstaller your_spec_or_cmd --add-data "text_compare_template.html;."


from pathlib import Path
import difflib
import datetime

def load_html_template(template_path: str, table_html: str) -> str:
    """
    Load the external HTML template and inject the {{DIFF_TABLE}} placeholder.
    Other placeholders are replaced in compare_text_files.
    """
    tpl = Path(template_path).read_text(encoding="utf-8")
    return tpl.replace("{{DIFF_TABLE}}", table_html)

def compare_text_files(file1: str, file2: str, project_path: str, template_file: str):
    """
    file1, file2: paths to input text/SQL files
    project_path: folder where the HTML report should be written
    template_file: path to your external 'text_compare_template.html'
    Returns: (html_path_str, unified_diff_text)
    """
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")

    f1 = Path(file1)
    f2 = Path(file2)
    out_dir = Path(project_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- read files exactly as-is ---
    def _read_lines(p: Path):
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            return fh.read().splitlines()

    a_lines = _read_lines(f1)
    b_lines = _read_lines(f2)

    # --- build only the diff TABLE (not full page) ---
    html_maker = difflib.HtmlDiff(wrapcolumn=120)
    table_html = html_maker.make_table(
        a_lines, b_lines,
        fromdesc=f1.name,
        todesc=f2.name,
        context=False, numlines=0
    )

    # --- load template and inject table ---
    final_html = load_html_template(template_file, table_html)

    # --- replace remaining placeholders ---
    final_html = (final_html
                  .replace("{{TITLE}}", "Text Compare Result")
                  .replace("{{LEFT_LABEL}}", f1.name)
                  .replace("{{RIGHT_LABEL}}", f2.name))

    # --- write report ---
    html_name = f"text_compare_result_{timestamp}.html"
    html_path = out_dir / html_name
    html_path.write_text(final_html, encoding="utf-8")

    # optional: plain unified diff text
    udiff = "\n".join(difflib.unified_diff(
        a_lines, b_lines, fromfile=str(f1), tofile=str(f2), lineterm=""
    ))

    return str(html_path), udiff

