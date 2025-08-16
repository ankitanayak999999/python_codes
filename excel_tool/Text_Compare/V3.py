# text_compare_helpers.py
from pathlib import Path
import datetime
import difflib

# --- small utils -------------------------------------------------------------

def _read_lines(p: Path):
    with p.open(encoding="utf-8", errors="replace") as fh:
        return fh.read().splitlines()

def _load_template(template_file: Path) -> str:
    with template_file.open("r", encoding="utf-8") as fh:
        return fh.read()

def _inject_template(html_tpl: str, *, title: str, left_label: str, right_label: str,
                     left_table_html: str, right_table_html: str) -> str:
    # Simple literal replacements (no templating engine required)
    return (html_tpl
            .replace("{{TITLE}}", title)
            .replace("{{LEFT_LABEL}}", left_label)
            .replace("{{RIGHT_LABEL}}", right_label)
            .replace("{{LEFT_TABLE}}", left_table_html)
            .replace("{{RIGHT_TABLE}}", right_table_html)
            )

# --- main API ----------------------------------------------------------------

def compare_text_files(
    file1: str,
    file2: str,
    project_path: str,
    template_file: str | None = None,
) -> str:
    """
    Build a side-by-side diff HTML with top toolbar (All / Differences / Same).

    Parameters
    ----------
    file1, file2 : paths to the two text files
    project_path : directory where the HTML will be written
    template_file : optional path to text_compare_template.html.
                    If None, we look for a file named like that next to THIS .py.

    Returns
    -------
    str : absolute path to the generated HTML file
    """
    f1 = Path(file1)
    f2 = Path(file2)
    out_dir = Path(project_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load files
    a_lines = _read_lines(f1)
    b_lines = _read_lines(f2)

    # Make one difflib table (it contains both left+right columns).
    # The HTML template hides the opposite side in each panel via CSS,
    # so we can reuse the same table HTML for both panels.
    hd = difflib.HtmlDiff(wrapcolumn=120)
    table_html = hd.make_table(
        a_lines, b_lines,
        fromdesc=str(f1),
        todesc=str(f2),
        context=False, numlines=0,
    )

    # Load template
    if template_file is None:
        template_file = Path(__file__).with_name("text_compare_template.html")
    else:
        template_file = Path(template_file)

    html_tpl = _load_template(template_file)

    # Fill placeholders
    title = "Text Compare Result"
    final_html = _inject_template(
        html_tpl,
        title=title,
        left_label=f1.name,
        right_label=f2.name,
        left_table_html=table_html,
        right_table_html=table_html,
    )

    # Write output
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    html_name = f"text_compare_result_{ts}.html"
    html_path = out_dir / html_name
    html_path.write_text(final_html, encoding="utf-8")

    return str(html_path)


def unified_diff_preview(file1: str, file2: str, max_lines: int = 40) -> str:
    """
    Optional: return a small unified-diff text preview (for logging/CLI).
    """
    f1, f2 = Path(file1), Path(file2)
    a_lines = _read_lines(f1)
    b_lines = _read_lines(f2)
    ud = difflib.unified_diff(a_lines, b_lines, fromfile=str(f1), tofile=str(f2), lineterm="")
    lines = list(ud)
    return "\n".join(lines[:max_lines])


# example_use.py
from pathlib import Path
from text_compare_helpers import compare_text_files, unified_diff_preview

def main():
    project_path = r"C:\Users\raksahu\Downloads\python\input"
    file1 = r"C:\Users\raksahu\Downloads\python\input\file_1.sql"
    file2 = r"C:\Users\raksahu\Downloads\python\input\file_2.sql"

    # If your HTML template is in the same folder as this .py, you can omit template_file
    html_out = compare_text_files(file1, file2, project_path)
    print("HTML written to:", html_out)

    # Optional: quick CLI preview
    print("\n--- unified diff (first lines) ---")
    print(unified_diff_preview(file1, file2, max_lines=20))

if __name__ == "__main__":
    main()
