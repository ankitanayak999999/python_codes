from pathlib import Path
import difflib
import datetime

def _read_lines(p: Path):
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        return fh.read().splitlines()

def _load_template(template_path: str) -> str:
    return Path(template_path).read_text(encoding="utf-8")

def _fill_template(tpl: str, *, table_html: str, left_label: str, right_label: str) -> str:
    # We inject the SAME full table into both panels; CSS in the template
    # will hide left/right columns in each panel.
    return (
        tpl.replace("{{LEFT_LABEL}}", left_label)
           .replace("{{RIGHT_LABEL}}", right_label)
           .replace("{{FILE1_CONTENT}}", table_html)
           .replace("{{FILE2_CONTENT}}", table_html)
    )

def compare_text_files(file1: str, file2: str, project_path: str, template_file: str) -> tuple[str, str]:
    """
    Build a side-by-side HTML diff with a top toolbar and two independent
    scrollable panels (left/right). Returns (html_output_path, unified_diff_text).
    """
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    f1 = Path(file1)
    f2 = Path(file2)
    out_dir = Path(project_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read
    a_lines = _read_lines(f1)
    b_lines = _read_lines(f2)

    # Labels
    left_label  = f1.name
    right_label = f2.name

    # One HtmlDiff table (we'll place it twice; CSS will hide columns)
    html_maker = difflib.HtmlDiff(wrapcolumn=120)
    table_html = html_maker.make_table(
        a_lines, b_lines,
        fromdesc=left_label, todesc=right_label,
        context=False, numlines=0
    )

    # Load + fill template
    tpl = _load_template(template_file)
    final_html = _fill_template(
        tpl,
        table_html=table_html,
        left_label=left_label,
        right_label=right_label
    )

    # Write HTML
    html_name = f"text_compare_result_{now}.html"
    html_path = out_dir / html_name
    html_path.write_text(final_html, encoding="utf-8")

    # Optional unified diff text (useful for logs)
    udiff = "\n".join(
        difflib.unified_diff(a_lines, b_lines, fromfile=str(f1), tofile=str(f2), lineterm="")
    )
    return str(html_path), udiff
