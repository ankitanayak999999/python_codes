from pathlib import Path
from difflib import SequenceMatcher
import datetime

# ---------- small helpers ----------
def read_lines(p: Path):
    with open(p, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read().splitlines()

def make_line_panels(left_lines, right_lines, left_label, right_label):
    sm = SequenceMatcher(None, left_lines, right_lines)
    left_html, right_html = [], []

    left_html.append(f"<div><b>{left_label}</b></div>\n")
    right_html.append(f"<div><b>{right_label}</b></div>\n")

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        for idx in range(max(i2 - i1, j2 - j1)):
            l_text = left_lines[i1 + idx] if i1 + idx < i2 else ""
            r_text = right_lines[j1 + idx] if j1 + idx < j2 else ""

            if tag == "equal":
                cls = "same"
            elif tag == "insert":
                cls = "add"; l_text = ""
            elif tag == "delete":
                cls = "del"; r_text = ""
            else:
                cls = "chg"

            left_html.append(f"<div class='{cls}'><span class='line-num'>{i1+idx+1 if l_text else ''}</span>{l_text}</div>")
            right_html.append(f"<div class='{cls}'><span class='line-num'>{j1+idx+1 if r_text else ''}</span>{r_text}</div>")

    return "\n".join(left_html), "\n".join(right_html)

# ---------- main function ----------
def compare_text_files(file1, file2, output_dir, template_file):
    left_lines = read_lines(Path(file1))
    right_lines = read_lines(Path(file2))

    left_html, right_html = make_line_panels(left_lines, right_lines,
                                             f"File 1 — {Path(file1).name}",
                                             f"File 2 — {Path(file2).name}")

    with open(template_file, "r", encoding="utf-8") as tf:
        template = tf.read()

    out_html = template.replace("{{LEFT}}", left_html).replace("{{RIGHT}}", right_html)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"text_compare_result_{ts}.html"

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(out_html)

    print("✅ HTML written to:", out_path)
    return str(out_path)

# ---------- CLI entry ----------
def main():
    file1 = r"/Users/reyansh/Documents/VS_CODE/files/input/file_1.sql"
    file2 = r"/Users/reyansh/Documents/VS_CODE/files/input/file_2.sql"
    output_dir = r"/Users/reyansh/Documents/VS_CODE/files/input"
    template_file = r"/Users/reyansh/Documents/VS_CODE/PYTHON_PROJECT/Tools /Text_Compare/text_compare_template.html"

    compare_text_files(file1, file2, output_dir, template_file)

if __name__ == "__main__":
    main()
