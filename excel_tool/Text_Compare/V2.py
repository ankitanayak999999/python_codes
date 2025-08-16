from pathlib import Path
import difflib

def compare_text_files(file1: str, file2: str, project_path: str):
    """
    Create a side-by-side HTML diff for two text files and save it under project_path.
    
    Parameters
    ----------
    file1 : str
        Path to the first text file.
    file2 : str
        Path to the second text file.
    project_path : str
        Directory where the HTML report will be written.

    Returns
    -------
    html_path : str
        Full path to the generated HTML diff file.
    unified_diff : str
        Short unified-diff (text) you can log/print if desired.

    Notes
    -----
    - Uses UTF-8 reading with errors='replace' (won't crash on odd bytes).
    - Produces a single HTML file named `diff_<file1>_vs_<file2>.html`.
    """
    f1 = Path(file1)
    f2 = Path(file2)
    out_dir = Path(project_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read files (keep original lines; no normalization so you see real diffs)
    def _read_lines(p: Path):
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            return fh.read().splitlines()

    a_lines = _read_lines(f1)
    b_lines = _read_lines(f2)

    # Build filenames for display and output
    from_desc = f1.name
    to_desc   = f2.name
    safe_from = f1.stem[:50].replace(" ", "_")
    safe_to   = f2.stem[:50].replace(" ", "_")
    html_name = f"diff_{safe_from}_vs_{safe_to}.html"
    html_path = out_dir / html_name

    # HTML side-by-side diff
    html_maker = difflib.HtmlDiff(wrapcolumn=120)
    html = html_maker.make_file(a_lines, b_lines, fromdesc=from_desc, todesc=to_desc, context=False, numlines=3)
    html_path.write_text(html, encoding="utf-8")

    # Small unified diff (text) if you want to log/print somewhere
    udiff = "\n".join(
        difflib.unified_diff(
            a_lines, b_lines,
            fromfile=str(f1),
            tofile=str(f2),
            lineterm=""
        )
    )

    return str(html_path), udiff


# Example (you can delete this block when integrating):
if __name__ == "__main__":
    hp, ud = compare_text_files("file1.txt", "file2.txt", "./diff_reports")
    print("HTML written to:", hp)
    print("Unified diff preview (first 20 lines):")
    print("\n".join(ud.splitlines()[:20]))
