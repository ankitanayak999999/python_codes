import os
import sys
import re
import json
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Optional libs (used automatically if present)
_HAS_SQLPARSE = False
_HAS_BLACK = False
_HAS_AUTOPEP8 = False
_HAS_LXML = False

try:
    import sqlparse  # type: ignore
    _HAS_SQLPARSE = True
except Exception:
    pass

try:
    import black  # type: ignore
    _HAS_BLACK = True
except Exception:
    pass

try:
    import autopep8  # type: ignore
    _HAS_AUTOPEP8 = True
except Exception:
    pass

try:
    from lxml import etree  # type: ignore
    _HAS_LXML = True
except Exception:
    pass


APP_TITLE = "Converter Tool — SQL / JSON / XML / Python"
DEFAULT_INDENT = 2
MODES = ["SQL", "JSON", "XML", "Python"]
EXT_MAP = {"SQL": ".sql", "JSON": ".json", "XML": ".xml", "Python": ".py"}


# ---------- Formatters (with standard-library fallbacks) ----------

def format_json(text: str, indent: int, sort_keys: bool, ensure_ascii: bool) -> str:
    obj = json.loads(text)
    return json.dumps(obj, indent=indent, sort_keys=sort_keys, ensure_ascii=ensure_ascii)


def _minidom_pretty_xml(text: str, indent: int) -> str:
    # Standard library fallback for XML pretty print
    import xml.dom.minidom as minidom
    try:
        dom = minidom.parseString(text.encode("utf-8"))
    except Exception:
        # Try without encoding when text already a str
        dom = minidom.parseString(text)
    pretty = dom.toprettyxml(indent=" " * indent)
    # Remove blank lines that minidom tends to add
    lines = [ln for ln in pretty.splitlines() if ln.strip()]
    return "\n".join(lines) + "\n"


def format_xml(text: str, indent: int) -> str:
    if _HAS_LXML:
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.fromstring(text.encode("utf-8"), parser)
        # lxml pretty_print respects indentation visually
        return etree.tostring(root, pretty_print=True, encoding="unicode")
    # Fallback to minidom
    return _minidom_pretty_xml(text, indent)


# A very lightweight SQL formatter fallback if sqlparse is not available.
_SQL_MAJOR_BREAKS = [
    "SELECT", "FROM", "WHERE", "JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN",
    "FULL JOIN", "OUTER JOIN", "GROUP BY", "ORDER BY", "HAVING", "LIMIT",
    "UNION", "UNION ALL", "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE", "ON"
]
_SQL_KEYWORDS = set(k for k in _SQL_MAJOR_BREAKS) | {
    "AND", "OR", "CASE", "WHEN", "THEN", "ELSE", "END", "AS", "DISTINCT", "WITH",
    "OVER", "PARTITION", "BY", "IS", "NULL", "NOT", "IN", "EXISTS", "LIKE", "BETWEEN"
}

def _basic_sql_format(text: str, indent: int) -> str:
    # Normalize whitespace
    s = re.sub(r"\s+", " ", text.strip())

    # Insert newlines before major breaks (longer phrases first to avoid partial matches)
    for kw in sorted(_SQL_MAJOR_BREAKS, key=len, reverse=True):
        pattern = r"\s+(?i:" + re.escape(kw) + r")\b"
        s = re.sub(pattern, "\n" + kw, s)

    # Uppercase keywords
    def upcase_keyword(m):
        return m.group(0).upper()
    s = re.sub(
        r"\b(" + "|".join(sorted({k.lower() for k in _SQL_KEYWORDS}, key=len, reverse=True)) + r")\b",
        upcase_keyword,
        s,
        flags=re.IGNORECASE,
    )

    # Split into lines and apply a simple indentation heuristic
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]

    def indent_level(line: str) -> int:
        if line.startswith(("SELECT", "INSERT", "UPDATE", "DELETE", "WITH")):
            return 0
        if line.startswith(("FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING", "LIMIT", "UNION", "UNION ALL")):
            return 1
        if "JOIN" in line:
            return 2
        if line.startswith("ON"):
            return 3
        return 1

    out = []
    for ln in lines:
        out.append((" " * (indent * indent_level(ln))) + ln)
    return "\n".join(out) + "\n"


def format_sql(text: str, indent: int) -> str:
    if _HAS_SQLPARSE:
        # sqlparse gives much better results if available
        return sqlparse.format(
            text,
            reindent=True,
            indent_width=indent,
            keyword_case="upper",
            strip_comments=False,
        ) + ("\n" if not text.endswith("\n") else "")
    return _basic_sql_format(text, indent)


def format_python(text: str) -> str:
    if _HAS_BLACK:
        # Black ignores custom indent but provides consistent results
        return black.format_str(text, mode=black.Mode())
    if _HAS_AUTOPEP8:
        return autopep8.fix_code(text)
    # Last-resort fallback: minimal normalization (keeps comments; not perfect)
    import textwrap
    ded = textwrap.dedent(text).rstrip() + "\n"
    return ded


# ---------- GUI ----------

class ConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1200x700")

        # Top controls frame
        ctrl = ttk.Frame(root, padding=6)
        ctrl.grid(row=0, column=0, sticky="ew")
        ctrl.columnconfigure(99, weight=1)  # spacer stretch

        ttk.Label(ctrl, text="Mode:").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.mode = tk.StringVar(value="JSON")
        self.mode_cb = ttk.Combobox(ctrl, textvariable=self.mode, values=MODES, state="readonly", width=10)
        self.mode_cb.grid(row=0, column=1, sticky="w")

        ttk.Label(ctrl, text="Indent:").grid(row=0, column=2, sticky="w", padx=(12, 6))
        self.indent = tk.IntVar(value=DEFAULT_INDENT)
        self.indent_spin = ttk.Spinbox(ctrl, from_=2, to=8, textvariable=self.indent, width=4)
        self.indent_spin.grid(row=0, column=3, sticky="w")

        self.json_sort = tk.BooleanVar(value=True)
        self.json_ascii = tk.BooleanVar(value=False)
        self.wrap_var = tk.BooleanVar(value=False)

        self.json_sort_cb = ttk.Checkbutton(ctrl, text="JSON: Sort keys", variable=self.json_sort)
        self.json_ascii_cb = ttk.Checkbutton(ctrl, text="JSON: Ensure ASCII", variable=self.json_ascii)
        self.wrap_cb = ttk.Checkbutton(ctrl, text="Word wrap", variable=self.wrap_var, command=self._toggle_wrap)

        # JSON options shown always (harmless in other modes)
        self.json_sort_cb.grid(row=0, column=4, padx=(12, 0))
        self.json_ascii_cb.grid(row=0, column=5, padx=(6, 0))
        self.wrap_cb.grid(row=0, column=6, padx=(12, 0))

        self.open_btn = ttk.Button(ctrl, text="Open File… (Ctrl/Cmd+O)", command=self.open_file)
        self.open_btn.grid(row=0, column=7, padx=(12, 0))
        self.format_btn = ttk.Button(ctrl, text="Format (Ctrl/Cmd+Enter)", command=self.format_now)
        self.format_btn.grid(row=0, column=8, padx=(6, 0))
        self.save_btn = ttk.Button(ctrl, text="Save Output… (Ctrl/Cmd+S)", command=self.save_output)
        self.save_btn.grid(row=0, column=9, padx=(6, 0))
        self.copy_btn = ttk.Button(ctrl, text="Copy Output", command=self.copy_output)
        self.copy_btn.grid(row=0, column=10, padx=(6, 0))
        self.clear_btn = ttk.Button(ctrl, text="Clear", command=self.clear_both)
        self.clear_btn.grid(row=0, column=11, padx=(6, 0))
        self.install_btn = ttk.Button(ctrl, text="Install Helpers", command=self.install_helpers)
        self.install_btn.grid(row=0, column=12, padx=(12, 0))

        # Paned window for side-by-side text areas
        paned = ttk.Panedwindow(root, orient="horizontal")
        paned.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        root.rowconfigure(1, weight=1)
        root.columnconfigure(0, weight=1)

        self.left_frame = ttk.Frame(paned, padding=(0, 0, 6, 0))
        self.right_frame = ttk.Frame(paned, padding=(6, 0, 0, 0))
        paned.add(self.left_frame, weight=1)
        paned.add(self.right_frame, weight=1)

        # Left (Input)
        ttk.Label(self.left_frame, text="INPUT").grid(row=0, column=0, sticky="w")
        self.in_text = tk.Text(self.left_frame, wrap="none", undo=True)
        self._attach_scrollers(self.left_frame, self.in_text, start_row=1)

        # Right (Output)
        ttk.Label(self.right_frame, text="FORMATTED OUTPUT").grid(row=0, column=0, sticky="w")
        self.out_text = tk.Text(self.right_frame, wrap="none", undo=True)
        self._attach_scrollers(self.right_frame, self.out_text, start_row=1)

        # Status bar
        self.status = tk.StringVar(value="Ready")
        status_bar = ttk.Label(root, textvariable=self.status, anchor="w")
        status_bar.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 6))

        # Key bindings
        root.bind_all("<Control-Return>", lambda e: self.format_now())
        root.bind_all("<Command-Return>", lambda e: self.format_now())  # macOS
        root.bind_all("<Control-o>", lambda e: self.open_file())
        root.bind_all("<Command-o>", lambda e: self.open_file())
        root.bind_all("<Control-s>", lambda e: self.save_output())
        root.bind_all("<Command-s>", lambda e: self.save_output())

        # Apply initial wrap setting
        self._toggle_wrap()

    def _attach_scrollers(self, parent: ttk.Frame, text_widget: tk.Text, start_row: int = 0):
        yscroll = ttk.Scrollbar(parent, orient="vertical", command=text_widget.yview)
        xscroll = ttk.Scrollbar(parent, orient="horizontal", command=text_widget.xview)
        text_widget.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        text_widget.grid(row=start_row, column=0, sticky="nsew")
        yscroll.grid(row=start_row, column=1, sticky="ns")
        xscroll.grid(row=start_row + 1, column=0, sticky="ew")

        parent.rowconfigure(start_row, weight=1)
        parent.columnconfigure(0, weight=1)

    def _toggle_wrap(self):
        mode = "word" if self.wrap_var.get() else "none"
        self.in_text.configure(wrap=mode)
        self.out_text.configure(wrap=mode)

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Open file",
            filetypes=[
                ("All files", "*.*"),
                ("SQL", "*.sql"),
                ("JSON", "*.json"),
                ("XML", "*.xml"),
                ("Python", "*.py"),
                ("Text", "*.txt"),
            ],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            messagebox.showerror("Open failed", f"Could not open file:\n{e}")
            self.status.set("Open failed.")
            return

        # Guess mode by extension
        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            self.mode.set("JSON")
        elif ext == ".xml":
            self.mode.set("XML")
        elif ext == ".sql":
            self.mode.set("SQL")
        elif ext == ".py":
            self.mode.set("Python")

        self.in_text.delete("1.0", "end")
        self.in_text.insert("1.0", content)
        self.status.set(f"Loaded: {path}")

    def save_output(self):
        mode = self.mode.get()
        default_ext = EXT_MAP.get(mode, ".txt")
        path = filedialog.asksaveasfilename(
            title="Save output as…",
            defaultextension=default_ext,
            filetypes=[
                (f"{mode}", f"*{default_ext}"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            out = self.out_text.get("1.0", "end-1c")
            with open(path, "w", encoding="utf-8") as f:
                f.write(out)
            self.status.set(f"Saved output to: {path}")
        except Exception as e:
            messagebox.showerror("Save failed", f"Could not save output:\n{e}")
            self.status.set("Save failed.")

    def format_now(self):
        raw = self.in_text.get("1.0", "end-1c")
        if not raw.strip():
            self.status.set("Nothing to format.")
            return
        mode = self.mode.get()
        indent = max(2, min(8, int(self.indent.get() or DEFAULT_INDENT)))
        try:
            if mode == "JSON":
                formatted = format_json(
                    raw, indent=indent, sort_keys=self.json_sort.get(), ensure_ascii=self.json_ascii.get()
                )
            elif mode == "XML":
                formatted = format_xml(raw, indent=indent)
            elif mode == "SQL":
                formatted = format_sql(raw, indent=indent)
            elif mode == "Python":
                formatted = format_python(raw)
            else:
                formatted = raw
            self.out_text.delete("1.0", "end")
            self.out_text.insert("1.0", formatted)
            self.status.set(f"Formatted as {mode}.")
        except Exception as e:
            messagebox.showerror(f"{mode} formatting error", f"Error while formatting:\n\n{e}")
            self.status.set("Formatting error.")

    def copy_output(self):
        text = self.out_text.get("1.0", "end-1c")
        if not text:
            self.status.set("Nothing to copy.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status.set("Output copied to clipboard.")

    def clear_both(self):
        self.in_text.delete("1.0", "end")
        self.out_text.delete("1.0", "end")
        self.status.set("Cleared.")

    def install_helpers(self):
        pkgs = []
        # Offer only missing ones
        if not _HAS_SQLPARSE:
            pkgs.append("sqlparse")
        if not _HAS_BLACK:
            pkgs.append("black")
        if not _HAS_AUTOPEP8:
            pkgs.append("autopep8")
        if not _HAS_LXML:
            pkgs.append("lxml")

        if not pkgs:
            messagebox.showinfo("Install Helpers", "All optional helpers are already available.")
            return

        confirm = messagebox.askyesno(
            "Install Helpers",
            "This will run:\n\n  pip install " + " ".join(pkgs) + "\n\nProceed?",
        )
        if not confirm:
            return

        try:
            cmd = [sys.executable, "-m", "pip", "install"] + pkgs
            proc = subprocess.run(cmd, capture_output=True, text=True)
            output = (proc.stdout or "") + "\n" + (proc.stderr or "")
            # Show result
            top = tk.Toplevel(self.root)
            top.title("Install Helpers — Output")
            top.geometry("800x400")
            txt = tk.Text(top, wrap="word")
            ys = ttk.Scrollbar(top, orient="vertical", command=txt.yview)
            txt.configure(yscrollcommand=ys.set)
            txt.pack(side="left", fill="both", expand=True)
            ys.pack(side="right", fill="y")
            txt.insert("1.0", output)
            txt.mark_set("insert", "1.0")
            txt.focus_set()
            if proc.returncode == 0:
                self.status.set("Helpers installed. Restart app for best results.")
            else:
                self.status.set("Helper install failed (see window).")
        except Exception as e:
            messagebox.showerror("Install failed", f"Could not run pip:\n{e}")
            self.status.set("Helper install failed.")


def main():
    root = tk.Tk()
    # Native-looking widgets on mac/win
    try:
        root.call("tk", "scaling", 1.0)
        style = ttk.Style()
        if sys.platform == "darwin":
            style.theme_use("aqua")
        else:
            # use default or clam
            style.theme_use(style.theme_use())
    except Exception:
        pass
    app = ConverterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
