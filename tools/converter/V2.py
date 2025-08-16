import os, sys, re, json, keyword
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

CONFIG_FILE = "colors.json"

# -------------------- Color Config --------------------

DEFAULT_COLORS = {
    "kw": "#0057b7",       # keywords
    "builtin": "#7b1fa2",  # builtins
    "str": "#d35400",      # strings
    "num": "#2e7d32",      # numbers
    "com": "#9e9e9e",      # comments
    "tag": "#1565c0",      # xml tags
    "attr": "#6d4c41",     # xml attr
    "punct": "#455a64",    # punctuation
    "bool": "#00838f",     # booleans/null
    "key": "#ad1457",      # json keys
}

def load_colors():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            return DEFAULT_COLORS.copy()
    return DEFAULT_COLORS.copy()

def save_colors(colors):
    with open(CONFIG_FILE, "w") as f:
        json.dump(colors, f, indent=2)

# -------------------- Highlighter --------------------

class Highlighter:
    def __init__(self, text_widget: tk.Text, colors: dict):
        self.t = text_widget
        self.colors = colors
        self._apply_palette()

    def _apply_palette(self):
        self.t.tag_configure("sel", background="#cfe8ff")
        for tag, fg in self.colors.items():
            self.t.tag_configure(tag, foreground=fg)

    def set_colors(self, colors):
        self.colors = colors
        self._apply_palette()

    def clear(self):
        for tag in self.colors.keys():
            self.t.tag_remove(tag, "1.0", "end")

    def apply(self, tool: str):
        self.clear()
        txt = self.t.get("1.0", "end-1c")
        if not txt: return
        if tool == "SQL": self._sql(txt)
        elif tool == "JSON": self._json(txt)
        elif tool == "XML": self._xml(txt)
        elif tool == "Python": self._python(txt)

    def _tag(self, tag, matches):
        for m in matches:
            self.t.tag_add(tag, self._idx(m.start()), self._idx(m.end()))

    def _idx(self, off):
        line = int(self.t.count("1.0", f"1.0+{off}c", "lines")[0]) + 1
        line_start = int(self.t.count("1.0", f"{line}.0", "chars")[0])
        col = off - line_start
        return f"{line}.{col}"

    def _sql(self, txt):
        self._tag("com", re.finditer(r"--.*?$", txt, flags=re.M))
        self._tag("str", re.finditer(r"'([^']|\\')*'", txt))
        self._tag("num", re.finditer(r"\b\d+(\.\d+)?\b", txt))
        kw = r"\b(" + "|".join(["select","from","where","join","insert","update","delete","with","group","order","by","limit"]) + r")\b"
        self._tag("kw", re.finditer(kw, txt, re.I))

    def _json(self, txt):
        self._tag("key", re.finditer(r'"([^"\\]|\\.)*"\s*(?=:\s)', txt))
        self._tag("str", re.finditer(r'"([^"\\]|\\.)*"', txt))
        self._tag("num", re.finditer(r"-?\b\d+(\.\d+)?([eE][+-]?\d+)?\b", txt))
        self._tag("bool", re.finditer(r"\b(true|false|null)\b", txt))
        self._tag("punct", re.finditer(r"[:,\[\]\{\}]", txt))

    def _xml(self, txt):
        self._tag("com", re.finditer(r"<!--.*?-->", txt, re.S))
        self._tag("tag", re.finditer(r"</?[A-Za-z_][\w\-.]*", txt))
        self._tag("attr", re.finditer(r"\s[A-Za-z_][\w\-.]*\s*=", txt))
        self._tag("str", re.finditer(r"(['\"]).*?\1", txt))

    def _python(self, txt):
        self._tag("com", re.finditer(r"#.*?$", txt, re.M))
        self._tag("str", re.finditer(r"(\"\"\".*?\"\"\"|'''.*?'''|\"([^\"\\]|\\.)*\"|'([^'\\]|\\.)*')", txt, re.S))
        self._tag("num", re.finditer(r"\b\d+(\.\d+)?\b", txt))
        self._tag("kw", re.finditer(r"\b(" + "|".join(keyword.kwlist) + r")\b", txt))
        self._tag("builtin", re.finditer(r"\b(print|len|range|dict|list|set|tuple|int|float|str|bool)\b", txt))

# -------------------- Color Picker Dialog --------------------

class ColorDialog(tk.Toplevel):
    def __init__(self, master, colors, callback):
        super().__init__(master)
        self.title("Customize Colors")
        self.colors = colors.copy()
        self.callback = callback
        self.resizable(False, False)

        row = 0
        for tag, col in self.colors.items():
            ttk.Label(self, text=tag).grid(row=row, column=0, sticky="w", padx=6, pady=3)
            lbl = tk.Label(self, text=col, bg=col, width=10, relief="ridge")
            lbl.grid(row=row, column=1, padx=6, pady=3)
            def pick(t=tag,l=lbl):
                c = colorchooser.askcolor(self.colors[t])[1]
                if c:
                    self.colors[t] = c
                    l.config(bg=c, text=c)
            ttk.Button(self, text="Pick", command=pick).grid(row=row, column=2, padx=6, pady=3)
            row += 1

        ttk.Button(self, text="Save & Apply", command=self._save).grid(row=row, column=0, columnspan=3, pady=8)

    def _save(self):
        save_colors(self.colors)
        self.callback(self.colors)
        self.destroy()

# -------------------- Integration --------------------

# In your ConverterApp toolbar, add:
#   self.colors = load_colors()
#   self.colors_btn = ttk.Button(ctrl, text="Colors…", command=self.edit_colors)
#   self.colors_btn.grid(row=0, column=12, padx=(12,0))
#
# In LinedText, pass Highlighter with self.colors
#   self.hl = Highlighter(self.text, app.colors)
#
# And define in ConverterApp:
#
# def edit_colors(self):
#     ColorDialog(self.root, self.colors, self.apply_colors)
#
# def apply_colors(self, colors):
#     self.colors = colors
#     self.left.hl.set_colors(colors)
#     self.right.hl.set_colors(colors)
