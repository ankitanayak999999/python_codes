#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import difflib
import datetime
import platform
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "Text Compare (GUI) – v2"

# -------------------------- Composite widget --------------------------
class TextPane(ttk.Frame):
    """
    A composite widget: line-number gutter + Text with both scrollbars.
    API: set_text(), get_text(), clear(), yview_moveto(), yview(), xview()
    """
    def __init__(self, master, title=""):
        super().__init__(master)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(1, weight=1)

        # Header (title + copy)
        head = ttk.Frame(self)
        head.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 2))
        ttk.Label(head, text=title, font=("TkDefaultFont", 10, "bold")).pack(side="left")
        ttk.Button(head, text="Copy", width=6, command=self.copy_all).pack(side="left", padx=6)

        # Line-number gutter (disabled Text)
        self.gutter = tk.Text(self, width=6, padx=4, state="disabled", wrap="none",
                              relief="flat", background="#f0f0f0")
        self._set_mono_font(self.gutter)
        self.gutter.grid(row=1, column=0, sticky="nsw", padx=(0, 4))

        # Vertical scrollbar
        self.vsb = ttk.Scrollbar(self, orient="vertical")
        self.vsb.grid(row=1, column=1, sticky="ns")

        # Main Text + horizontal scrollbar
        self.text = tk.Text(self, wrap="none", undo=True)
        self._set_mono_font(self.text)
        self.text.grid(row=1, column=2, sticky="nsew")
        self.hsb = ttk.Scrollbar(self, orient="horizontal", command=self.text.xview)
        self.hsb.grid(row=2, column=2, sticky="ew")
        self.text.configure(xscrollcommand=self.hsb.set)

        # connect vertical scrolling
        self.vsb.config(command=self.text.yview)
        self.text.configure(yscrollcommand=self._on_yscroll)

        # tags
        self.text.tag_configure("rep_line", background="#ffd7d7")  # red-ish for all mismatches
        self.text.tag_configure("row_even", background="#f7f7f7")  # zebra
        self.text.tag_configure("char_diff", underline=True)

        # slightly nicer tab stops
        self.text.configure(tabs=("1c",))

        # Mouse wheel bindings (so each pane can scroll independently)
        self.text.bind("<MouseWheel>", self._mw)
        self.text.bind("<Button-4>", self._mw)  # Linux
        self.text.bind("<Button-5>", self._mw)

    def _set_mono_font(self, widget):
        # Choose a decent monospaced font across platforms
        try:
            if platform.system() == "Darwin":
                widget.configure(font=("Menlo", 12))
            elif platform.system() == "Windows":
                widget.configure(font=("Consolas", 11))
            else:
                widget.configure(font=("DejaVu Sans Mono", 11))
        except tk.TclError:
            pass

    def _mw(self, event):
        # Normalize delta
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        else:
            delta = -1 if event.delta > 0 else 1
        self.text.yview_scroll(delta, "units")
        return "break"

    def _on_yscroll(self, *args):
        # update scrollbar position
        self.vsb.set(*args)

        # refresh gutter numbers for visible window
        first = self.text.index("@0,0")
        last = self.text.index(f"@0,{self.text.winfo_height()}")
        first_ln = int(first.split(".")[0])
        last_ln = int(last.split(".")[0]) + 1

        lines = [f"{i}\n" for i in range(first_ln, last_ln)]
        self.gutter.configure(state="normal")
        self.gutter.delete("1.0", "end")
        self.gutter.insert("1.0", "".join(lines))
        self.gutter.configure(state="disabled")
        self.gutter.yview_moveto(self.text.yview()[0])

    # Public API
    def yview_moveto(self, fraction: float):
        self.text.yview_moveto(fraction)
        # keep gutter aligned
        self._on_yscroll(*self.text.yview())

    def yview(self, *args):
        self.text.yview(*args)
        self._on_yscroll(*self.text.yview())

    def xview(self, *args):
        self.text.xview(*args)

    def set_text(self, content: str):
        self.text.delete("1.0", "end")
        if content:
            self.text.insert("1.0", content)
        self._refresh_gutter()
        self._stripe_rows()

    def get_text(self):
        return self.text.get("1.0", "end-1c")

    def copy_all(self):
        """Copy all text from this pane to clipboard."""
        txt = self.get_text()
        self.clipboard_clear()
        self.clipboard_append(txt)
        self.update()  # keep contents after app closes

    def clear(self):
        self.text.delete("1.0", "end")
        self._refresh_gutter()

    # Internal helpers
    def _refresh_gutter(self):
        total = int(self.text.index("end-1c").split(".")[0]) or 1
        self.gutter.configure(state="normal")
        self.gutter.delete("1.0", "end")
        self.gutter.insert("1.0", "".join(f"{i}\n" for i in range(1, total + 1)))
        self.gutter.configure(state="disabled")

    def _stripe_rows(self):
        self.text.tag_remove("row_even", "1.0", "end")
        total = int(self.text.index("end-1c").split(".")[0]) or 1
        for ln in range(1, total + 1, 2):
            self.text.tag_add("row_even", f"{ln}.0", f"{ln}.end")


# -------------------------- Main App --------------------------
class TextCompareApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.minsize(1200, 700)
        try:
            self.call("tk", "scaling", 1.2)
        except tk.TclError:
            pass

        self.ignore_case = tk.BooleanVar(value=False)
        self.ignore_ws = tk.BooleanVar(value=False)
        self.view_mode = tk.StringVar(value="all")  # all | diff | same

        self._opcodes = []
        self._left_lines_raw = []
        self._right_lines_raw = []

        self._build_ui()
        self._bind_shortcuts()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # Toolbar
        bar = ttk.Frame(self, padding=8)
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(99, weight=1)

        ttk.Button(bar, text="Open Left…", command=self.open_left).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(bar, text="Open Right…", command=self.open_right).grid(row=0, column=1, padx=6)
        ttk.Button(bar, text="Clear Left", command=lambda: self.left.clear()).grid(row=0, column=2, padx=6)
        ttk.Button(bar, text="Clear Right", command=lambda: self.right.clear()).grid(row=0, column=3, padx=6)

        ttk.Label(bar, text="").grid(row=0, column=99, sticky="ew")  # spacer

        ttk.Checkbutton(bar, text="Ignore case", variable=self.ignore_case).grid(row=0, column=100, padx=6)
        ttk.Checkbutton(bar, text="Ignore whitespace", variable=self.ignore_ws).grid(row=0, column=101, padx=6)

        # View mode (All / Difference / Same)
        vm = ttk.Frame(bar)
        vm.grid(row=0, column=102, padx=(12, 0))
        for i, (lab, val) in enumerate([("All", "all"), ("Difference", "diff"), ("Same", "same")]):
            ttk.Radiobutton(vm, text=lab, value=val, variable=self.view_mode,
                            command=self._apply_view_mode).grid(row=0, column=i, padx=2)

        ttk.Button(bar, text="Compare (Ctrl/Cmd+R)", command=self.compare).grid(row=0, column=103, padx=10)
        ttk.Button(bar, text="Export Diff (Ctrl/Cmd+E)", command=self.export_diff).grid(row=0, column=104, padx=(6, 0))

        # Two panes
        main = ttk.Frame(self, padding=(8, 0, 8, 8))
        main.grid(row=2, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self.left = TextPane(main, "Left (Original / A)")
        self.right = TextPane(main, "Right (Modified / B)")
        self.left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        # Perfect two-way vertical sync (scroll either pane; the other follows)
        def sync_from_left(*_):
            frac = self.left.text.yview()[0]
            self.right.yview_moveto(frac)
            return "break"

        def sync_from_right(*_):
            frac = self.right.text.yview()[0]
            self.left.yview_moveto(frac)
            return "break"

        # Wrap each yscrollcommand so we can update gutter & mirror position
        self.left.text.configure(yscrollcommand=lambda *a: (self.left._on_yscroll(*a), sync_from_left()))
        self.right.text.configure(yscrollcommand=lambda *a: (self.right._on_yscroll(*a), sync_from_right()))

        # Status bar
        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, anchor="w", relief="sunken", padding=(8, 2)).grid(
            row=3, column=0, sticky="ew"
        )

    def _bind_shortcuts(self):
        mod = "Command" if platform.system() == "Darwin" else "Control"
        self.bind_all(f"<{mod}-r>", lambda e: self.compare())
        self.bind_all(f"<{mod}-e>", lambda e: self.export_diff())

    # -------------------------- File ops --------------------------
    def _read_text_file(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1", errors="ignore") as f:
                return f.read()

    def open_left(self):
        path = filedialog.askopenfilename(title="Open Left (A)")
        if not path:
            return
        self.left.set_text(self._read_text_file(path))
        self.status.set(f"Loaded Left: {path.split('/')[-1]}")

    def open_right(self):
        path = filedialog.askopenfilename(title="Open Right (B)")
        if not path:
            return
        self.right.set_text(self._read_text_file(path))
        self.status.set(f"Loaded Right: {path.split('/')[-1]}")

    def export_diff(self):
        a = self.left.get_text().splitlines(False)
        b = self.right.get_text().splitlines(False)

        if self.ignore_ws.get():
            a = [" ".join(x.split()) for x in a]
            b = [" ".join(x.split()) for x in b]
        if self.ignore_case.get():
            a = [x.lower() for x in a]
            b = [x.lower() for x in b]

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = filedialog.asksaveasfilename(defaultextension=".patch",
                                            initialfile=f"diff_{ts}.patch",
                                            title="Save Unified Diff")
        if not dest:
            return
        try:
            with open(dest, "w", encoding="utf-8") as f:
                f.write("\n".join(difflib.unified_diff(a, b, fromfile="left", tofile="right", lineterm="")))
            self.status.set(f"Saved diff: {dest.split('/')[-1]}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save diff:\n{e}")

    # -------------------------- Compare & view modes --------------------------
    def compare(self):
        self.status.set("Comparing…")

        # keep raw lines (with newlines) for display
        self._left_lines_raw = self.left.get_text().splitlines(keepends=True)
        self._right_lines_raw = self.right.get_text().splitlines(keepends=True)

        # normalized versions for comparison
        L = [l[:-1] if l.endswith("\n") else l for l in self._left_lines_raw]
        R = [r[:-1] if r.endswith("\n") else r for r in self._right_lines_raw]

        if self.ignore_ws.get():
            L = [" ".join(x.split()) for x in L]
            R = [" ".join(x.split()) for x in R]
        if self.ignore_case.get():
            L = [x.lower() for x in L]
            R = [x.lower() for x in R]

        sm = difflib.SequenceMatcher(None, L, R)
        self._opcodes = sm.get_opcodes()

        # render full text and apply highlights
        self.left.set_text("".join(self._left_lines_raw))
        self.right.set_text("".join(self._right_lines_raw))

        for tag, i1, i2, j1, j2 in self._opcodes:
            if tag == "equal":
                continue
            # mark mismatched lines red
            for li in range(i1, i2):
                self.left.text.tag_add("rep_line", f"{li+1}.0", f"{li+1}.end")
            for rj in range(j1, j2):
                self.right.text.tag_add("rep_line", f"{rj+1}.0", f"{rj+1}.end")

            # character-level emphasis for replacements
            if tag == "replace":
                pairs = min(i2 - i1, j2 - j1)
                for k in range(pairs):
                    Ls = self._left_lines_raw[i1 + k].rstrip("\n")
                    Rs = self._right_lines_raw[j1 + k].rstrip("\n")
                    for t, a1, a2, b1, b2 in difflib.SequenceMatcher(None, Ls, Rs).get_opcodes():
                        if t == "equal":
                            continue
                        self.left.text.tag_add("char_diff", f"{i1+k+1}.{a1}", f"{i1+k+1}.{a2}")
                        self.right.text.tag_add("char_diff", f"{j1+k+1}.{b1}", f"{j1+k+1}.{b2}")

        # re-apply zebra after tags
        self.left._stripe_rows()
        self.right._stripe_rows()

        self._apply_view_mode()
        self.status.set("Compared")

    def _apply_view_mode(self):
        if not self._opcodes:
            return
        mode = self.view_mode.get()  # all | diff | same
        if mode == "all":
            # show full content (already rendered), just refresh gutters/stripes
            self.left._refresh_gutter(); self.left._stripe_rows()
            self.right._refresh_gutter(); self.right._stripe_rows()
            return

        left_out, right_out = [], []
        left_tags, right_tags = [], []
        lcur = 1
        rcur = 1

        for tag, i1, i2, j1, j2 in self._opcodes:
            eq = (tag == "equal")
            include = (mode == "same" and eq) or (mode == "diff" and not eq)
            if not include:
                continue

            for li in range(i1, i2):
                left_out.append(self._left_lines_raw[li])
                if not eq:  # diff mode: keep red highlight
                    left_tags.append((lcur, "rep_line"))
                lcur += 1

            for rj in range(j1, j2):
                right_out.append(self._right_lines_raw[rj])
                if not eq:
                    right_tags.append((rcur, "rep_line"))
                rcur += 1

        # load filtered text
        self.left.set_text("".join(left_out))
        self.right.set_text("".join(right_out))

        # reapply line-level tags in filtered views
        for ln, tname in left_tags:
            self.left.text.tag_add(tname, f"{ln}.0", f"{ln}.end")
        for ln, tname in right_tags:
            self.right.text.tag_add(tname, f"{ln}.0", f"{ln}.end")

        self.left._stripe_rows(); self.right._stripe_rows()


# -------------------------- Run --------------------------
def main():
    app = TextCompareApp()
    app.mainloop()

if __name__ == "__main__":
    main()
