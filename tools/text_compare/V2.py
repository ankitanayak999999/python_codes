#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import difflib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import datetime

APP_TITLE = "Text Compare (GUI) – v2"

# -------------------------- Composite widgets --------------------------
class TextPane(ttk.Frame):
    """
    A composite widget: (gutter line numbers) + Text with both scrollbars.
    Provides API: set_text(), clear(), get_text(), yview, yview_moveto, xview, etc.
    """
    def __init__(self, master, title=""):
        super().__init__(master)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(1, weight=1)

        # Header
        head = ttk.Frame(self)
        head.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0,2))
        ttk.Label(head, text=title, font=("TkDefaultFont", 10, "bold")).pack(side="left")
        ttk.Button(head, text="Copy", width=6, command=self.copy_all).pack(side="left", padx=6)

        # Line-number gutter (disabled Text)
        self.gutter = tk.Text(self, width=6, padx=4, state="disabled", wrap="none")
        self.gutter.configure(font=("Menlo", 11) if self._try_font("Menlo") else ("Consolas", 11))
        self.gutter.grid(row=1, column=0, sticky="nsw", padx=(0,4))

        # Vertical scrollbar
        self.vsb = ttk.Scrollbar(self, orient="vertical")
        self.vsb.grid(row=1, column=1, sticky="ns")

        # Text + horizontal scrollbar
        self.text = tk.Text(self, wrap="none", undo=True)
        self.text.configure(font=("Menlo", 11) if self._try_font("Menlo") else ("Consolas", 11))
        self.text.grid(row=1, column=2, sticky="nsew")

        self.hsb = ttk.Scrollbar(self, orient="horizontal", command=self.text.xview)
        self.hsb.grid(row=2, column=2, sticky="ew")
        self.text.configure(xscrollcommand=self.hsb.set)

        # Connect vertical scroll via our handler (so we can sync gutter)
        self.text.configure(yscrollcommand=self._on_yscroll)

        # Tags for highlighting and row striping
        self.text.tag_configure("del_line", background="#ffd7d7")   # red-ish for mismatches
        self.text.tag_configure("ins_line", background="#ffd7d7")   # also red-ish (as requested)
        self.text.tag_configure("rep_line", background="#ffd7d7")   # also red-ish
        self.text.tag_configure("row_even", background="#f7f7f7")
        self.text.tag_configure("char_diff", underline=True)

        # Improve readability
        self.text.configure(tabs=("1c",))  # tab width ~ 8 spaces visually aligned

        # Mouse wheel bindings
        self.text.bind("<MouseWheel>", self._mw)
        self.text.bind("<Button-4>", self._mw)  # Linux
        self.text.bind("<Button-5>", self._mw)

    def _try_font(self, name):
        try:
            self.text.configure(font=(name, 11))
            return True
        except tk.TclError:
            return False

    def _mw(self, event):
        # Normalize delta
        if event.num == 4: delta = -1
        elif event.num == 5: delta = 1
        else: delta = -1 if event.delta > 0 else 1
        self.text.yview_scroll(delta, "units")
        return "break"

    def _on_yscroll(self, *args):
        # keep vsb and gutter synced
        self.vsb.set(*args)

        # update gutter to match visible lines
        first = self.text.index("@0,0")
        last = self.text.index("@0,%d" % self.text.winfo_height())
        first_line = int(first.split(".")[0])
        last_line = int(last.split(".")[0]) + 1

        lines = [f"{i}\n" for i in range(first_line, last_line)]
        self.gutter.configure(state="normal")
        self.gutter.delete("1.0", "end")
        self.gutter.insert("1.0", "".join(lines))
        self.gutter.configure(state="disabled")
        # and scroll gutter vertically with text
        self.gutter.yview_moveto(self.text.yview()[0])

    def yview_moveto(self, fraction: float):
        self.text.yview_moveto(fraction)

    def yview(self, *args):
        self.text.yview(*args)

    def xview(self, *args):
        self.text.xview(*args)

    def set_text(self, content: str):
        self.text.delete("1.0", "end")
        if content:
            self.text.insert("1.0", content)
        self._refresh_gutter()
        self._stripe_rows()

    def _refresh_gutter(self):
        total = int(self.text.index("end-1c").split(".")[0]) or 1
        self.gutter.configure(state="normal")
        self.gutter.delete("1.0", "end")
        self.gutter.insert("1.0", "".join(f"{i}\n" for i in range(1, total+1)))
        self.gutter.configure(state="disabled")

    def _stripe_rows(self):
        # clear previous stripes
        self.text.tag_remove("row_even", "1.0", "end")
        # apply zebra striping
        total = int(self.text.index("end-1c").split(".")[0]) or 1
        for ln in range(1, total+1, 2):
            self.text.tag_add("row_even", f"{ln}.0", f"{ln}.end")

    def clear(self):
        self.text.delete("1.0", "end")
        self._refresh_gutter()

    def get_text(self):
        return self.text.get("1.0", "end-1c")

# -------------------------- Main App --------------------------
class TextCompareApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.minsize(1200, 700)
        try:
            self.call('tk', 'scaling', 1.2)
        except tk.TclError:
            pass

        self.ignore_case = tk.BooleanVar(value=False)
        self.ignore_ws = tk.BooleanVar(value=False)
        self.view_mode = tk.StringVar(value="all")  # all | diff | same

        self._build_ui()
        self._bind_shortcuts()

        # state for diffing
        self._opcodes = []
        self._left_lines_raw = []
        self._right_lines_raw = []

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # Toolbar
        bar = ttk.Frame(self, padding=8)
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(99, weight=1)
        ttk.Button(bar, text="Open Left…", command=self.open_left).grid(row=0, column=0, padx=(0,6))
        ttk.Button(bar, text="Open Right…", command=self.open_right).grid(row=0, column=1, padx=6)
        ttk.Button(bar, text="Clear Left", command=lambda: self.left.clear()).grid(row=0, column=2, padx=6)
        ttk.Button(bar, text="Clear Right", command=lambda: self.right.clear()).grid(row=0, column=3, padx=6)

        ttk.Label(bar, text="").grid(row=0, column=99, sticky="ew")  # spacer

        ttk.Checkbutton(bar, text="Ignore case", variable=self.ignore_case).grid(row=0, column=100, padx=6)
        ttk.Checkbutton(bar, text="Ignore whitespace", variable=self.ignore_ws).grid(row=0, column=101, padx=6)

        # View mode
        vm = ttk.Frame(bar)
        vm.grid(row=0, column=102, padx=(12,0))
        for i, (lab, val) in enumerate([("All", "all"), ("Difference", "diff"), ("Same", "same")]):
            ttk.Radiobutton(vm, text=lab, value=val, variable=self.view_mode, command=self._apply_view_mode).grid(row=0, column=i, padx=2)

        ttk.Button(bar, text="Compare (Ctrl/Cmd+R)", command=self.compare).grid(row=0, column=103, padx=10)
        ttk.Button(bar, text="Export Diff (Ctrl/Cmd+E)", command=self.export_diff).grid(row=0, column=104, padx=(6,0))

        # Panes
        main = ttk.Frame(self, padding=(8,0,8,8))
        main.grid(row=2, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self.left = TextPane(main, "Left (Original / A)")
        self.right = TextPane(main, "Right (Modified / B)")
        self.left.grid(row=0, column=0, sticky="nsew", padx=(0,6))
        self.right.grid(row=0, column=1, sticky="nsew", padx=(6,0))

        # Perfect two-way vertical sync
        def sync_left(*args):
            frac = self.left.text.yview()[0]
            self.right.yview_moveto(frac)
            return "break"

        def sync_right(*args):
            frac = self.right.text.yview()[0]
            self.left.yview_moveto(frac)
            return "break"

        self.left.text['yscrollcommand'] = lambda *a: (self.left._on_yscroll(*a), sync_left())
        self.right.text['yscrollcommand'] = lambda *a: (self.right._on_yscroll(*a), sync_right())

        # Status bar
        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, anchor="w", relief="sunken", padding=(8,2)).grid(row=3, column=0, sticky="ew")

    def _bind_shortcuts(self):
        mod = "Command" if (hasattr(os, "uname") and os.uname().sysname == "Darwin") else "Control"
        self.bind_all(f"<{mod}-r>", lambda e: self.compare())
        self.bind_all(f"<{mod}-e>", lambda e: self.export_diff())

    # -------------------------- File ops --------------------------
    def open_left(self):
        path = filedialog.askopenfilename(title="Open Left (A)")
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f: txt = f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1", errors="ignore") as f: txt = f.read()
        self.left.set_text(txt); self.status.set(f"Loaded Left: {os.path.basename(path)}")

    def open_right(self):
        path = filedialog.askopenfilename(title="Open Right (B)")
        if not path: return
        try:
            with open(path, "r", encoding="utf-8") as f: txt = f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1", errors="ignore") as f: txt = f.read()
        self.right.set_text(txt); self.status.set(f"Loaded Right: {os.path.basename(path)}")

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
        dest = filedialog.asksaveasfilename(defaultextension=".patch", initialfile=f"diff_{ts}.patch")
        if not dest: return
        try:
            with open(dest, "w", encoding="utf-8") as f:
                f.write("\n".join(difflib.unified_diff(a, b, fromfile="left", tofile="right", lineterm="")))
            self.status.set(f"Saved diff: {os.path.basename(dest)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save diff:\n{e}")

    # -------------------------- Compare --------------------------
    def compare(self):
        self.status.set("Comparing…")
        # raw lines (keep newline so indices remain stable)
        left_raw = self.left.get_text()
        right_raw = self.right.get_text()
        self._left_lines_raw = left_raw.splitlines(keepends=True)
        self._right_lines_raw = right_raw.splitlines(keepends=True)

        # compare lines (normalized as requested)
        left_cmp  = [l[:-1] if l.endswith("\n") else l for l in self._left_lines_raw]
        right_cmp = [r[:-1] if r.endswith("\n") else r for r in self._right_lines_raw]

        if self.ignore_ws.get():
            left_cmp  = [" ".join(x.split()) for x in left_cmp]
            right_cmp = [" ".join(x.split()) for x in right_cmp]
        if self.ignore_case.get():
            left_cmp  = [x.lower() for x in left_cmp]
            right_cmp = [x.lower() for x in right_cmp]

        sm = difflib.SequenceMatcher(None, left_cmp, right_cmp)
        self._opcodes = sm.get_opcodes()

        # Always render full text first, then apply view filters
        self.left.set_text("".join(self._left_lines_raw))
        self.right.set_text("".join(self._right_lines_raw))

        # Line-level highlights (all mismatches red per your request)
        for tag, i1, i2, j1, j2 in self._opcodes:
            if tag == "equal": continue
            for li in range(i1, i2):
                self.left.text.tag_add("rep_line", f"{li+1}.0", f"{li+1}.end")
            for rj in range(j1, j2):
                self.right.text.tag_add("rep_line", f"{rj+1}.0", f"{rj+1}.end")

            # character-level emphasis for replacements (best-effort)
            if tag == "replace":
                pairs = min(i2 - i1, j2 - j1)
                for k in range(pairs):
                    L = self._left_lines_raw[i1 + k].rstrip("\n")
                    R = self._right_lines_raw[j1 + k].rstrip("\n")
                    for t, a1, a2, b1, b2 in difflib.SequenceMatcher(None, L, R).get_opcodes():
                        if t == "equal": continue
                        self.left.text.tag_add("char_diff", f"{i1+k+1}.{a1}", f"{i1+k+1}.{a2}")
                        self.right.text.tag_add("char_diff", f"{j1+k+1}.{b1}", f"{j1+k+1}.{b2}")

        # Apply zebra again after tags
        self.left._stripe_rows(); self.right._stripe_rows()

        # Apply view mode (all/diff/same)
        self._apply_view_mode()
        self.status.set("Compared")

    # View-mode filter rebuilds the panes showing only selected lines
    def _apply_view_mode(self):
        if not self._opcodes:
            return

        mode = self.view_mode.get()  # all | diff | same
        if mode == "all":
            # Show full, nothing to filter; just resync gutters/stripes.
            self.left._refresh_gutter(); self.right._refresh_gutter()
            self.left._stripe_rows(); self.right._stripe_rows()
            return

        left_out = []
        right_out = []
        left_map_tags = []   # (ln_start, ln_end, tagname)
        right_map_tags = []

        l_cursor = 1
        r_cursor = 1

        for tag, i1, i2, j1, j2 in self._opcodes:
            is_equal = (tag == "equal")
            if (mode == "same" and is_equal) or (mode == "diff" and not is_equal):
                # include these opcode ranges in output
                for li in range(i1, i2):
                    left_out.append(self._left_lines_raw[li])
                    # remember to re-add red highlight for diff mode
                    if not is_equal:
                        left_map_tags.append((l_cursor, l_cursor, "rep_line"))
                    l_cursor += 1
                for rj in range(j1, j2):
                    right_out.append(self._right_lines_raw[rj])
                    if not is_equal:
                        right_map_tags.append((r_cursor, r_cursor, "rep_line"))
                    r_cursor += 1

        # load filtered text
        self.left.set_text("".join(left_out))
        self.right.set_text("".join(right_out))

        # reapply highlight tags for filtered panes
        for s, e, tname in left_map_tags:
            self.left.text.tag_add(tname, f"{s}.0", f"{e}.end")
        for s, e, tname in right_map_tags:
            self.right.text.tag_add(tname, f"{s}.0", f"{e}.end")

        self.left._stripe_rows(); self.right._stripe_rows()

# -------------------------- Run --------------------------
def main():
    app = TextCompareApp()
    app.mainloop()

if __name__ == "__main__":
    main()
