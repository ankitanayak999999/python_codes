#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import difflib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import os
import datetime

APP_TITLE = "Text Compare (GUI)"
APP_MIN_W, APP_MIN_H = 1100, 650

class TextCompareApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.minsize(APP_MIN_W, APP_MIN_H)
        try:
            self.call('tk', 'scaling', 1.2)  # a little nicer on HiDPI
        except tk.TclError:
            pass

        self._build_ui()
        self._bind_shortcuts()

    # ----------------------- UI -----------------------
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # Top bar (buttons & options)
        top = ttk.Frame(self, padding=(8, 8))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(99, weight=1)  # spacer

        self.btn_open_left = ttk.Button(top, text="Open Left…", command=self.open_left)
        self.btn_open_right = ttk.Button(top, text="Open Right…", command=self.open_right)
        self.btn_clear_left = ttk.Button(top, text="Clear Left", command=lambda: self.clear_text(self.left_text))
        self.btn_clear_right = ttk.Button(top, text="Clear Right", command=lambda: self.clear_text(self.right_text))
        self.btn_compare = ttk.Button(top, text="Compare (⌘/Ctrl+R)", command=self.compare)
        self.btn_export = ttk.Button(top, text="Export Diff (⌘/Ctrl+E)", command=self.export_diff)

        self.ignore_case_var = tk.BooleanVar(value=False)
        self.ignore_ws_var = tk.BooleanVar(value=False)
        self.chk_ignore_case = ttk.Checkbutton(top, text="Ignore case", variable=self.ignore_case_var)
        self.chk_ignore_ws = ttk.Checkbutton(top, text="Ignore whitespace", variable=self.ignore_ws_var)

        self.btn_open_left.grid(row=0, column=0, padx=(0,6))
        self.btn_open_right.grid(row=0, column=1, padx=6)
        self.btn_clear_left.grid(row=0, column=2, padx=6)
        self.btn_clear_right.grid(row=0, column=3, padx=6)
        ttk.Label(top, text="").grid(row=0, column=99, sticky="ew")  # spacer
        self.chk_ignore_case.grid(row=0, column=100, padx=(6,6))
        self.chk_ignore_ws.grid(row=0, column=101, padx=(6,6))
        self.btn_compare.grid(row=0, column=102, padx=(12,6))
        self.btn_export.grid(row=0, column=103, padx=(6,0))

        # Middle bar (labels & quick copy)
        mid = ttk.Frame(self, padding=(8, 0, 8, 4))
        mid.grid(row=1, column=0, sticky="ew")
        mid.columnconfigure(0, weight=1)
        mid.columnconfigure(1, weight=1)

        left_head = ttk.Frame(mid)
        right_head = ttk.Frame(mid)
        left_head.grid(row=0, column=0, sticky="w")
        right_head.grid(row=0, column=1, sticky="w")

        ttk.Label(left_head, text="Left (Original / A)", font=("TkDefaultFont", 10, "bold")).pack(side="left")
        ttk.Button(left_head, text="Copy", command=lambda: self.copy_text(self.left_text), width=6).pack(side="left", padx=(8,0))

        ttk.Label(right_head, text="Right (Modified / B)", font=("TkDefaultFont", 10, "bold")).pack(side="left")
        ttk.Button(right_head, text="Copy", command=lambda: self.copy_text(self.right_text), width=6).pack(side="left", padx=(8,0))

        # Main pane: two scrolled text widgets
        main = ttk.Frame(self, padding=(8, 0, 8, 8))
        main.grid(row=2, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self.left_text = ScrolledText(main, wrap="none", undo=True)
        self.right_text = ScrolledText(main, wrap="none", undo=True)
        self.left_text.grid(row=0, column=0, sticky="nsew", padx=(0,4))
        self.right_text.grid(row=0, column=1, sticky="nsew", padx=(4,0))

        # Configure tags (colors kept simple & readable)
        self._config_tags(self.left_text)
        self._config_tags(self.right_text)

        # Sync scrolling
        self._sync_scroll(self.left_text, self.right_text)

        # Status bar
        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w", padding=(8, 2)).grid(
            row=3, column=0, sticky="ew"
        )

    def _config_tags(self, widget: ScrolledText):
        widget.tag_configure("del_line", background="#ffecec")     # red-ish background
        widget.tag_configure("ins_line", background="#eaffea")     # green-ish background
        widget.tag_configure("rep_line", background="#fff7cc")     # yellow-ish background
        widget.tag_configure("char_del", underline=True)           # emphasis within changed lines
        widget.tag_configure("char_ins", underline=True)

        # Make text a bit nicer
        try:
            widget.configure(font=("Menlo", 12))
        except tk.TclError:
            try:
                widget.configure(font=("Consolas", 11))
            except tk.TclError:
                pass

    def _sync_scroll(self, t1: ScrolledText, t2: ScrolledText):
        # Vertical sync
        def on_scrollbar(*args):
            t1.yview(*args)
            t2.yview_moveto(t1.yview()[0])
            return "break"

        def on_mousewheel(event, target: ScrolledText, peer: ScrolledText):
            # normalize delta across platforms
            delta = 0
            if event.num == 4: delta = -1
            elif event.num == 5: delta = 1
            elif event.delta: delta = -1 if event.delta > 0 else 1
            target.yview_scroll(delta, "units")
            peer.yview_moveto(target.yview()[0])
            return "break"

        # Tie t1's scrollbar to a custom command to update t2
        t1.vbar.config(command=on_scrollbar)
        # Mouse wheel bindings (macOS/Windows/Linux)
        for widget in (t1, t2):
            widget.bind("<MouseWheel>", lambda e, w=widget: on_mousewheel(e, w, t2 if w is t1 else t1))
            widget.bind("<Button-4>",  lambda e, w=widget: on_mousewheel(e, w, t2 if w is t1 else t1))
            widget.bind("<Button-5>",  lambda e, w=widget: on_mousewheel(e, w, t2 if w is t1 else t1))

    def _bind_shortcuts(self):
        mod = "Command" if self._is_macos() else "Control"
        self.bind_all(f"<{mod}-o>", lambda e: self.open_left())
        self.bind_all(f"<{mod}-e>", lambda e: self.export_diff())
        self.bind_all(f"<{mod}-r>", lambda e: self.compare())
        self.bind_all(f"<{mod}-l>", lambda e: self.clear_text(self.left_text))
        self.bind_all(f"<{mod}-k>", lambda e: self.clear_text(self.right_text))

    def _is_macos(self):
        return os.uname().sysname == "Darwin" if hasattr(os, "uname") else False

    # ----------------------- Actions -----------------------
    def open_left(self):
        path = filedialog.askopenfilename(title="Open Left (A)")
        if not path: return
        try:
            txt = self._read_text_file(path)
            self.left_text.delete("1.0", "end")
            self.left_text.insert("1.0", txt)
            self._clear_all_tags()
            self.status.set(f"Loaded Left: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file:\n{e}")

    def open_right(self):
        path = filedialog.askopenfilename(title="Open Right (B)")
        if not path: return
        try:
            txt = self._read_text_file(path)
            self.right_text.delete("1.0", "end")
            self.right_text.insert("1.0", txt)
            self._clear_all_tags()
            self.status.set(f"Loaded Right: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file:\n{e}")

    def clear_text(self, widget: ScrolledText):
        widget.delete("1.0", "end")
        self._clear_all_tags()
        self.status.set("Cleared")

    def copy_text(self, widget: ScrolledText):
        txt = widget.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(txt)
        self.status.set("Copied")

    def export_diff(self):
        left = self.left_text.get("1.0", "end-1c")
        right = self.right_text.get("1.0", "end-1c")
        a = left.splitlines(keepends=False)
        b = right.splitlines(keepends=False)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = f"diff_{ts}.patch"
        path = filedialog.asksaveasfilename(
            defaultextension=".patch", initialfile=default, title="Save Unified Diff"
        )
        if not path:
            return

        if self.ignore_ws_var.get():
            a_norm = [self._normalize_ws(s) for s in a]
            b_norm = [self._normalize_ws(s) for s in b]
        else:
            a_norm, b_norm = a, b

        if self.ignore_case_var.get():
            a_norm = [s.lower() for s in a_norm]
            b_norm = [s.lower() for s in b_norm]

        diff_lines = difflib.unified_diff(
            a_norm, b_norm, fromfile="left", tofile="right", lineterm=""
        )
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(diff_lines))
            self.status.set(f"Saved diff: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save diff:\n{e}")

    def compare(self):
        self._clear_all_tags()
        left_raw = self.left_text.get("1.0", "end-1c")
        right_raw = self.right_text.get("1.0", "end-1c")

        # For highlighting, we compute ops on possibly normalized content,
        # but we still apply tags to the ORIGINAL text positions line-wise.
        left_lines = left_raw.splitlines(keepends=True)
        right_lines = right_raw.splitlines(keepends=True)
        left_cmp = [l[:-1] if l.endswith("\n") else l for l in left_lines]
        right_cmp = [r[:-1] if r.endswith("\n") else r for r in right_lines]

        if self.ignore_ws_var.get():
            left_cmp  = [self._normalize_ws(s) for s in left_cmp]
            right_cmp = [self._normalize_ws(s) for s in right_cmp]
        if self.ignore_case_var.get():
            left_cmp  = [s.lower() for s in left_cmp]
            right_cmp = [s.lower() for s in right_cmp]

        sm = difflib.SequenceMatcher(None, left_cmp, right_cmp)
        opcodes = sm.get_opcodes()  # list of (tag, i1, i2, j1, j2)

        # Track line indexes to apply tags
        left_line_offsets = self._line_start_indices(left_lines)
        right_line_offsets = self._line_start_indices(right_lines)

        # Apply line-level tags, and character-level inside 'replace'
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                continue

            if tag in ("delete", "replace"):
                # highlight left lines
                for li in range(i1, i2):
                    self._tag_full_line(self.left_text, left_line_offsets, li, "del_line" if tag=="delete" else "rep_line")

            if tag in ("insert", "replace"):
                # highlight right lines
                for rj in range(j1, j2):
                    self._tag_full_line(self.right_text, right_line_offsets, rj, "ins_line" if tag=="insert" else "rep_line")

            # Char-level detail for replacements (best effort)
            if tag == "replace":
                # Align pairs by index where possible
                pairs = min(i2 - i1, j2 - j1)
                for k in range(pairs):
                    l_idx = i1 + k
                    r_idx = j1 + k
                    l_line = (left_lines[l_idx][:-1] if left_lines[l_idx].endswith("\n") else left_lines[l_idx])
                    r_line = (right_lines[r_idx][:-1] if right_lines[r_idx].endswith("\n") else right_lines[r_idx])
                    self._highlight_char_diff(self.left_text, left_line_offsets, l_idx, l_line, r_line, side="left")
                    self._highlight_char_diff(self.right_text, right_line_offsets, r_idx, l_line, r_line, side="right")

        self.status.set("Compared")

    # ----------------------- Helpers -----------------------
    def _highlight_char_diff(self, widget, line_offsets, line_idx, left_line, right_line, side="left"):
        sm = difflib.SequenceMatcher(None, left_line, right_line)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            # Map char spans to text indices
            start_idx = f"{line_idx+1}.{i1 if side=='left' else j1}"
            end_idx   = f"{line_idx+1}.{i2 if side=='left' else j2}"
            widget.tag_add("char_del" if side=="left" else "char_ins", start_idx, end_idx)

    def _tag_full_line(self, widget, line_offsets, line_idx, tag_name):
        if line_idx >= len(line_offsets):
            return
        start = f"{line_idx+1}.0"
        end = f"{line_idx+1}.end"
        widget.tag_add(tag_name, start, end)

    def _line_start_indices(self, lines):
        # Not needed for current tagging (we use "line.column"), but kept for clarity/extension
        return [0] * len(lines)

    def _normalize_ws(self, s: str) -> str:
        return " ".join(s.split())

    def _read_text_file(self, path: str) -> str:
        # Try utf-8 then fall back latin-1
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1", errors="ignore") as f:
                return f.read()

    def _clear_all_tags(self):
        for t in ("del_line", "ins_line", "rep_line", "char_del", "char_ins"):
            self.left_text.tag_remove(t, "1.0", "end")
            self.right_text.tag_remove(t, "1.0", "end")

def main():
    app = TextCompareApp()
    app.mainloop()

if __name__ == "__main__":
    main()
