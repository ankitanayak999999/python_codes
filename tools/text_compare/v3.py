#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import difflib
import os
import sys
import platform
import html
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_TITLE   = "Text Compare (GUI) – Final"
DEBOUNCE_MS = 200  # debounce for auto-compare


# ----------------------------------------------------------------------
# PyInstaller-friendly template resolver
# ----------------------------------------------------------------------
def get_template_path():
    """
    Return absolute path to template/diff_template.html.
    Works in dev and when frozen by PyInstaller (onefile/onedir).
    """
    base = getattr(sys, "_MEIPASS", os.path.dirname(__file__))
    return os.path.join(base, "template", "diff_template.html")


# ======================================================================
#  UI: TextPane (one side of the diff)
# ======================================================================
class TextPane(ttk.Frame):
    def __init__(self, master, title=""):
        super().__init__(master)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(1, weight=1)

        head = ttk.Frame(self)
        head.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 2))
        ttk.Label(head, text=title, font=("TkDefaultFont", 10, "bold")).pack(side="left")
        ttk.Button(head, text="Copy", width=6, command=self.copy_all).pack(side="left", padx=6)

        # Line-number gutter
        self.gutter = tk.Text(self, width=6, padx=4, state="disabled", wrap="none",
                              relief="flat", background="#f0f0f0")
        self._set_mono_font(self.gutter)
        self.gutter.grid(row=1, column=0, sticky="nsw", padx=(0, 4))

        # Scrollbars
        self.vsb = ttk.Scrollbar(self, orient="vertical")
        self.vsb.grid(row=1, column=1, sticky="ns")

        # Main text widget
        self.text = tk.Text(self, wrap="none", undo=True)
        self._set_mono_font(self.text)
        self.text.grid(row=1, column=2, sticky="nsew")
        self.hsb = ttk.Scrollbar(self, orient="horizontal", command=self.text.xview)
        self.hsb.grid(row=2, column=2, sticky="ew")
        self.text.configure(xscrollcommand=self.hsb.set)

        # Link vertical scrollbar; keep gutter in sync
        self.vsb.config(command=self.text.yview)
        self.text.configure(yscrollcommand=self._on_yscroll)

        # Tags
        self.text.tag_configure("row_even",    background="#f7f7f7")   # zebra
        self.text.tag_configure("rep_line",    background="#ffd7d7")   # line-level diff
        self.text.tag_configure("cursor_line", background="#fff59d")   # current hunk (yellow)

        # char-level colored tags
        self.text.tag_configure("char_del", foreground="#c62828")      # red   (left deletions)
        self.text.tag_configure("char_add", foreground="#2e7d32")      # green (right insertions)
        self.text.tag_configure("char_rep", foreground="#ef6c00")      # orange (replacements)

        self.text.configure(tabs=("1c",))

        # Wheel (cross-platform)
        self.text.bind("<MouseWheel>", self._mw)
        self.text.bind("<Button-4>", self._mw)  # Linux
        self.text.bind("<Button-5>", self._mw)

    def _set_mono_font(self, widget):
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
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -1 if event.delta > 0 else 1
        self.text.yview_scroll(delta, "units")
        return "break"

    def _on_yscroll(self, *args):
        self.vsb.set(*args)
        first = int(self.text.index("@0,0").split(".")[0])
        last  = int(self.text.index(f"@0,{self.text.winfo_height()}").split(".")[0]) + 1
        nums = [f"{i}\n" for i in range(first, last)]
        self.gutter.configure(state="normal")
        self.gutter.delete("1.0", "end")
        self.gutter.insert("1.0", "".join(nums))
        self.gutter.configure(state="disabled")
        self.gutter.yview_moveto(self.text.yview()[0])

    def yview_moveto(self, fraction: float):
        self.text.yview_moveto(fraction)
        self._on_yscroll(*self.text.yview())

    def set_text(self, content: str):
        self.text.delete("1.0", "end")
        if content:
            self.text.insert("1.0", content)
        self.text.edit_modified(False)
        self._refresh_gutter()
        self._stripe_rows()

    def get_text(self):
        return self.text.get("1.0", "end-1c")

    def copy_all(self):
        txt = self.get_text()
        self.clipboard_clear()
        self.clipboard_append(txt)
        self.update()

    def clear(self):
        self.text.delete("1.0", "end")
        self.text.edit_modified(False)
        self._refresh_gutter()

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


# ======================================================================
#  EXPORTS: template-only HTML exporter (isolated for easy edits)
# ======================================================================
class HtmlExporter:
    """
    Template-only HTML exporter.

    Customize here:
      - get_template_path() decides where the template file lives
      - _default_css() only used if template contains {{CSS}}
    """
    def export(self, parent_window, view_name: str, left_rows, right_rows):
        template_path = get_template_path()
        if not os.path.exists(template_path):
            messagebox.showerror("Template Missing",
                                 f"Template not found:\n{template_path}")
            return

        try:
            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read()
        except Exception as e:
            messagebox.showerror("Template Error", f"Failed to read template:\n{e}")
            return

        rows_html = self.build_rows(left_rows, right_rows)
        try:
            html_text = self.render_with_template(template, view_name, rows_html)
        except ValueError as ve:
            messagebox.showerror("Template Error", str(ve))
            return

        out = filedialog.asksaveasfilename(
            parent=parent_window,
            defaultextension=".html",
            initialfile="diff_export.html",
            title="Save HTML"
        )
        if not out:
            return
        try:
            with open(out, "w", encoding="utf-8") as f:
                f.write(html_text)
            messagebox.showinfo("Export Complete", f"Saved HTML: {os.path.basename(out)}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save HTML:\n{e}")

    # ---- helpers used by export() ----
    def build_rows(self, left_rows, right_rows) -> str:
        """
        Build HTML <tr> rows only (no surrounding table).
        Each item: (line_no:int|None, text:str, is_diff:bool)
        """
        def td(txt, is_diff):
            cls = ' class="diff"' if is_diff else ''
            return f"<td{cls}><pre>{html.escape(txt)}</pre></td>"

        n = max(len(left_rows), len(right_rows))
        left_rows += [(None, "", False)] * (n - len(left_rows))
        right_rows += [(None, "", False)] * (n - len(right_rows))

        rows_html = []
        for i in range(n):
            lnum, ltxt, ldiff = left_rows[i]
            rnum, rtxt, rdiff = right_rows[i]
            lnum = "" if lnum is None else lnum
            rnum = "" if rnum is None else rnum
            rows_html.append(
                "<tr>"
                f"<td class='col-num'>{lnum}</td>{td(ltxt, ldiff)}"
                f"<td class='col-num'>{rnum}</td>{td(rtxt, rdiff)}"
                "</tr>"
            )
        return "\n".join(rows_html)

    def render_with_template(self, template: str, view_name: str, rows_html: str) -> str:
        if "{{TABLE_ROWS}}" not in template:
            raise ValueError("Template missing required marker {{TABLE_ROWS}}.")
        out = template
        # Optionals
        out = out.replace("{{TITLE}}", APP_TITLE)
        out = out.replace("{{VIEW}}", html.escape(view_name))
        out = out.replace("{{LEFT_HEADER}}",  "Left (Original / A)")
        out = out.replace("{{RIGHT_HEADER}}", "Right (Modified / B)")
        if "{{CSS}}" in out:
            out = out.replace("{{CSS}}", self._default_css())
        # Required
        out = out.replace("{{TABLE_ROWS}}", rows_html)
        return out

    def _default_css(self) -> str:
        # Injected only if template uses {{CSS}}
        return (
            "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#fff}"
            "table{border-collapse:separate;border-spacing:0;width:max-content;min-width:100%}"
            "th,td{border:1px solid #e0e0e0;vertical-align:top}"
            "thead th{position:sticky;top:0;background:#f6f8fa;padding:8px 10px;font:600 12px/1.2 system-ui}"
            "td{padding:0}"
            "pre{margin:0;padding:6px 10px;font:13px/1.45 Menlo,Consolas,monospace;white-space:pre}"
            "tr:nth-child(even) td{background:#fafafa}"
            "td.diff{background:#ffd7d7}"
            ".col-num{width:64px;min-width:64px;text-align:right;background:#f5f5f5;color:#6e7781;padding:6px 8px;font:12px Menlo,Consolas,monospace}"
            ".viewer{height:calc(100vh - 50px);margin:0 12px 12px;border:1px solid #e5e7eb;border-radius:6px;overflow:auto;background:#fff}"
        )


# ======================================================================
#  App: auto-compare + view modes + navigation + live counter
# ======================================================================
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
        self.ignore_ws   = tk.BooleanVar(value=False)
        self.view_mode   = tk.StringVar(value="all")  # all | diff | same

        self._opcodes = []
        self._left_lines_raw  = []
        self._right_lines_raw = []

        # Maps: original line -> display line (depends on view)
        self._left_map_display  = {}
        self._right_map_display = {}

        # Navigation targets: list of (left_display_line|None, right_display_line|None)
        self._nav_targets = []
        self._nav_index   = -1

        self._exporter = HtmlExporter()
        self._compare_after_id = None  # debounce handle

        self._build_ui()
        self._bind_auto_refresh()

        # Initial compute
        self._schedule_compare()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        bar = ttk.Frame(self, padding=8)
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(98, weight=0)
        bar.columnconfigure(99, weight=1)  # spacer

        ttk.Button(bar, text="Open Left…",  command=self.open_left).grid(row=0, column=0, padx=(0,6))
        ttk.Button(bar, text="Open Right…", command=self.open_right).grid(row=0, column=1, padx=6)
        ttk.Button(bar, text="Clear Left",  command=lambda: (self.left.clear(),  self._schedule_compare())).grid(row=0, column=2, padx=6)
        ttk.Button(bar, text="Clear Right", command=lambda: (self.right.clear(), self._schedule_compare())).grid(row=0, column=3, padx=6)

        # Navigation + counter
        ttk.Button(bar, text="Prev Change", command=self.prev_change).grid(row=0, column=90, padx=(12,4))
        ttk.Button(bar, text="Next Change", command=self.next_change).grid(row=0, column=91, padx=(4,8))
        self.nav_counter_var = tk.StringVar(value="0 / 0")
        ttk.Label(bar, textvariable=self.nav_counter_var).grid(row=0, column=92, padx=(0,12))

        ttk.Label(bar, text="").grid(row=0, column=99, sticky="ew")  # spacer
        ttk.Checkbutton(bar, text="Ignore case",       variable=self.ignore_case).grid(row=0, column=100, padx=6)
        ttk.Checkbutton(bar, text="Ignore whitespace", variable=self.ignore_ws).grid(row=0, column=101, padx=6)

        vm = ttk.Frame(bar); vm.grid(row=0, column=102, padx=(12,0))
        for i, (lab, val) in enumerate([("All","all"),("Difference","diff"),("Same","same")]):
            ttk.Radiobutton(vm, text=lab, value=val, variable=self.view_mode,
                            command=self._apply_view_mode).grid(row=0, column=i, padx=2)

        ttk.Button(bar, text="Export HTML", command=self.export_html).grid(row=0, column=104, padx=(12,0))

        main = ttk.Frame(self, padding=(8,0,8,8))
        main.grid(row=2, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self.left  = TextPane(main, "Left (Original / A)")
        self.right = TextPane(main, "Right (Modified / B)")
        self.left.grid(row=0, column=0, sticky="nsew", padx=(0,6))
        self.right.grid(row=0, column=1, sticky="nsew", padx=(6,0))

        # Two-way vertical sync
        def sync_from_left(*_):
            self.right.yview_moveto(self.left.text.yview()[0]); return "break"
        def sync_from_right(*_):
            self.left.yview_moveto(self.right.text.yview()[0]); return "break"
        self.left.text.configure(yscrollcommand=lambda *a: (self.left._on_yscroll(*a),  sync_from_left()))
        self.right.text.configure(yscrollcommand=lambda *a: (self.right._on_yscroll(*a), sync_from_right()))

        # Keyboard shortcuts for navigation
        self.bind_all("<F7>", lambda e: self.prev_change())
        self.bind_all("<Shift-F7>", lambda e: self.next_change())
        self.bind_all("<Alt-Up>", lambda e: self.prev_change())
        self.bind_all("<Alt-Down>", lambda e: self.next_change())

        # Status bar
        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, anchor="w", relief="sunken", padding=(8,2)).grid(
            row=3, column=0, sticky="ew"
        )

    # ---------------- Auto-refresh wiring ----------------
    def _bind_auto_refresh(self):
        self.ignore_case.trace_add("write", lambda *_: self._schedule_compare())
        self.ignore_ws.trace_add("write", lambda *_: self._schedule_compare())
        # Text changes
        for pane in (self.left.text, self.right.text):
            pane.bind("<<Modified>>", self._on_text_modified)

    def _on_text_modified(self, event):
        w = event.widget
        if w.edit_modified():
            w.edit_modified(False)
            self._schedule_compare()

    def _schedule_compare(self):
        if getattr(self, "_compare_after_id", None):
            try: self.after_cancel(self._compare_after_id)
            except Exception: pass
        self._compare_after_id = self.after(DEBOUNCE_MS, self.compare)

    # ---------------- File ops ----------------
    def _read_text_file(self, path: str) -> str:
        # tolerant file read
        try:
            with open(path, "r", encoding="utf-8") as f: return f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1", errors="ignore") as f: return f.read()

    def open_left(self):
        p = filedialog.askopenfilename(title="Open Left (A)")
        if not p: return
        self.left.set_text(self._read_text_file(p))
        self._schedule_compare()
        self._set_status(f"Loaded Left: {os.path.basename(p)}")

    def open_right(self):
        p = filedialog.askopenfilename(title="Open Right (B)")
        if not p: return
        self.right.set_text(self._read_text_file(p))
        self._schedule_compare()
        self._set_status(f"Loaded Right: {os.path.basename(p)}")

    # ---------------- Export ----------------
    def export_html(self):
        left_rows, right_rows = self._collect_view_rows_for_export()
        self._exporter.export(self, self.view_mode.get().capitalize(), left_rows, right_rows)

    # ---------------- Compare & view modes ----------------
    def compare(self):
        # raw (preserve newlines)
        self._left_lines_raw  = self.left.get_text().splitlines(keepends=True)
        self._right_lines_raw = self.right.get_text().splitlines(keepends=True)

        # normalized per options
        L = [l[:-1] if l.endswith("\n") else l for l in self._left_lines_raw]
        R = [r[:-1] if r.endswith("\n") else r for r in self._right_lines_raw]
        if self.ignore_ws.get():
            L = [" ".join(x.split()) for x in L]
            R = [" ".join(x.split()) for x in R]
        if self.ignore_case.get():
            L = [x.lower() for x in L]
            R = [x.lower() for x in R]

        self._opcodes = difflib.SequenceMatcher(None, L, R).get_opcodes()
        self._render_full_view()
        self._apply_view_mode()  # also rebuilds navigation targets & char diffs

    def _render_full_view(self):
        self.left.set_text("".join(self._left_lines_raw))
        self.right.set_text("".join(self._right_lines_raw))

        # identity maps (orig -> display)
        left_total  = int(self.left.text.index("end-1c").split(".")[0] or 1)
        right_total = int(self.right.text.index("end-1c").split(".")[0] or 1)
        self._left_map_display  = {i: i for i in range(1, left_total + 1)}
        self._right_map_display = {i: i for i in range(1, right_total + 1)}

        # clear tags
        for tagname in ("rep_line", "char_del", "char_add", "char_rep", "cursor_line"):
            self.left.text.tag_remove(tagname, "1.0", "end")
            self.right.text.tag_remove(tagname, "1.0", "end")

        # apply line-level diff tags
        for tag, i1, i2, j1, j2 in self._opcodes:
            if tag == "equal":
                continue
            for li in range(i1, i2):
                self.left.text.tag_add("rep_line", f"{li+1}.0", f"{li+1}.end")
            for rj in range(j1, j2):
                self.right.text.tag_add("rep_line", f"{rj+1}.0", f"{rj+1}.end")

        self.left._stripe_rows(); self.right._stripe_rows()
        self._rebuild_nav_targets()
        self._paint_char_diffs_for_current_view()

    def _apply_view_mode(self):
        if not self._opcodes:
            self._left_map_display, self._right_map_display = {}, {}
            self._nav_targets, self._nav_index = [], -1
            self._update_counter()
            return

        mode = self.view_mode.get()
        if mode == "all":
            # IMPORTANT FIX: actually restore full panes
            self._render_full_view()
            self._paint_char_diffs_for_current_view()
            return

        # Build filtered views and mapping orig -> display index
        left_out, right_out = [], []
        left_tags, right_tags = [], []
        left_orig_indices, right_orig_indices = [], []
        lcur = rcur = 1

        for tag, i1, i2, j1, j2 in self._opcodes:
            eq = (tag == "equal")
            include = (mode == "same" and eq) or (mode == "diff" and not eq)
            if not include: continue

            for li in range(i1, i2):
                left_out.append(self._left_lines_raw[li])
                left_orig_indices.append(li + 1)
                if not eq: left_tags.append(lcur)
                lcur += 1

            for rj in range(j1, j2):
                right_out.append(self._right_lines_raw[rj])
                right_orig_indices.append(rj + 1)
                if not eq: right_tags.append(rcur)
                rcur += 1

        # render filtered
        self.left.set_text("".join(left_out))
        self.right.set_text("".join(right_out))

        for tagname in ("rep_line", "char_del", "char_add", "char_rep", "cursor_line"):
            self.left.text.tag_remove(tagname, "1.0", "end")
            self.right.text.tag_remove(tagname, "1.0", "end")
        for ln in left_tags:
            self.left.text.tag_add("rep_line", f"{ln}.0", f"{ln}.end")
        for ln in right_tags:
            self.right.text.tag_add("rep_line", f"{ln}.0", f"{ln}.end")
        self.left._stripe_rows(); self.right._stripe_rows()

        self._left_map_display  = {orig: idx+1 for idx, orig in enumerate(left_orig_indices)}
        self._right_map_display = {orig: idx+1 for idx, orig in enumerate(right_orig_indices)}
        self._rebuild_nav_targets()
        self._paint_char_diffs_for_current_view()

    # -------- char-level colored diffs for current view --------
    def _paint_char_diffs_for_current_view(self):
        # clear old char tags
        for tagname in ("char_del", "char_add", "char_rep"):
            self.left.text.tag_remove(tagname, "1.0", "end")
            self.right.text.tag_remove(tagname, "1.0", "end")

        def map_left(orig):  return self._left_map_display.get(orig)
        def map_right(orig): return self._right_map_display.get(orig)

        for tag, i1, i2, j1, j2 in self._opcodes:
            if tag != "replace":
                continue  # only 'replace' has both lines to compare char-by-char

            pairs = min(i2 - i1, j2 - j1)
            for k in range(pairs):
                left_orig  = i1 + k + 1
                right_orig = j1 + k + 1

                ldisp = map_left(left_orig)
                rdisp = map_right(right_orig)
                if ldisp is None or rdisp is None:
                    continue  # not visible in current view

                Ls = self._left_lines_raw[i1 + k].rstrip("\n")
                Rs = self._right_lines_raw[j1 + k].rstrip("\n")
                for t, a1, a2, b1, b2 in difflib.SequenceMatcher(None, Ls, Rs).get_opcodes():
                    if t == "equal":
                        continue
                    if t == "delete":
                        self.left.text.tag_add("char_del", f"{ldisp}.{a1}", f"{ldisp}.{a2}")
                    elif t == "insert":
                        self.right.text.tag_add("char_add", f"{rdisp}.{b1}", f"{rdisp}.{b2}")
                    else:  # replace
                        self.left.text.tag_add("char_rep", f"{ldisp}.{a1}", f"{ldisp}.{a2}")
                        self.right.text.tag_add("char_rep", f"{rdisp}.{b1}", f"{rdisp}.{b2}")

    # ---------------- Navigation (Prev/Next Change) ----------------
    def _rebuild_nav_targets(self):
        targets = []
        for tag, i1, i2, j1, j2 in self._opcodes:
            if tag == "equal": continue
            left_orig  = (i1 + 1) if i1 < i2 else None
            right_orig = (j1 + 1) if j1 < j2 else None
            left_disp  = self._left_map_display.get(left_orig) if left_orig else None
            right_disp = self._right_map_display.get(right_orig) if right_orig else None
            if left_disp is None and right_disp is None:
                continue
            targets.append((left_disp, right_disp))

        self._nav_targets = targets
        self._nav_index = -1
        self.left.text.tag_remove("cursor_line", "1.0", "end")
        self.right.text.tag_remove("cursor_line", "1.0", "end")

        self._update_counter()
        if not targets:
            self._set_status("No differences.")
        else:
            self._set_status(f"{len(targets)} change(s). Use Next/Prev.")

    def _scroll_to_lines(self, left_line, right_line):
        self.left.text.tag_remove("cursor_line", "1.0", "end")
        self.right.text.tag_remove("cursor_line", "1.0", "end")

        if left_line is not None:
            total = max(1, int(self.left.text.index("end-1c").split(".")[0] or 1))
            self.left.yview_moveto((max(1, left_line)-1) / total)
            self.left.text.tag_add("cursor_line", f"{left_line}.0", f"{left_line}.end")

        if right_line is not None:
            total = max(1, int(self.right.text.index("end-1c").split(".")[0] or 1))
            self.right.yview_moveto((max(1, right_line)-1) / total)
            self.right.text.tag_add("cursor_line", f"{right_line}.0", f"{right_line}.end")

    def next_change(self):
        if not self._nav_targets:
            self._set_status("No differences.")
            return
        self._nav_index = (self._nav_index + 1) % len(self._nav_targets)
        L, R = self._nav_targets[self._nav_index]
        self._scroll_to_lines(L, R)
        self._update_counter()
        self._set_status(f"Change {self._nav_index+1}/{len(self._nav_targets)}")

    def prev_change(self):
        if not self._nav_targets:
            self._set_status("No differences.")
            return
        self._nav_index = (self._nav_index - 1) % len(self._nav_targets)
        L, R = self._nav_targets[self._nav_index]
        self._scroll_to_lines(L, R)
        self._update_counter()
        self._set_status(f"Change {self._nav_index+1}/{len(self._nav_targets)}")

    def _update_counter(self):
        total = len(self._nav_targets)
        idx = 0 if self._nav_index < 0 else (self._nav_index + 1)
        self.nav_counter_var.set(f"{idx} / {total}")

    def _set_status(self, msg: str):
        self.status.set(msg)

    # ---------------- Data for exporter ----------------
    def _collect_view_rows_for_export(self):
        # If no diff computed yet, dump panes
        if not self._opcodes and (self.left.get_text() or self.right.get_text()):
            L = self.left.get_text().splitlines(keepends=True)
            R = self.right.get_text().splitlines(keepends=True)
            left_rows  = [(i+1, L[i].rstrip("\n"), False) for i in range(len(L))]
            right_rows = [(i+1, R[i].rstrip("\n"), False) for i in range(len(R))]
            return left_rows, right_rows

        mode = self.view_mode.get()
        if mode == "all":
            L = [(i+1, self._left_lines_raw[i].rstrip("\n"), False) for i in range(len(self._left_lines_raw))]
            R = [(i+1, self._right_lines_raw[i].rstrip("\n"), False) for i in range(len(self._right_lines_raw))]
            for tag, i1, i2, j1, j2 in self._opcodes:
                if tag == "equal": continue
                for li in range(i1, i2):
                    if li < len(L): L[li] = (L[li][0], L[li][1], True)
                for rj in range(j1, j2):
                    if rj < len(R): R[rj] = (R[rj][0], R[rj][1], True)
            return L, R

        # filtered
        left_out, right_out = [], []
        lcur = rcur = 1
        for tag, i1, i2, j1, j2 in self._opcodes:
            eq = (tag == "equal")
            include = (mode == "same" and eq) or (mode == "diff" and not eq)
            if not include: continue
            for li in range(i1, i2):
                left_out.append((lcur, self._left_lines_raw[li].rstrip("\n"), not eq)); lcur += 1
            for rj in range(j1, j2):
                right_out.append((rcur, self._right_lines_raw[rj].rstrip("\n"), not eq)); rcur += 1
        return left_out, right_out


# ======================================================================
#  Run
# ======================================================================
def main():
    app = TextCompareApp()
    app.mainloop()

if __name__ == "__main__":
    main()
