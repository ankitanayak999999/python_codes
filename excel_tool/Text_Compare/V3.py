from pathlib import Path
import difflib
import datetime
from typing import Optional, Tuple, List

# ---------- small helpers ----------

def _read_lines(p: Path) -> List[str]:
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        return fh.read().splitlines()

def _load_html_template(template_file: Optional[str]) -> str:
    """Load template from explicit path, or from the same dir as this .py, or use a minimal fallback."""
    # 1) explicit path
    if template_file:
        tp = Path(template_file)
        if tp.is_file():
            return tp.read_text(encoding="utf-8", errors="replace")

    # 2) same directory as this file
    local = Path(__file__).parent / "text_compare_template.html"
    if local.is_file():
        return local.read_text(encoding="utf-8", errors="replace")

    # 3) minimal fallback (single placeholder)
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>{{TITLE}}</title>
<style>
  body{margin:0;font-family:Arial,sans-serif;background:#1e1e1e;color:#ddd}
  .toolbar{position:sticky;top:0;background:#333;padding:10px;text-align:center;z-index:100}
  .buttons button{margin:0 6px;padding:6px 12px;border-radius:6px;border:1px solid #666;background:#444;color:#ddd;cursor:pointer}
  .buttons button.active{background:#1976d2;border-color:#1976d2}
  .wrap{display:flex;gap:8px;padding:8px;height:calc(100vh - 64px);box-sizing:border-box}
  .panel{flex:1;display:flex;flex-direction:column;border:1px solid #444;border-radius:8px;overflow:hidden}
  .panel h2{margin:0;background:#2a2a2a;padding:8px 12px;border-bottom:1px solid #444;font-size:14px;font-weight:600}
  .scroll{flex:1;overflow:auto;background:#111}
  table.diff{width:100%;border-collapse:collapse;font-family:Consolas,monospace;font-size:12px;line-height:1.35}
  table.diff td, table.diff th{border-bottom:1px solid #222;padding:2px 6px;white-space:pre}
  table.diff .diff_add{background:#114a2f} table.diff .diff_chg{background:#4a3411} table.diff .diff_sub{background:#4a1111}
  /* hide line# + gutter cols for compact view */
  table.diff th:nth-child(1), table.diff td:nth-child(1),
  table.diff th:nth-child(2), table.diff td:nth-child(2){display:none}
  /* visibility filters */
  .hide-same .same{display:none}
  .hide-diff .diff_add, .hide-diff .diff_chg, .hide-diff .diff_sub{display:none}
</style>
</head>
<body>
  <div class="toolbar">
    <div class="buttons">
      <button id="btnAll" class="active">Show All</button>
      <button id="btnDiff">Show Only Differences</button>
      <button id="btnSame">Show Only Same</button>
    </div>
    <div id="labels" style="margin-top:6px;font-size:12px;color:#bbb;">
      Left: {{LEFT_LABEL}} &nbsp;|&nbsp; Right: {{RIGHT_LABEL}}
    </div>
  </div>

  <div class="wrap" id="wrap">
    <div class="panel left">
      <h2>File 1</h2>
      <div class="scroll" id="leftPane">
        {{DIFF_TABLE}}
      </div>
    </div>
    <div class="panel right">
      <h2>File 2</h2>
      <div class="scroll" id="rightPane">
        {{DIFF_TABLE}}
      </div>
    </div>
  </div>

<script>
(function(){
  const wrap = document.getElementById('wrap');
  const btnAll  = document.getElementById('btnAll');
  const btnDiff = document.getElementById('btnDiff');
  const btnSame = document.getElementById('btnSame');
  const btns = [btnAll, btnDiff, btnSame];

  function setActive(b){
    btns.forEach(x=>x.classList.remove('active'));
    b.classList.add('active');
  }
  btnAll.addEventListener('click', ()=>{wrap.classList.remove('hide-same'); wrap.classList.remove('hide-diff'); setActive(btnAll);});
  btnDiff.addEventListener('click', ()=>{wrap.classList.add('hide-same'); wrap.classList.remove('hide-diff'); setActive(btnDiff);});
  btnSame.addEventListener('click', ()=>{wrap.classList.add('hide-diff'); wrap.classList.remove('hide-same'); setActive(btnSame);});

  // keep the two panes vertically in sync
  const left  = document.getElementById('leftPane');
  const right = document.getElementById('rightPane');
  let lock = false;
  left.addEventListener('scroll', ()=>{ if(lock) return; lock = true; right.scrollTop = left.scrollTop; lock=false;});
  right.addEventListener('scroll', ()=>{ if(lock) return; lock = true; left.scrollTop = right.scrollTop; lock=false;});
})();
</script>
</body></html>"""

# ---------- main function ----------

def compare_text_files(
    file1: str,
    file2: str,
    project_path: str,
    template_file: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Build side-by-side HTML diff and a unified diff text.
    Returns (html_path, unified_diff_text).
    """
    # paths
    out_dir = Path(project_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    f1 = Path(file1)
    f2 = Path(file2)

    # read
    a_lines = _read_lines(f1)
    b_lines = _read_lines(f2)

    # labels + output name
    from_desc = f1.name
    to_desc   = f2.name

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    html_path = out_dir / f"text_compare_result_{ts}.html"

    # build diff TABLE once
    html_maker = difflib.HtmlDiff(wrapcolumn=120)
    table_html = html_maker.make_table(a_lines, b_lines, fromdesc=from_desc, todesc=to_desc)

    # template handling
    tpl = _load_html_template(template_file)

    # replace placeholders (supports both old and new templates)
    final_html = (
        tpl.replace("{{TITLE}}", "Text Compare Result")
           .replace("{{LEFT_LABEL}}", from_desc)
           .replace("{{RIGHT_LABEL}}", to_desc)
    )
    if "{{LEFT_TABLE}}" in final_html and "{{RIGHT_TABLE}}" in final_html:
        final_html = (final_html
                      .replace("{{LEFT_TABLE}}", table_html)
                      .replace("{{RIGHT_TABLE}}", table_html))
    else:
        final_html = final_html.replace("{{DIFF_TABLE}}", table_html)

    # write HTML
    html_path.write_text(final_html, encoding="utf-8")

    # optional unified text diff (useful for logs)
    utext = "\n".join(
        difflib.unified_diff(
            a_lines, b_lines, fromfile=str(f1), tofile=str(f2), lineterm=""
        )
    )

    return str(html_path), utext


# ---------- example runner (safe to remove) ----------
if __name__ == "__main__":
    # demo paths — change to yours or wire into your tool
    demo_out = str(Path(__file__).parent)  # output in current folder
    f1 = r"C:\path\to\file_1.sql"
    f2 = r"C:\path\to\file_2.sql"

    html, ud = compare_text_files(f1, f2, demo_out)  # uses template next to this .py if present
    print("HTML written to:", html)
