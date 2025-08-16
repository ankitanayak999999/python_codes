import os

def write_diff_html(diff_blocks_html: str, output_html_path: str, template_filename: str = "diff_template.html"):
    """
    Loads the external HTML template, injects diff content, and writes the final report.

    Parameters
    ----------
    diff_blocks_html : str
        The concatenated HTML for all diff sections (your generated <div class="diff-container">...</div> blocks).
    output_html_path : str
        Full path for the final HTML report to be written.
    template_filename : str
        The HTML template filename (default: 'diff_template.html') located in the same folder as this script.
    """
    # Find the template in the same folder as this Python file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, template_filename)

    # Read the template
    with open(template_path, "r", encoding="utf-8") as f:
        template_html = f.read()

    # Inject your diff content
    final_html = template_html.replace("{{ diff_content }}", diff_blocks_html)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)

    # Write the final report
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(final_html)


def generate_sql_diff_report(file1_path, file2_path, output_dir):
    # ... your logic to read files, diff them, and build `diff_content` ...

    # Example of one block you might append while looping pairs:
    # diff_content += f'''
    #   <div class="diff-container diff-added">
    #     <div class="diff-header">Object: XYZ</div>
    #     <h3>File 1</h3><pre>{escaped_sql1}</pre>
    #     <h3>File 2</h3><pre>{escaped_sql2}</pre>
    #   </div>
    # '''

    output_html = os.path.join(output_dir, "sql_diff_report.html")
    write_diff_html(diff_blocks_html=diff_content, output_html_path=output_html)
    return output_html
