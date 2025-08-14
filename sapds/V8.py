import os
import re
import pandas as pd
import xml.etree.ElementTree as ET
from collections import defaultdict

def parse_sapds_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    ns = {'ns': root.tag.split('}')[0].strip('{')}

    # Prepare list for final output
    output_rows = []

    # Map of external functions (lookup/lookup_ext) definitions
    external_lookups = {}

    # Project name mapping
    project_name_map = {}
    for proj in root.findall(".//ns:DIProject", ns):
        pid = proj.attrib.get("Id")
        pname = proj.attrib.get("Name")
        if pid and pname:
            project_name_map[pid] = pname

    # Capture all lookup_ext / lookup definitions (even outside dataflow)
    for func in root.findall(".//ns:DIUserDefinedFunction", ns):
        fname = func.attrib.get("Name", "")
        if fname.lower().startswith("lookup") or fname.lower().startswith("lookup_ext"):
            external_lookups[fname] = {
                'datastore': func.findtext(".//ns:DataStoreName", default="", namespaces=ns),
                'schema': func.findtext(".//ns:SchemaName", default="", namespaces=ns),
                'table': func.findtext(".//ns:TableName", default="", namespaces=ns)
            }

    # Parse all dataflows
    for df in root.findall(".//ns:DIDataflow", ns):
        df_name = df.attrib.get("Name", "")
        project_id = df.attrib.get("ProjectId", "")
        project_name = project_name_map.get(project_id, "")

        job_name = ""
        job_elem = df.find("../..")  # Navigate up to job
        if job_elem is not None:
            job_name = job_elem.attrib.get("Name", "")

        # Transformation-level lookup tracking
        transformation_usage = defaultdict(set)

        # Lookups directly in dataflow
        for func_call in df.findall(".//ns:FunctionCall", ns):
            fname = func_call.attrib.get("Name", "")
            if fname.lower().startswith("lookup") or fname.lower().startswith("lookup_ext"):
                datastore = func_call.findtext(".//ns:DataStoreName", default="", namespaces=ns)
                schema = func_call.findtext(".//ns:SchemaName", default="", namespaces=ns)
                table = func_call.findtext(".//ns:TableName", default="", namespaces=ns)

                # If blank, try from external definitions
                if not (datastore or schema or table):
                    if fname in external_lookups:
                        datastore = external_lookups[fname]['datastore']
                        schema = external_lookups[fname]['schema']
                        table = external_lookups[fname]['table']

                pos = func_call.attrib.get("Position", "")
                transformation_usage[(fname, datastore, schema, table)].add(pos)

        # Custom SQL parsing
        for trans in df.findall(".//ns:DITransformation", ns):
            sql_text = trans.findtext(".//ns:CustomSQL", default="", namespaces=ns)
            if sql_text:
                tables_in_sql = re.findall(r'\b(?:from|join)\s+([^\s;]+)', sql_text, re.I)
                schema = "CUSTOM_SQL"
                table = ",".join(sorted(set(tables_in_sql)))
                fname = trans.attrib.get("Name", "")
                pos = "N/A"
                transformation_usage[(fname, "", schema, table)].add(pos)
                output_rows.append({
                    "Project_Name": project_name,
                    "Job_Name": job_name,
                    "Dataflow_Name": df_name,
                    "DataStore_Name": "",
                    "Schema_Name": schema,
                    "Table_Name": table,
                    "Transformation_Name": fname,
                    "Transformation_Position": pos,
                    "Transformation_Usages_Count": 1,
                    "Custom_SQL": f"\"{sql_text.strip()}\""
                })

        # Append lookup rows
        for (fname, datastore, schema, table), pos_set in transformation_usage.items():
            output_rows.append({
                "Project_Name": project_name,
                "Job_Name": job_name,
                "Dataflow_Name": df_name,
                "DataStore_Name": datastore,
                "Schema_Name": schema,
                "Table_Name": table,
                "Transformation_Name": fname,
                "Transformation_Position": ",".join(sorted(pos_set)),
                "Transformation_Usages_Count": len(pos_set),
                "Custom_SQL": ""
            })

    return pd.DataFrame(output_rows)


if __name__ == "__main__":
    xml_file = "input.xml"  # change to your file
    df = parse_sapds_xml(xml_file)
    df.drop_duplicates(inplace=True)
    output_file = os.path.splitext(xml_file)[0] + "_parsed.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Saved output to {output_file}")
