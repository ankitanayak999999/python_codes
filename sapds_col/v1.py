import xml.etree.ElementTree as ET
import pandas as pd
import os

def extract_column_mappings(xml_file, project_name, job_name, dataflow_name, transformation_name, output_csv):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    records = []

    # Look for DIElement + DIAttribute(ui_mapping_text)
    for elem in root.findall(".//DIElement"):
        column_name = elem.attrib.get("name")
        mapping_text = None

        for attr in elem.findall("./DIAttributes/DIAttribute[@name='ui_mapping_text']"):
            mapping_text = attr.attrib.get("value")

        if column_name and mapping_text:
            records.append({
                "PROJECT_NAME": project_name,
                "JOB_NAME": job_name,
                "DATAFLOW_NAME": dataflow_name,
                "TRANSFORMATION_NAME": transformation_name,
                "COLUMN_NAME": column_name,
                "MAPPING_TEXT": mapping_text
            })

    # Save to CSV
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"✅ Extracted {len(records)} mappings to {output_csv}")


# Example usage
if __name__ == "__main__":
    xml_file = r"C:\Users\raksahu\Downloads\python\input\sap_ds_xml_files\New folder\export_df.xml"
    output_csv = "column_mappings.csv"

    # Pass identifiers (you can later make these dynamic if needed)
    extract_column_mappings(
        xml_file,
        project_name="PROJ_MIS",
        job_name="JOB_STG",
        dataflow_name="DF_STG_E",
        transformation_name="Sort",
        output_csv=output_csv
    )
