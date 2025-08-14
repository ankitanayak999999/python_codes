import xml.etree.ElementTree as ET

def parse_xml_robust(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        for object in root.findall('.//Object'):
            object_name = object.attrib.get('name')
            if object_name:
                print(f"Processing object: {object_name}")

            # Check if 'Source' and 'Target' elements exist
            sources = object.findall('Source')
            targets = object.findall('Target')

            if sources:
                for source in sources:
                    source_name = source.attrib.get('name', 'N/A')
                    print(f"  Source: {source_name}")
            else:
                print("  No source found for this object.")

            if targets:
                for target in targets:
                    target_name = target.attrib.get('name', 'N/A')
                    print(f"  Target: {target_name}")
            else:
                print("  No target found for this object.")
            
    except ET.ParseError as e:
        print(f"Error parsing XML file: {file_path}. Details: {e}")
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
parse_xml_robust("your_sap_ds_job.xml")
