import requests
import xml.etree.ElementTree as ET
from pathlib import Path
import json

#first run docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.2 as sudo
class GrobidClient:
    def __init__(self, grobid_server="http://localhost:8070"):
        self.base_url = grobid_server
        self.session = requests.Session()

    def process_pdf(self, pdf_path, output_format="xml"):
        """
        Process PDF with GROBID and return structured data

        Args:
            pdf_path: Path to PDF file
            output_format: 'xml', 'json', or 'text'
        """
        url = f"{self.base_url}/api/processFulltextDocument"

        with open(pdf_path, 'rb') as pdf_file:
            files = {'input': pdf_file}

            response = self.session.post(url, files=files)
            response.raise_for_status()

            if output_format == "xml":
                return response.text
            elif output_format == "json":
                return self.xml_to_json(response.text)
            elif output_format == "text":
                return self.xml_to_text(response.text)

    # def xml_to_text(self, xml_content):
    #     """Extract plain text from GROBID XML output"""
    #     try:
    #         root = ET.fromstring(xml_content)
    #
    #         # Define namespaces used in GROBID XML
    #         ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
    #
    #         extracted_text = {}
    #
    #         # Extract title
    #         title_elem = root.find('.//tei:titleStmt/tei:title', ns)
    #         extracted_text['title'] = title_elem.text if title_elem is not None else ""
    #
    #         # Extract abstract
    #         abstract_elem = root.find('.//tei:abstract', ns)
    #         if abstract_elem is not None:
    #             abstract_text = ""
    #             for p in abstract_elem.findall('.//tei:p', ns):
    #                 if p.text:
    #                     abstract_text += p.text + " "
    #             extracted_text['abstract'] = abstract_text.strip()
    #
    #         # Extract authors
    #         authors = []
    #         for author in root.findall('.//tei:author', ns):
    #             first_name = author.find('.//tei:forename', ns)
    #             last_name = author.find('.//tei:surname', ns)
    #             if first_name is not None and last_name is not None:
    #                 authors.append(f"{first_name.text} {last_name.text}")
    #         extracted_text['authors'] = authors
    #
    #         # Extract main body text
    #         body_text = ""
    #         body = root.find('.//tei:body', ns)
    #         if body is not None:
    #             for div in body.findall('.//tei:div', ns):
    #                 # Extract section headers
    #                 head = div.find('./tei:head', ns)
    #                 if head is not None and head.text:
    #                     body_text += f"\n\n{head.text}\n"
    #
    #                 # Extract paragraphs
    #                 for p in div.findall('.//tei:p', ns):
    #                     if p.text:
    #                         body_text += p.text + " "
    #                     for child in p:
    #                         if child.text:
    #                             body_text += child.text + " "
    #                         if child.tail:
    #                             body_text += child.tail + " "
    #                     body_text += "\n"
    #
    #         extracted_text['body'] = body_text.strip()
    #
    #         return extracted_text
    #
    #     except ET.ParseError as e:
    #         print(f"Error parsing XML: {e}")
    #         return {"error": str(e)}
    #
    # def xml_to_json(self, xml_content):
    #     """Convert XML to JSON structure"""
    #     text_data = self.xml_to_text(xml_content)
    #     return json.dumps(text_data, indent=2)
    #
    # def process_folder(self, folder_path, output_dir=None):
    #     """Process all PDFs in a folder"""
    #     folder = Path(folder_path)
    #     output_dir = Path(output_dir) if output_dir else folder
    #
    #     results = []
    #     for pdf_file in folder.glob("*.pdf"):
    #         print(f"Processing {pdf_file.name}...")
    #         try:
    #             result = self.process_pdf(pdf_file, output_format="text")
    #             results.append({
    #                 'filename': pdf_file.name,
    #                 'data': result
    #             })
    #
    #             # Save individual results
    #             output_file = output_dir / f"{pdf_file.stem}_extracted.json"
    #             with open(output_file, 'w', encoding='utf-8') as f:
    #                 json.dump(result, f, indent=2, ensure_ascii=False)
    #
    #         except Exception as e:
    #             print(f"Error processing {pdf_file.name}: {e}")
    #             results.append({
    #                 'filename': pdf_file.name,
    #                 'error': str(e)
    #             })
    #
    #     return results


# Usage examples - XML output focused
def main():
    # Initialize GROBID client
    client = GrobidClient()

    # Process single PDF and save XML
    pdf_path = "/home/juanho-liang/PY/myagent/PDFtestG/EANO-ESMO Clinical Practice Guideline Leptomeningeal Metastasis.pdf"
    try:
        xml_result = client.process_pdf(pdf_path, output_format="xml")

        # Save XML file
        xml_file = pdf_path.replace('.pdf', '_grobid.xml')
        with open(xml_file, 'w', encoding='utf-8') as f:
            f.write(xml_result)
        print(f"GROBID XML saved to: {xml_file}")

        # Display XML structure preview
        print("\nXML structure preview:")
        print(xml_result[:1000] + "..." if len(xml_result) > 1000 else xml_result)

    except Exception as e:
        print(f"Error: {e}")


def process_folder_to_xml(folder_path, output_dir="./grobid_xml"):
    """Process all PDFs in a folder and save as XML files"""
    import os
    from pathlib import Path

    client = GrobidClient()
    folder = Path(folder_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    results = []
    for pdf_file in folder.glob("*.pdf"):
        print(f"Processing {pdf_file.name}...")
        try:
            xml_result = client.process_pdf(pdf_file, output_format="xml")

            # Save XML file
            xml_file = output_dir / f"{pdf_file.stem}_grobid.xml"
            with open(xml_file, 'w', encoding='utf-8') as f:
                f.write(xml_result)

            results.append({
                'pdf_file': str(pdf_file),
                'xml_file': str(xml_file),
                'status': 'success'
            })
            print(f"  → Saved: {xml_file}")

        except Exception as e:
            print(f"  → Error processing {pdf_file.name}: {e}")
            results.append({
                'pdf_file': str(pdf_file),
                'error': str(e),
                'status': 'failed'
            })

    # Save processing log
    log_file = output_dir / "processing_log.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nProcessing complete. XML files saved to: {output_dir}")
    print(f"Processing log saved to: {log_file}")

    return results

if __name__ == "__main__":
    # Process single PDF to XML
    # main()

    # Process entire folder to XML
    print("\n" + "=" * 50)
    print("Batch processing example:")

    # Uncomment and modify paths as needed:
    process_folder_to_xml("/home/juanho-liang/PY2/hema_agent/Guidelines_processed", "./xml_output_hemaguide")

    print("\nTo use:")
    print("1. Replace 'paper.pdf' with your PDF path")
    print("2. For batch processing, uncomment the folder processing line")
    print("3. XML files will be saved with '_grobid.xml' suffix")
    print("4. Use the XML files with your own chunking strategy")