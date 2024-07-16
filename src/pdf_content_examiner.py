# pdf_content_examiner.py

import os
from pypdf import PdfReader

def examine_pdf(file_path):
    print(f"Examining PDF: {file_path}")
    try:
        with open(file_path, 'rb') as file:
            pdf = PdfReader(file)
            for i, page in enumerate(pdf.pages):
                print(f"\nPage {i+1} content:")
                text = page.extract_text()
                print(text[:500] + "..." if len(text) > 500 else text)
                print("-" * 50)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main():
    base_path = "B:\\Angelo Data\\GitHub\\Quotation_Project_HACV\\data"
    manufacturers = ["daikin", "melco"]

    for manufacturer in manufacturers:
        manufacturer_path = os.path.join(base_path, manufacturer)
        for filename in os.listdir(manufacturer_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(manufacturer_path, filename)
                examine_pdf(file_path)

if __name__ == "__main__":
    main()