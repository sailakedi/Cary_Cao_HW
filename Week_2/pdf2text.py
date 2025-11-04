"""
import tempfile
from pdf2image import convert_from_path

with tempfile.TemporaryDirectory() as path:
    images_from_path = convert_from_path("/users/jiancao/Downloads/Algorithms Fourth Edition.pdf", 
                                        first_page = 0, 
                                        last_page = 10,
                                        output_folder=path)

"""

from pdf2image import convert_from_path
import pytesseract
import os

def extract_text_from_pdf(pdf_path, 
                          first_page=0, 
                          last_page=10,
                          output_image_path="output_images",
                          output_text_path=None,
                          dpi=300,
                          lang='eng'):
    
    os.makedirs(output_image_path, exist_ok=True)
          
    # Convert PDF to images
    images = convert_from_path(pdf_path, first_page=first_page, last_page=last_page, dpi=dpi)
    print(f"Converted {len(images)} pages to images.")
    
    saved_images = []
    text = ""

    for i, image in enumerate(images):
        image_file = os.path.join(output_image_path, f"page_{(first_page + i + 1):03}.jpeg")
        image.save(image_file, 'JPEG')
        saved_images.append(image_file)

        # Use pytesseract to do OCR on the image
        page_text = pytesseract.image_to_string(image, lang=lang)
        text += page_text
        text += f"\n--- Page {first_page + i + 1} ---\n\n\n"

    if output_image_path:
        print(f"Saved {len(saved_images)} image files.")

    if output_text_path:
        with open(output_text_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Extracted text saved to {output_text_path}")
    return text

if __name__ == "__main__":
    pdf_path         = "/users/jiancao/Downloads/Algorithms Fourth Edition.pdf"
    output_image_path = "pdf_images"
    output_text_path = "/users/jiancao/Downloads/Algorithms Fourth Edition_text.txt"
    text = extract_text_from_pdf(pdf_path, 
                                 first_page=0, 
                                 last_page=10, 
                                 output_image_path=output_image_path, 
                                 output_text_path=output_text_path)
    #print("\n=== Extracted Text Preview ===\n")
    #print(text[:2000])

    