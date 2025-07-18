import os
import sys
import io
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import argparse


def pdf_to_images(pdf_path, output_folder="images", dpi=300):
    """
    Convert a multi-page PDF to PNG images.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_folder (str): Output folder for images (default: "images")
        dpi (int): Resolution for the output images (default: 300)
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        
        # Get the base filename without extension
        pdf_name = Path(pdf_path).stem
        
        print(f"Converting {pdf_path} to images...")
        print(f"Total pages: {len(pdf_document)}")
        
        # Convert each page to image
        for page_num in range(len(pdf_document)):
            # Get the page
            page = pdf_document[page_num]
            
            # Create a transformation matrix for the desired DPI
            # Default is 72 DPI, so we scale by dpi/72
            mat = fitz.Matrix(dpi/72, dpi/72)
            
            # Render page to an image
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Save the image
            output_filename = f"{pdf_name}_page_{page_num + 1:03d}.png"
            output_filepath = output_path / output_filename
            
            img.save(output_filepath, "PNG")
            print(f"Saved: {output_filepath}")
        
        pdf_document.close()
        print(f"\nConversion complete! {len(pdf_document)} pages converted to {output_folder}/")
        
    except FileNotFoundError:
        print(f"Error: PDF file '{pdf_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error converting PDF: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert PDF to PNG images")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", "-o", default="images", 
                       help="Output folder for images (default: images)")
    parser.add_argument("--dpi", "-d", type=int, default=300,
                       help="DPI for output images (default: 300)")
    
    args = parser.parse_args()
    
    # Check if PDF file exists
    if not Path(args.pdf_path).exists():
        print(f"Error: PDF file '{args.pdf_path}' does not exist.")
        sys.exit(1)
    
    # Convert PDF to images
    pdf_to_images(args.pdf_path, args.output, args.dpi)


if __name__ == "__main__":
    main()