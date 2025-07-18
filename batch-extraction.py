import os
import json
import logging
import sys
from pathlib import Path
import importlib.util

# Import the extract_text_from_image function from extract-text-cu.py
spec = importlib.util.spec_from_file_location("extract_text_cu", "extract-text-cu.py")
extract_text_cu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract_text_cu)
extract_text_from_image = extract_text_cu.extract_text_from_image

def batch_extract_text(images_directory: str = "images", output_file: str = "output.txt"):
    """
    Process all images in the specified directory and extract text from each.
    
    Args:
        images_directory: Path to the directory containing images
        output_file: Path to the output text file
    """
    logging.basicConfig(level=logging.INFO)
    
    # Check if images directory exists
    if not os.path.exists(images_directory):
        print(f"Images directory '{images_directory}' not found!")
        return
    
    # Get all image files from the directory
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for file in os.listdir(images_directory):
        if Path(file).suffix.lower() in image_extensions:
            image_files.append(os.path.join(images_directory, file))
    
    if not image_files:
        print(f"No image files found in '{images_directory}' directory!")
        return
    
    # Sort files for consistent processing order
    image_files.sort()
    
    print(f"Found {len(image_files)} image files to process...")
    
    # Initialize counters and prepare output file
    processed_count = 0
    failed_count = 0
    
    # Create/clear the output file at the start
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"BATCH TEXT EXTRACTION STARTED\n")
            f.write(f"Total files to process: {len(image_files)}\n")
            f.write(f"{'='*80}\n\n")
        print(f"Output file '{output_file}' created and ready for writing...")
    except Exception as e:
        print(f"Error creating output file: {e}")
        return
    
    # Process each image and write results immediately
    for i, image_file in enumerate(image_files, 1):
        try:
            print(f"Processing ({i}/{len(image_files)}): {image_file}")
            result_json = extract_text_from_image(image_file)
            
            # Parse the JSON result
            if isinstance(result_json, str):
                result = json.loads(result_json)
            else:
                result = result_json
            
            # Extract the markdown text
            markdown_text = result['result']['contents'][0]['markdown']
            
            # Prepare the text block for this image
            filename = os.path.basename(image_file)
            text_block = []
            text_block.append(f"{'='*60}")
            text_block.append(f"FILE: {filename}")
            text_block.append(f"PROCESSED: {i}/{len(image_files)}")
            text_block.append(f"{'='*60}")
            text_block.append("")
            text_block.append(markdown_text)
            text_block.append("")
            text_block.append("")
            
            # Write to output file immediately
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write('\n'.join(text_block))
            
            processed_count += 1
            print(f"✓ Successfully processed and saved: {filename}")
            
        except Exception as e:
            failed_count += 1
            filename = os.path.basename(image_file)
            error_msg = f"✗ Failed to process {filename}: {str(e)}"
            print(error_msg)
            logging.error(error_msg)
            
            # Write error info to output file immediately
            error_block = []
            error_block.append(f"{'='*60}")
            error_block.append(f"FILE: {filename}")
            error_block.append(f"PROCESSED: {i}/{len(image_files)}")
            error_block.append(f"{'='*60}")
            error_block.append("")
            error_block.append(f"ERROR: Failed to extract text - {str(e)}")
            error_block.append("")
            error_block.append("")
            
            try:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write('\n'.join(error_block))
            except Exception as write_error:
                print(f"Error writing error info to file: {write_error}")
    
    # Write final summary to output file
    try:
        summary_block = []
        summary_block.append(f"{'='*80}")
        summary_block.append(f"BATCH EXTRACTION COMPLETE")
        summary_block.append(f"{'='*80}")
        summary_block.append(f"Total files processed successfully: {processed_count}")
        summary_block.append(f"Total files failed: {failed_count}")
        summary_block.append(f"Total files attempted: {len(image_files)}")
        summary_block.append(f"{'='*80}")
        
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write('\n'.join(summary_block))
        
        print(f"\n{'='*60}")
        print(f"BATCH EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Total files processed: {processed_count}")
        print(f"Total files failed: {failed_count}")
        print(f"Output saved to: {output_file}")
        
    except Exception as e:
        print(f"Error writing summary to output file: {e}")

def main():
    """Main function to run the batch extraction."""
    print("Starting batch text extraction...")
    batch_extract_text()

if __name__ == "__main__":
    main()