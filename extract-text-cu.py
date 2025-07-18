import logging
import json
import os
import sys
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from azure.core.credentials import AzureKeyCredential
from PIL import Image
from io import BytesIO

def extract_text_from_image(file_path: str) -> str:
    """
    Extract text from an image file using Azure AI Content Understanding.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        JSON string containing the extracted text and analysis results
    """
    load_dotenv(find_dotenv())
    logging.basicConfig(level=logging.INFO)

    AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT")
    AZURE_AI_API_VERSION = os.getenv("AZURE_AI_API_VERSION", "2025-05-01-preview")
    AZURE_AI_KEY = os.getenv("AZURE_AI_KEY")

    if not AZURE_AI_ENDPOINT:
        raise ValueError("AZURE_AI_ENDPOINT environment variable is not set")
    if not AZURE_AI_KEY:
        raise ValueError("AZURE_AI_KEY environment variable is not set")

    # Add the specific directory to the path to use shared modules
    azure_ai_dir = Path(r"C:\Users\mboutilier\Documents\azure-ai-content-understanding-python")
    sys.path.append(str(azure_ai_dir))
    
    try:
        from python.content_understanding_client import AzureContentUnderstandingClient
    except ImportError as e:
        raise ImportError(f"Failed to import AzureContentUnderstandingClient: {e}")

    client = AzureContentUnderstandingClient(
        endpoint=AZURE_AI_ENDPOINT,
        api_version=AZURE_AI_API_VERSION,
        subscription_key=AZURE_AI_KEY,
        x_ms_useragent="azure-ai-content-understanding-python/content_extraction",
    )

    def save_image(image_id: str, response):
        """Save an image from the analysis response to cache directory."""
        try:
            raw_image = client.get_image_from_analyze_operation(
                analyze_response=response,
                image_id=image_id
            )
            image = Image.open(BytesIO(raw_image))
            Path(".cache").mkdir(exist_ok=True)
            image.save(f".cache/{image_id}.jpg", "JPEG")
        except Exception as e:
            logging.error(f"Failed to save image {image_id}: {e}")

    ANALYZER_ID = 'prebuilt-documentAnalyzer'

    try:
        # Analyze the file
        response = client.begin_analyze(ANALYZER_ID, file_location=file_path)
        result_json = client.poll_result(response)
        return json.dumps(result_json, indent=2)
    except Exception as e:
        logging.error(f"Failed to analyze file {file_path}: {e}")
        raise

def main():
    """Main function to process command line arguments and extract text from image."""
    if len(sys.argv) < 2:
        print("Usage: python extract-text.py <path_to_image>")
        print("Example: python extract-text.py images/mark-abenaki_page_001.png")
        sys.exit(1)

    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    try:
        result = extract_text_from_image(file_path)

        # If `result` is a string, parse it to a dict
        if isinstance(result, str):
            result = json.loads(result)

        markdown_text = result['result']['contents'][0]['markdown']
        print(markdown_text)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()