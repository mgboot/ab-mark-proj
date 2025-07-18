"""
Azure OpenAI Image Processing Module

This module processes images using Azure OpenAI's vision-capable models.
It supports local image files and provides configurable system prompts.

Security: Uses API key authentication loaded from .env file.
Performance: Includes retry logic and proper error handling.
"""

import os
import base64
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import time
import json

try:
    from openai import AzureOpenAI
except ImportError:
    raise ImportError("Please install the openai package: pip install openai")

try:
    from dotenv import load_dotenv
except ImportError:
    raise ImportError("Please install python-dotenv: pip install python-dotenv")

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default system prompt for Abenaki Gospel text transcription
DEFAULT_SYSTEM_PROMPT = """
You are an expert at transcribing text from images EXACTLY as it appears on the page, especially pages containing text in multiple languages.

When shown a page from the Gospel of Mark with text in the Abenaki, English, and French languages, you will transcribe all the text that you see faithfully, including the verse number (which is generally at the beginning of the Abenaki portion).

The only thing you do not need to transcribe is a page number, which may or may not be at the very bottom of the page.

You've got this!
"""


class AzureOpenAIImageProcessor:
    """
    Azure OpenAI Image Processor with vision capabilities.
    
    Uses API key authentication and loads configuration from .env file.
    Implements retry logic and proper error handling.
    """
    
    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        api_version: str = "2024-02-01",
        deployment_name: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the Azure OpenAI Image Processor.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL
            api_version: API version to use
            deployment_name: Name of the deployed model
            api_key: API key for authentication
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        # Load configuration from environment variables (from .env file)
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version
        self.deployment_name = deployment_name or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if not self.azure_endpoint:
            raise ValueError("Azure OpenAI endpoint must be provided or set in AZURE_OPENAI_ENDPOINT environment variable")
        
        if not self.deployment_name:
            raise ValueError("Deployment name must be provided or set in AZURE_OPENAI_DEPLOYMENT_NAME environment variable")
        
        if not self.api_key:
            raise ValueError("API key must be provided or set in AZURE_OPENAI_API_KEY environment variable")
        
        # Initialize Azure OpenAI client with API key authentication
        self.client = self._initialize_client()
        
        logger.info(f"Initialized Azure OpenAI client for endpoint: {self.azure_endpoint}")
        logger.info(f"Using deployment: {self.deployment_name}")
    
    def _initialize_client(self) -> AzureOpenAI:
        """Initialize the Azure OpenAI client with API key authentication."""
        try:
            return AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                api_version=self.api_version
            )
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            raise
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode a local image file to base64 string.
        
        Args:
            image_path: Path to the local image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            image_file = Path(image_path)
            if not image_file.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            if not image_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
                raise ValueError(f"Unsupported image format: {image_file.suffix}")
            
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            logger.info(f"Successfully encoded image: {image_path}")
            return encoded_image
            
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {str(e)}")
            raise
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"All retry attempts failed: {str(e)}")
                    raise
                
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    
    def process_image(
        self,
        image_path: str,
        system_prompt: str = "",
        user_prompt: str = "Describe this image in detail.",
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Process an image using Azure OpenAI vision model.
        
        Args:
            image_path: Path to the local image file
            system_prompt: System prompt with instructions for the model
            user_prompt: User prompt describing the task
            max_tokens: Maximum tokens in the response
            temperature: Sampling temperature (0.0 to 1.0)
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            # Validate inputs
            if not image_path:
                raise ValueError("Image path cannot be empty")
            
            # Encode image to base64
            base64_image = self._encode_image_to_base64(image_path)
            
            # Prepare messages
            messages = []
            
            # Add system prompt if provided
            if system_prompt.strip():
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Add user message with image
            user_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
            messages.append(user_message)
            
            logger.info(f"Processing image: {image_path}")
            logger.info(f"System prompt length: {len(system_prompt)} characters")
            logger.info(f"User prompt: {user_prompt}")
            
            # Make API call with retry logic
            def make_api_call():
                return self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            
            response = self._retry_with_backoff(make_api_call)
            
            # Extract and return results
            result = {
                "success": True,
                "image_path": image_path,
                "content": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason
            }
            
            logger.info(f"Successfully processed image. Tokens used: {result['usage']['total_tokens']}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "image_path": image_path
            }


def process_dictionary_image(
    image_path: str,
    system_prompt: Optional[str] = None,
    user_prompt: str = "See image here. Do your thing.",
    max_tokens: int = 2000,
    temperature: float = 0.1,
    processor: Optional[AzureOpenAIImageProcessor] = None
) -> Dict[str, Any]:
    """
    Process a dictionary image and return formatted results.
    
    This function can be called from other scripts to process images en masse.
    
    Args:
        image_path: Path to the image file to process
        system_prompt: Custom system prompt (uses DEFAULT_SYSTEM_PROMPT if None)
        user_prompt: User prompt for the specific task
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        processor: Existing processor instance (creates new one if None)
        
    Returns:
        Dictionary containing success status, content, and metadata
    """
    
    # Use default system prompt if none provided
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    
    try:
        # Create processor if not provided
        if processor is None:
            processor = AzureOpenAIImageProcessor(max_retries=3)
        
        # Process the image
        result = processor.process_image(
            image_path=image_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to process dictionary image {image_path}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "image_path": image_path
        }


def process_multiple_images(
    image_paths: list[str],
    output_dir: Optional[str] = None,
    save_individual_results: bool = True,
    combine_results: bool = True
) -> Dict[str, Any]:
    """
    Process multiple dictionary images and optionally save results.
    
    Args:
        image_paths: List of image file paths to process
        output_dir: Directory to save results (creates 'results' dir if None)
        save_individual_results: Whether to save individual TSV files for each image
        combine_results: Whether to create a combined TSV file
        
    Returns:
        Dictionary with processing results and statistics
    """
    
    if output_dir is None:
        output_dir = "results"
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialize processor once for all images
    processor = AzureOpenAIImageProcessor(max_retries=3)
    
    results = []
    combined_tsv_content = []
    total_tokens = 0
    successful_count = 0
    failed_count = 0
    
    logger.info(f"Starting batch processing of {len(image_paths)} images")
    
    for i, image_path in enumerate(image_paths, 1):
        logger.info(f"Processing image {i}/{len(image_paths)}: {image_path}")
        
        try:
            # Process the image
            result = process_dictionary_image(
                image_path=image_path,
                processor=processor
            )
            
            results.append(result)
            
            if result["success"]:
                successful_count += 1
                total_tokens += result["usage"]["total_tokens"]
                
                # Save individual TSV file if requested
                if save_individual_results:
                    image_name = Path(image_path).stem
                    tsv_filename = f"{image_name}.tsv"
                    tsv_path = Path(output_dir) / tsv_filename
                    
                    with open(tsv_path, 'w', encoding='utf-8') as f:
                        f.write(result["content"])
                    
                    logger.info(f"Saved individual result to: {tsv_path}")
                
                # Add to combined results
                if combine_results:
                    # Add a comment line to identify the source image
                    combined_tsv_content.append(f"# Source: {image_path}")
                    combined_tsv_content.append(result["content"])
                    combined_tsv_content.append("")  # Empty line between images
                
            else:
                failed_count += 1
                logger.error(f"Failed to process {image_path}: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            failed_count += 1
            logger.error(f"Exception while processing {image_path}: {str(e)}")
            results.append({
                "success": False,
                "error": str(e),
                "image_path": image_path
            })
    
    # Save combined TSV file if requested
    if combine_results and combined_tsv_content:
        combined_path = Path(output_dir) / "combined_dictionary_entries.tsv"
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(combined_tsv_content))
        logger.info(f"Saved combined results to: {combined_path}")
    
    # Create summary
    summary = {
        "total_images": len(image_paths),
        "successful": successful_count,
        "failed": failed_count,
        "total_tokens_used": total_tokens,
        "output_directory": output_dir,
        "individual_results": results
    }
    
    logger.info(f"Batch processing complete: {successful_count} successful, {failed_count} failed, {total_tokens} tokens used")
    
    return summary


def main():
    """
    Example usage of the Azure OpenAI Image Processor.
    
    Configuration is loaded from .env file with these variables:
    - AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint
    - AZURE_OPENAI_DEPLOYMENT_NAME: Your vision model deployment name
    - AZURE_OPENAI_API_KEY: Your API key
    """
    
    # Example usage - process a single image
    try:
        # Example image path (update with your actual image)
        image_path = "images/mark-abenaki_page_001.png"
        
        # Process the image using the main function
        result = process_dictionary_image(
            image_path=image_path,
            user_prompt="See image here. Do your thing."
        )
        
        # Display results
        if result["success"]:
            print(f"\n✅ Successfully processed: {result['image_path']}")
            print(f"📊 Tokens used: {result['usage']['total_tokens']}")
            print(f"🔄 Finish reason: {result['finish_reason']}")
            print(f"\n📝 Extracted content:\n{result['content']}")
        else:
            print(f"❌ Error processing image: {result['error']}")
            
    except Exception as e:
        logger.error(f"Example execution failed: {str(e)}")
        print(f"❌ Failed to run example: {str(e)}")


if __name__ == "__main__":
    main()