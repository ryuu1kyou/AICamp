import os

from image_difference_finder import ImageDifferenceFinder
# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
image_path1 = "image_before.png"
image_path2 = "image_after.png"

finder = ImageDifferenceFinder(api_key)
finder.process_images(image_path1, image_path2)
