import base64
import json
import math
import os
from typing import List, Dict, Any
from PIL import Image
import requests

class ImageDifferenceFinder:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Constructor: Receives the OpenAI API key and model name

        :param api_key: OpenAI API key
        :param model: Model to be used, defaults to "gpt-4"
        """
        self.api_key = api_key
        self.model = model
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode an image file to a Base64 string

        :param image_path: Path of the image to encode
        :return: Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def get_image_dimensions(self, image_path: str) -> (int, int):
        """
        Get the dimensions of an image

        :param image_path: Path of the image to get dimensions
        :return: Tuple of (width, height)
        """
        with Image.open(image_path) as img:
            return img.size  # returns (width, height)
    
    def prepare_payload(self, image_path1: str, image_path2: str) -> Dict[str, Any]:
        """
        Prepare the payload for the OpenAI API request

        :param image_path1: Path of the first image
        :param image_path2: Path of the second image
        :return: API request payload
        """
        base64_image1 = self.encode_image(image_path1)
        base64_image2 = self.encode_image(image_path2)

        width, height = self.get_image_dimensions(image_path1)
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": f"Please find the differences between two images of size {width}*{height}. Provide the rectangular regions (position, width, height) that enclose the differences. Return the results in an array as there may be multiple differences.",
                    "image1": f"data:image/png;base64,{base64_image1}",
                    "image2": f"data:image/png;base64,{base64_image2}"
                }
            ],
            "functions": [
                {
                    "name": "image_diff",
                    "description": "Region definition for image difference",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "regions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "position_x": {
                                            "type": "number",
                                            "description": "X coordinate of the clipping region (top-left, px)",
                                        },
                                        "position_y": {
                                            "type": "number",
                                            "description": "Y coordinate of the clipping region (top-left, px)",
                                        },
                                        "width": {
                                            "type": "number",
                                            "description": "Width of the clipping region (px)",
                                        },
                                        "height": {
                                            "type": "number",
                                            "description": "Height of the clipping region (px)",
                                        }
                                    },
                                    "required": ["position_x", "position_y", "width", "height"]
                                }
                            }
                        },
                        "required": ["regions"]
                    }
                }
            ],
            "function_call": {"name": "image_diff"},
            "max_tokens": 300
        }

        return payload

    def query_openai(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query the OpenAI API

        :param payload: API request payload
        :return: API response data
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            try:
                print(response.json())
            except json.JSONDecodeError:
                print("Response content is not valid JSON.")
            return None
        else:
            return response.json()
    
    def merge_and_adjust_regions(self, regions: List[Dict[str, int]], distance_threshold: int, margin: int, width: int, height: int) -> List[Dict[str, int]]:
        """
        Merge and adjust overlapping regions

        :param regions: List of region information
        :param distance_threshold: Distance threshold for merging
        :param margin: Margin around the region
        :param width: Width of the image
        :param height: Height of the image
        :return: Adjusted list of regions
        """
        def distance(r1, r2):
            center1 = (r1["position_x"] + r1["width"] / 2, r1["position_y"] + r1["height"] / 2)
            center2 = (r2["position_x"] + r2["width"] / 2, r2["position_y"] + r2["height"] / 2)
            return math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

        merged_regions = []

        while regions:
            region = regions.pop(0)
            overlapping_regions = [region]
            
            for other in list(regions):
                if distance(region, other) <= distance_threshold:
                    overlapping_regions.append(other)
                    regions.remove(other)
            
            min_x = min(r["position_x"] for r in overlapping_regions) - margin
            min_y = min(r["position_y"] for r in overlapping_regions) - margin
            max_x = max(r["position_x"] + r["width"] for r in overlapping_regions) + margin
            max_y = max(r["position_y"] + r["height"] for r in overlapping_regions) + margin
            
            min_x = max(min_x, 0)
            min_y = max(min_y, 0)
            max_x = min(max_x, width)
            max_y = min(max_y, height)
            
            merged_width = max_x - min_x
            merged_height = max_y - min_y
            
            merged_regions.append({
                "position_x": int(min_x),
                "position_y": int(min_y),
                "width": int(merged_width),
                "height": int(merged_height)
            })

        return merged_regions
    
    def save_cropped_region(self, image: Image.Image, region: Dict[str, int], original_filename: str, index: int):
        """
        Crop and save the region

        :param image: Original image
        :param region: Dictionary of the region to crop
        :param original_filename: Original image filename
        :param index: Index of the saved file
        """
        cropped_image = image.crop((region["position_x"], region["position_y"], 
                                    region["position_x"] + region["width"], region["position_y"] + region["height"]))
        filename = f"{os.path.splitext(original_filename)[0]}_diff{index}.png"
        cropped_image.save(filename)
        print(f"Saved: {filename}")

    def process_images(self, image_path1: str, image_path2: str, distance_threshold: int = 250, margin: int = 100):
        """
        Process the images to detect differences

        :param image_path1: Path of the first image
        :param image_path2: Path of the second image
        :param distance_threshold: Distance threshold for merging
        :param margin: Margin around the region
        """
        payload = self.prepare_payload(image_path1, image_path2)
        result = self.query_openai(payload)

        if result:
            if "choices" in result and len(result["choices"]) > 0:
                for choice in result["choices"]:
                    message = choice.get("message", {})
                    if "function_call" in message:
                        arguments = json.loads(message["function_call"]["arguments"])
                        regions = arguments.get("regions", [])
                        
                        if not regions:
                            print("No differences found.")
                            return
                        
                        width, height = self.get_image_dimensions(image_path1)
                        adjusted_regions = self.merge_and_adjust_regions(regions, distance_threshold, margin, width, height)

                        original_image1 = Image.open(image_path1)
                        original_image2 = Image.open(image_path2)

                        for idx, region in enumerate(adjusted_regions):
                            self.save_cropped_region(original_image1, region, image_path1, idx + 1)
                            self.save_cropped_region(original_image2, region, image_path2, idx + 1)

                        print(json.dumps(adjusted_regions, indent=2))
                    else:
                        print("Error: Function call not found in the response")
            else: 
                print("Error: No choices found in the response")
        else:
            print("Error: Failed to get a valid response from OpenAI")

