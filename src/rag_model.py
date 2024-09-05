import openai
from openai import OpenAI
from typing import Optional, List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import re
import os
import torch
import torch.nn.functional as F
from PIL import Image
import pdf2image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import base64
from io import BytesIO

# For the embedding model, use the 'royokong/e5-v' which supports multiple languages

class RAGModel:
    def __init__(self, api_key, model_name):
        openai.api_key = api_key
        self.model_name = model_name        
        self.processor = LlavaNextProcessor.from_pretrained('royokong/e5-v')
        self.model = LlavaNextForConditionalGeneration.from_pretrained('royokong/e5-v', torch_dtype=torch.float16).cuda()
        self.img_prompt = '<|start_header_id|>user<|end_header_id|>\n\n<image>\nAnalyze an ESG-related report image, and extract promise and evidence information: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
    
    def load_pdf_as_image(self, pdf_path: str, page_number: int) -> Image.Image:
        """
        Load a specific page from a PDF as an image.
        """
        images = pdf2image.convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
        return images[0]  

    def extract_json_text(self, text: str) -> Optional[str]:
        # Extract only the JSON data (the part enclosed in "{}").
        json_pattern = re.compile(r'\{[^{}]*\}')
        matches = json_pattern.findall(text)
        
        if matches:
            try:
                json_obj = json.loads(matches[0])
                return json.dumps(json_obj, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                pass        
        return None
    
    def resize_image(self, image: Image.Image, scale_factor: float = 0.05) -> Image.Image:
        """
        Resize the image by a given scale factor.
        
        :param image: Original PIL Image
        :param scale_factor: Scale factor for resizing (e.g., 0.5 for 50% of original size)
        :return: Resized PIL Image
        """
        if scale_factor <= 0 or scale_factor > 1:
            raise ValueError("Scale factor must be between 0 and 1")
        
        width, height = image.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        return image.resize((new_width, new_height), Image.LANCZOS)

    def image_to_base64(self, image: Image.Image, scale_factor: float = 0.05, quality: int = 95) -> str:
        """
        Convert a PIL Image to a base64 encoded string, with resizing and compression.
        
        :param image: Original PIL Image
        :param scale_factor: Scale factor for resizing (e.g., 0.5 for 50% of original size)
        :param quality: JPEG quality (0-100)
        :return: Base64 encoded string of the image
        """
        # Create a copy of the image to avoid modifying the original
        img_copy = image.copy()

        # Convert to RGB if it's not (this handles RGBA or other color modes)
        if img_copy.mode != 'RGB':
            img_copy = img_copy.convert('RGB')
        
        # Resize the image
        img_copy = self.resize_image(img_copy, scale_factor)
        
        # Save the image to a BytesIO object
        buffered = BytesIO()
        img_copy.save(buffered, format="PNG", quality=quality, optimize=True)
        
        # Encode to base64
        return base64.b64encode(buffered.getvalue()).decode()

# The parts of the prompt that explains the JSON structure are to be changed according to the language since the JSON structure differs for each language's dataset.

    def analyze_paragraph(self, image: Image.Image, pdf_name: str, page_number: int) -> Dict[str, str]:
        """
        Generate annotation results from paragraph text using an LLM, referencing similar data.

        Args:
            paragraph (str): Input paragraph text

        Returns:
            Dict[str, str]: Annotation results in JSON format
        """     

        # The prompt is written for Chinese data.

        prompt = f"""
        You are an expert in extracting ESG-related promise and their corresponding evidence from corporate reports that describe ESG matters.
        I will provide image of actual company reports, so analyze the given image and follow the instructions below to provide careful and consistent annotations.
        Although you are specified to output in JSON format, perform the thought process in natural language and output the result in JSON format at the end.
        
        Annotation procedure:
        1. You will be given an image of a page from an ESG report.
        2. Determine if a promise is included, and indicate "Yes" if included, "No" if not included. (promise_status)
        3. If a promise is included (if promise_status is "Yes"), also provide the following information:
        - The specific part of the promise (extract verbatim from the text in the image without changing a single word) (promise_string)
        - When the promise can be verified ("already", "Less than 2 years", "2 to 5 years", "More than 5 years", "N/A") (verification_timeline)
        - Whether evidence is included ("Yes", "No", "N/A") (evidence_status)
        4. If evidence is included (if evidence_status is "Yes"), also provide the following information:
        - The part containing the evidence (extract directly from the text in the image without changing a single word) (evidence_string)
        - The quality of the relationship between the promise and evidence ("Clear", "Not Clear", "Misleading", "N/A") (evidence_quality)
           
        Definitions and criteria for annotation labels:
        1. promise_status - A promise is composed of a statement (a company principle, commitment, or strategy related to ESG criteria).:
        - "Yes": A promise exists.
        - "No": No promise exists.
        
        2. verification_timeline - The Verification Timeline is the assessment of when we could possibly see the final results of a given ESG-related action and thus verify the statement.:
        - "already": Qualifies ESG-related measures that have already been and keep on being applied and every small measure whose results can already be verified anyway.
        - "Less than 2 years": ESG-related measures whose results can be verified within 2 years.
        - "2 to 5 years": ESG-related measures whose results can be verified in 2 to 5 years.
        - "More than 5 years": ESG-related measures whose results can be verified in more than 5 years.
        - "N/A": When no promise exists. (Or when the promise is not verifiable.)

        3. evidence_status - Pieces of evidence are elements deemed the most relevant to exemplify and prove the core promise is being kept, which includes but is not limited to simple examples, company measures, numbers, etc.:
        - "Yes": Evidence supporting the promise exists.
        - "No": No evidence for the promise exists.
        - "N/A": When no promise exists.

        4. evidence_quality - The Evidence Quality is the assessment of the company's ability to back up their statement with enough clarity and precision.:
        - "Clear": There is no lack of information and what is said is intelligible and logical.
        - "Not Clear": An information is missing so much so that what is said may range from intelligible and logical to superficial and/or superfluous.
        - "Misleading": The evidence, whether true or not, has no obvious connection with the point raised and is used to divert attention.
        - "N/A": When no evidence or promise exists.
        
        Important notes:
        - Consider the context of the entire image thoroughly. It's important to understand the meaning of the entire page, not just individual text bloks.
        - Pay attention to both textual and visual elements such as charts, diagrams, and illustrations in the image that might contain ESG-related information.
        - "promise_string" and "evidence_string" should be extracted verbatim from the original text in the image. If there is no corresponding text (when promise_status or evidence_status is No), output "N/A". The promise are written simply and concisely, so carefully read the text and extract the parts that are truly considered appropriate.

        Analyze the following image and provide results in the format described above:
        """

        client = OpenAI(api_key = openai.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in extracting ESG-related promise and their corresponding evidence from corporate reports that describe ESG matters."},
                {
                 "role": "user",
                 "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self.image_to_base64(image, scale_factor=0.05, quality=95)}"
                        }
                    },
                 ]
                }
            ],
            functions=[
                {
                    "name": "analyze_esg_paragraph",
                    "description": "Analyze an ESG-related paragraph and extract promise and evidence information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "promise_status": {"type": "string", "enum": ["Yes", "No"]},
                            "promise_string": {"type": "string"},
                            "verification_timeline": {"type": "string", "enum": ["already", "Less than 2 years", "2 to 5 years", "More than 5 years", "N/A"]},
                            "evidence_status": {"type": "string", "enum": ["Yes", "No", "N/A"]},
                            "evidence_string": {"type": "string"},
                            "evidence_quality": {"type": "string", "enum": ["Clear", "Not Clear", "Potentially Misleading", "N/A"]}
                        },
                        "required": ["pdf", "page_number", "promise_status", "promise_string", "verification_timeline", "evidence_status", "evidence_string", "evidence_quality"]
                    }
                }
            ],
            function_call={"name": "analyze_esg_paragraph"},
            temperature=0
        )
        
        # Extract only the content generated by GPT from the response data containing a lot of information, and format it in JSON.
        generated_text = self.extract_json_text(response.choices[0].message.function_call.arguments)
        load_generated_text = json.loads(generated_text)

        # Add pdf_name and page_number to the result
        result = {
            "pdf": pdf_name,
            "page_number": str(page_number),
            **load_generated_text
        }
        
        result = json.dumps(result, indent=2, ensure_ascii=False)
        return result
