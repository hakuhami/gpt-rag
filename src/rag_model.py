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

    def embed_image(self, image: Image.Image) -> torch.Tensor:
        """
        Embed an image using the e5-v model.
        """
        inputs = self.processor([self.img_prompt], [image], return_tensors="pt", padding=True).to('cuda')
        with torch.no_grad():
            emb = self.e5v_model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
        return F.normalize(emb, dim=-1)    
    
    def prepare_documents(self, search_data: List[Dict], pdf_dir: str) -> None:
        """
        Prepare and encode the search data
        """
        self.search_data = search_data
        self.doc_embeddings = []
        self.doc_images = []
        for item in search_data:
            pdf_path = os.path.join(pdf_dir, f"{item['pdf']}")
            page_number = int(item['page_number'])
            image = self.load_pdf_as_image(pdf_path, page_number)
            embedding = self.embed_image(image)
            self.doc_embeddings.append(embedding)
            self.doc_images.append(image)
        self.doc_embeddings = torch.cat(self.doc_embeddings, dim=0)    
    
    # # Retrieve the top 6 items from the target search data with the highest cosine similarity to the input paragraph.
    def get_relevant_context(self, query_image: Image.Image, top_k: int = 6) -> List[Dict]:
        """
        Retrieve the top documents related to the query image
        """
        query_embedding = self.embed_image(query_image)
        similarities = F.cosine_similarity(query_embedding, self.doc_embeddings)
        top_indices = torch.argsort(similarities, descending=True)[:top_k]
        # return [self.search_data[i] for i in top_indices]
        return [
            {**self.search_data[i], "image": self.doc_images[i]}
            for i in top_indices
        ]

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
    
    def image_to_base64(self, image: Image.Image) -> str:
        """
        Convert a PIL Image to a base64 encoded string.
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
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
        
        # page_image = self.get_page_image(pdf_path, page_number)
        # relevant_docs = self.get_relevant_context(page_image)
        # context = "\n".join([json.dumps(doc, ensure_ascii=False, indent=2) for doc in relevant_docs])
        
        # relevant_docs = self.get_relevant_context(image)
        # context = "\n".join([json.dumps(doc, ensure_ascii=False, indent=2) for doc in relevant_docs])
        
        relevant_docs = self.get_relevant_context(image)
        
        # Prepare the context with both text information and base64 encoded images
        context = []
        for doc in relevant_docs:
            doc_info = {k: v for k, v in doc.items() if k != 'image'}
            doc_info['image_base64'] = self.image_to_base64(doc['image'])
            context.append(json.dumps(doc_info, ensure_ascii=False))

        context_str = "\n".join(context)        

        # The prompt is written for Chinese data.

        prompt = f"""
        You are an expert in extracting ESG-related promise and their corresponding evidence from corporate reports that describe ESG matters.
        I will provide image of actual company reports, so analyze the given image and follow the instructions below to provide careful and consistent annotations.
        Ensure that your response is a valid JSON object.
        Do not include any text before or after the JSON object.
        Regarding the "pdf" and "page_number", be sure to output the content of the given paragraph without altering it and in str format.:
        {{
            "pdf": str,
            "page_number": int,
            "promise_status": str,
            "promise_string": str or null,
            "verification_timeline": str,
            "evidence_status": str,
            "evidence_string": str or null,
            "evidence_quality": str
        }}:
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
        - For indirect evidence, carefully judge its relevance.
        - "promise_string" and "evidence_string" should be extracted verbatim from the original text in the image. If there is no corresponding text (when promise_status or evidence_status is No), output a blank. The promise are written simply and concisely, so carefully read the text and extract the parts that are truly considered appropriate.
        - Understand and appropriately interpret industry-specific terms and visual representations.

        The following are annotation examples of image similar to the one you want to analyze.
        Refer to these examples, think about why these examples have such annotation results, and then output the results.
        Examples for your reference are as follows:
        {context_str}

        The image is from the PDF file named "{pdf_name}" and is page number {page_number}.
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
                            "url": f"data:image/png;base64,{self.image_to_base64(image)}",
                            "detail": "high"
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
                            "pdf": {"type": "string"},
                            "page_number": {"type": "int"},
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
        
        result = json.dumps(load_generated_text, indent=2, ensure_ascii=False)
        return result