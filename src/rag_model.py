import openai
from openai import OpenAI
from typing import Optional, List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import re

# For the embedding model, use the 'multilingual-e5-large-instruct' which supports multiple languages

class RAGModel:
    def __init__(self, api_key, model_name):
        openai.api_key = api_key
        self.model_name = model_name
        self.embedder = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

    def prepare_documents(self, search_data: List[Dict]) -> None:
        """
        Prepare and encode the search data

        Args:
            search_data (List[Dict]): Data for search
        """
        self.search_data = search_data
        self.documents = [item['data'] for item in search_data]
        self.doc_embeddings = self.embedder.encode(self.documents)

    # Retrieve the top 5 items from the target search data with the highest cosine similarity to the input paragraph.
    def get_relevant_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve the top documents related to the query

        Args:
            query (str): Input query
            top_k (int): Number of documents to retrieve

        Returns:
            List[Dict]: List of relevant documents
        """
        query_embedding = self.embedder.encode([query])
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.search_data[i] for i in top_indices]

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

# The parts of the prompt that explains the JSON structure are to be changed according to the language since the JSON structure differs for each language's dataset.

    def analyze_paragraph(self, paragraph: str) -> Dict[str, str]:
        """
        Generate annotation results from paragraph text using an LLM, referencing similar data.

        Args:
            paragraph (str): Input paragraph text

        Returns:
            Dict[str, str]: Annotation results in JSON format
        """
        relevant_docs = self.get_relevant_context(paragraph)
        context = "\n".join([json.dumps(doc, ensure_ascii=False, indent=2) for doc in relevant_docs])

        prompt = f"""
        You are an expert in extracting ESG-related promise and their corresponding evidence from corporate reports that describe ESG matters.
        Follow the instructions below to provide careful and consistent annotations.
        Output the results in the following JSON format.
        Ensure that your response is a valid JSON object.
        Do not include any text before or after the JSON object.:
        {{
            "data": str,
            "promise_status": str,
            "promise_string": str or null,
            "verification_timeline": str,
            "evidence_status": str,
            "evidence_string": str or null,
            "evidence_quality": str
        }}:
        Although you are specified to output in JSON format, perform the thought process in natural language and output the result in JSON format at the end.
        
        Annotation procedure:
        1. You will be given the content of a paragraph.
        2. Determine if a promise is included, and indicate "Yes" if included, "No" if not included. (promise_status)
        3. If a promise is included (if promise_status is "Yes"), also provide the following information:
        - The specific part of the promise (extract verbatim from the text without changing a single word) (promise_string)
        - When the promise can be verified ("already", "within_2_years", "between_2_and_5_years", "more_than_5_years", "N/A") (verification_timeline)
        - Whether evidence is included ("Yes", "No", "N/A") (evidence_status)
        4. If evidence is included (if evidence_status is "Yes"), also provide the following information:
        - The part containing the evidence (extract directly from the text without changing a single word) (evidence_string)
        - The quality of the relationship between the promise and evidence ("Clear", "Not Clear", "Misleading", "N/A") (evidence_quality)
           
        Definitions and criteria for annotation labels:
        1. promise_status - A promise is composed of a statement (a company principle, commitment, or strategy related to ESG criteria).:
        - "Yes": A promise exists.
        - "No": No promise exists.
        
        2. verification_timeline - The Verification Timeline is the assessment of when we could possibly see the final results of a given ESG-related action and thus verify the statement.:
        - "already": Qualifies ESG-related measures that have already been and keep on being applied and every small measure whose results can already be verified anyway.
        - "within_2_years": ESG-related measures whose results can be verified within 2 years.
        - "between_2_and_5_years": ESG-related measures whose results can be verified in 2 to 5 years.
        - "more_than_5_years: ESG-related measures whose results can be verified in more than 5 years.
        - "N/A": When no promise exists.

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
        - Consider the context thoroughly. It's important to understand the meaning of the entire paragraph, not just individual sentences.
        - For indirect evidence, carefully judge its relevance.
        - "promise_string" and "evidence_string" should be extracted verbatim from the original text. If there is no corresponding text (when promise_status or evidence_status is No), output a blank.
        - Understand and appropriately interpret industry-specific terms.

        The following are annotation examples of texts similar to the text you want to analyze.
        Refer to these examples, think about why these examples have such annotation results, and then output the results.
        Examples for your reference are as follows:
        {context}

        Analyze the following text and provide results in the format described above:
        {paragraph}
        """

        client = OpenAI(api_key = openai.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in extracting ESG-related promise and their corresponding evidence from corporate reports that describe ESG matters."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        # Extract only the content generated by GPT from the response data containing a lot of information, and format it in JSON.
        generated_text = self.extract_json_text(response.choices[0].message.content)        
        load_generated_text = json.loads(generated_text)
        
        result = json.dumps(load_generated_text, indent=2, ensure_ascii=False)
        return result