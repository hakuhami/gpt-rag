import numpy as np
import openai
from openai import OpenAI
from typing import Optional, List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
from .roberta_evidence_quality_classificator import ESGQualityClassifier, QualityLabel

class RAGModel:
    def __init__(self, api_key, model_name):
        openai.api_key = api_key
        self.model_name = model_name
        self.embedder = SentenceTransformer('intfloat/multilingual-e5-large-instruct') 
        self.quality_classifier = ESGQualityClassifier()

    def prepare_documents(self, search_data: List[Dict]) -> None:
        """
        Prepare and encode the search data

        Args:
            search_data (List[Dict]): Data for search
        """
        self.search_data = search_data
        self.documents = [item['data'] for item in search_data]
        self.doc_embeddings = self.embedder.encode(self.documents)
    
    # リランクをしない場合
    def get_relevant_context(self, query: str, yes_with_evidence_count: int = 6, yes_without_evidence_count: int = 2, no_promise_count: int = 2) -> List[Dict]:
        """
        Retrieve documents related to the query, maintaining specific ratios of promise_status and evidence_status values.
        The evidence_quality label is removed from the returned documents.

        Args:
            query (str): Input query
            yes_with_evidence_count (int): Number of documents with promise_status "Yes" and evidence_status "Yes" to retrieve
            yes_without_evidence_count (int): Number of documents with promise_status "Yes" and evidence_status "No" to retrieve
            no_promise_count (int): Number of documents with promise_status "No" to retrieve

        Returns:
            List[Dict]: List of relevant documents with specified distribution of status values, excluding evidence_quality
        """
        query_embedding = self.embedder.encode([query])
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]

        # Create a list of (index, similarity, status) tuples
        indexed_similarities = [
            (i, sim, {
                'promise_status': self.search_data[i].get('promise_status', 'No'),
                'evidence_status': self.search_data[i].get('evidence_status', 'N/A')
            }) 
            for i, sim in enumerate(similarities)
        ]

        # Separate documents by status combinations
        yes_with_evidence = [
            (i, sim) for i, sim, status in indexed_similarities 
            if status['promise_status'] == 'Yes' and status['evidence_status'] == 'Yes'
        ]
        yes_without_evidence = [
            (i, sim) for i, sim, status in indexed_similarities 
            if status['promise_status'] == 'Yes' and status['evidence_status'] == 'No'
        ]
        no_promise = [
            (i, sim) for i, sim, status in indexed_similarities 
            if status['promise_status'] == 'No'
        ]

        # Sort each category by similarity
        yes_with_evidence.sort(key=lambda x: x[1], reverse=True)
        yes_without_evidence.sort(key=lambda x: x[1], reverse=True)
        no_promise.sort(key=lambda x: x[1], reverse=True)

        # Select top documents from each category
        selected_yes_with_evidence = yes_with_evidence[:yes_with_evidence_count]
        selected_yes_without_evidence = yes_without_evidence[:yes_without_evidence_count]
        selected_no_promise = no_promise[:no_promise_count]

        # Combine all selected documents in the specified order
        all_selected = (
            selected_yes_with_evidence +
            selected_yes_without_evidence +
            selected_no_promise
        )

        # Get the corresponding documents and remove evidence_quality
        result = []
        for i, _ in all_selected:
            doc = self.search_data[i].copy()  # Create a copy of the document
            if 'evidence_quality' in doc:
                del doc['evidence_quality']  # Remove evidence_quality if it exists
            result.append(doc)

        return result

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
        You are an expert in extracting and classifying promise and supporting evidence from corporate texts related to ESG.
        Extract and classify the promise and supporting evidence from the provided test data, and output each content to the corresponding label in the json format.
        Carefully consider the detailed task explanation and reference examples step-by-step before proceeding with the task.
        The content is provided under four tags: <json format>, <the details of the task>, <extraction/classification examples>, and <test data>.
        
                
        <json format>
        Output the results extracted and classified from the test data according to the json format below.
        The reference examples also follow the json format below.
        Put the text of the test data in the "data".
        
        {{
            "data": str,
            "promise_status": str,
            "promise_string": str or null,
            "verification_timeline": str,
            "evidence_status": str,
            "evidence_string": str or null
        }}:
        
        
        <the details of the task>
        
        Task Steps:
        1. Read the examples in <extraction/classification examples> carefully and learn the characteristics of extraction and classification.
        2. Put the text of the test data verbatim in the "data" label, and read it carefully.
        3. Classification task (About "promise_status"):
           If the test data contains the contents that are considered to be promise, it is classified as "Yes".
           If the test data does not contain the contents that are considered to be promise, it is classified as "No".
        4. Extraction task (About "promise_string"):
           If "promise_status" is "Yes", extract the promise from the test data. (extract verbatim from the text without changing a single word)
           If "promise_status" is "No", output a blank.
        5. Classification task (About "verification_timeline"):
           If "promise_status" is "Yes", after carefully reading the "promise_string", classify the time when the promise can be verified into one of the four options: "already", "within_2_years", "between_2_and_5_years", or "more_than_5_years".
           If "promise_status" is "No", output "N/A".
        6. Classification task (About "evidence_status"):
           If "promise_status" is "Yes" and there is content in the test data that is considered to be evidence supporting the content of "promise_string", classify it as "Yes".
           If "promise_status" is "Yes" and there is no content in the test data that is considered to be evidence supporting the content of "promise_string", classify it as "No".
           If "promise_status" is "No", output "N/A".
        7. Extraction task (About "evidence_string"):
           If "evidence_status" is "Yes", extract the evidence from the test data. (extract verbatim from the text without changing a single word)
           If "evidence_status" is "No", output a blank.
           
        Definitions of each label and the thought process behind the task:
        1. Read the <extraction/classification examples> carefully and learn what content is considered to be a promise or evidence.
        2. Based on the features learned from the examples in step 1, carefully read the contents of the test data.
        3, 4. In this task, "promise" is expressed as expressions such as a company's ESG-related "corporate philosophy," "commitments being implemented or planned," "strategies for the future," and "statements for the future."
              Based on the features of the promise learned in the first step, and taking these concepts into account, determine whether the test data contains the promise and which parts are the contents of the promise.
        5. Based on the features of the promise learned in the first step, think carefully about when the contents of "promise_string" can be verified, following the definition below.
           "already": When the promise have already been applied, or whether or not it is applied, can already be verified.
           "within_2_years": When the promise can be verified within 2 years. (When the promise can be verified in the near future.)
           "between_2_and_5_years": When the promise can be verified in 2 to 5 years. (When the promise can be verified in the not too distant future, though not in the near future.)
           "more_than_5_years: When the promise can be verified in more than 5 years. (When the promsie can be verified in the distant future.)
        6, 7. In this task, "evidence" is expressed as "specific examples of the contents of the promise," "detailed explanation of the contents of the promise," "current status of the contents of the promise," etc.
              Based on the features of the evidence learned in the first step, and taking these concepts into account, determine whether the test data contains the evidence supporting the promise and which parts are the contents of the evidnece.
                
        Important notes:
        You must output the results in the format specified by <json format>, but the thought process described above is carried out step by step using natural language, and then the reasoning results in natural language are output in <json format>.
        Consider the context and logical relationships of the sentences thoroughly. It's important to understand the meaning of the entire paragraph, not just individual sentences.
        The evidence for the promise may not be directly stated, so think carefully.
        "promise_string" and "evidence_string" should be extracted verbatim from the original text. If there is no corresponding text (when promise_status or evidence_status is No), output a blank.
        Concepts specific to each company or industry may appear in the text, so think carefully about their meaning and appropriately interpret them.  
        
        
        <extraction/classification examples>
        
        {context}
            
               
        <test data>

        {paragraph}
        """

        client = OpenAI(api_key = openai.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in extracting ESG-related promise and their corresponding evidence from corporate reports that describe ESG matters."},
                {"role": "user", "content": prompt}
            ],
            functions=[
                {
                    "name": "analyze_esg_paragraph",
                    "description": "Analyze an ESG-related paragraph and extract promise and evidence information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data": {"type": "string"},
                            "promise_status": {"type": "string", "enum": ["Yes", "No"]},
                            "promise_string": {"type": "string"},
                            "verification_timeline": {"type": "string", "enum": ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years", "N/A"]},
                            "evidence_status": {"type": "string", "enum": ["Yes", "No", "N/A"]},
                            "evidence_string": {"type": "string"}
                        },
                        "required": ["data", "promise_status", "promise_string", "verification_timeline", "evidence_status", "evidence_string"]
                    }
                }
            ],
            function_call={"name": "analyze_esg_paragraph"},
            temperature=0
        )
        
        # Extract only the content generated by GPT from the response data containing a lot of information, and format it in JSON.
        generated_text = self.extract_json_text(response.choices[0].message.function_call.arguments)
        result_dict = json.loads(generated_text)

        # RoBERTaによる根拠の質の評価
        if result_dict["promise_status"] == "Yes" and result_dict["evidence_status"] == "Yes":
            quality_result = self.quality_classifier.classify_evidence_quality(
                result_dict["promise_string"],
                result_dict["evidence_string"]
            )
            result_dict["evidence_quality"] = quality_result.quality_label
        else:
            result_dict["evidence_quality"] = QualityLabel.NOT_APPLICABLE

        result = json.dumps(result_dict, indent=2, ensure_ascii=False)
        return result
