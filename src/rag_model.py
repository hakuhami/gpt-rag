import numpy as np
import openai
from openai import OpenAI
from typing import Optional, List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
#from ragatouille import RAGPretrainedModel
import json
import re

# For the embedding model, use the 'multilingual-e5-large-instruct' which supports multiple languages

class RAGModel:
    def __init__(self, api_key, model_name):
        openai.api_key = api_key
        self.model_name = model_name
        self.embedder = SentenceTransformer('intfloat/multilingual-e5-large-instruct') 
        #self.reranker =  RAGPretrainedModel.from_pretrained("bclavie/JaColBERTv2")

    def prepare_documents(self, search_data: List[Dict]) -> None:
        """
        Prepare and encode the search data

        Args:
            search_data (List[Dict]): Data for search
        """
        self.search_data = search_data
        self.documents = [item['data'] for item in search_data]
        self.doc_embeddings = self.embedder.encode(self.documents)
        
    # # リランクをする場合
    # def rerank_documents(self, query: str, candidates: List[Dict]) -> List[Dict]:
    #     """
    #     JaColBERTを使用して候補文書をリランキング

    #     Args:
    #         query (str): 入力クエリ
    #         candidates (List[Dict]): 初期候補文書

    #     Returns:
    #         List[Dict]: リランキングされた文書
    #     """
        
    #     try:
    #         # リランカーのドキュメントをクリア
    #         self.reranker.clear_encoded_docs()
    #     except AttributeError:
    #         # in_memory_collectionが存在しない場合は無視
    #         pass
        
    #     # 候補文書をエンコード
    #     candidate_texts = [doc['data'] for doc in candidates]
    #     self.reranker.encode(candidate_texts)
        
    #     try:
    #         # エンコードと検索を実行
    #         self.reranker.encode(candidate_texts)
    #         rerank_results = self.reranker.search_encoded_docs(
    #             query=query,
    #             k=len(candidates)
    #         )
            
    #         # 結果の順序に基づいて文書を並び替え
    #         reranked_docs = []
    #         for result in rerank_results:
    #             original_idx = candidate_texts.index(result['content'])
    #             reranked_docs.append(candidates[original_idx])
                
    #     except Exception as e:
    #         # エラーが発生した場合は、元の順序をそのまま返す
    #         print(f"***Reranking error: {str(e)}")
    #         reranked_docs = candidates
        
    #     return reranked_docs    

    # def get_relevant_context(self, query: str, yes_count: int = 8, no_count: int = 2) -> List[Dict]:
    #     """
    #     コサイン類似度による初期検索とJaColBERTによるリランキングを使用して関連文書を検索

    #     Args:
    #         query (str): 入力クエリ
    #         yes_count (int): 取得するpromise_status "Yes"の文書数
    #         no_count (int): 取得するpromise_status "No"の文書数

    #     Returns:
    #         List[Dict]: 指定された分布のpromise_statusを持つ関連文書のリスト
    #     """
    #     # 初期検索（コサイン類似度）
    #     query_embedding = self.embedder.encode([query])
    #     similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        
    #     # Create a list of (index, similarity, promise_status) tuples
    #     indexed_similarities = [
    #         (i, sim, self.search_data[i].get('promise_status', 'No')) 
    #         for i, sim in enumerate(similarities)
    #     ]
        
    #     # Separate documents by promise_status
    #     yes_docs = [(i, sim) for i, sim, status in indexed_similarities if status == 'Yes']
    #     no_docs = [(i, sim) for i, sim, status in indexed_similarities if status == 'No']
        
    #     # Sort by similarity (descending order)
    #     yes_docs.sort(key=lambda x: x[1], reverse=True)
    #     no_docs.sort(key=lambda x: x[1], reverse=True)
        
    #     # 初期候補を選択（Yes: 24件、No: 6件）
    #     initial_yes = yes_docs[:24]
    #     initial_no = no_docs[:6]
        
    #     # 初期候補のドキュメントを取得
    #     initial_candidates = (
    #         [self.search_data[i] for i, _ in initial_yes] +
    #         [self.search_data[i] for i, _ in initial_no]
    #     )
        
    #     # JaColBERTを使用したリランキング
    #     reranked_docs = self.rerank_documents(query, initial_candidates)
        
    #     # 最終的な結果を選択（Yes: 8件、No: 2件）
    #     final_yes = []
    #     final_no = []
        
    #     for doc in reranked_docs:
    #         if doc.get('promise_status') == 'Yes' and len(final_yes) < yes_count:
    #             final_yes.append(doc)
    #         elif doc.get('promise_status') == 'No' and len(final_no) < no_count:
    #             final_no.append(doc)
                
    #         if len(final_yes) == yes_count and len(final_no) == no_count:
    #             break
        
    #     return final_yes + final_no
    
    # リランクをしない場合
    def get_relevant_context(self, query: str, yes_with_evidence_count: int = 6, yes_without_evidence_count: int = 2, no_promise_count: int = 2) -> List[Dict]:
        """
        Retrieve documents related to the query, maintaining specific ratios of promise_status and evidence_status values.

        Args:
            query (str): Input query
            yes_with_evidence_count (int): Number of documents with promise_status "Yes" and evidence_status "Yes" to retrieve
            yes_without_evidence_count (int): Number of documents with promise_status "Yes" and evidence_status "No" to retrieve
            no_promise_count (int): Number of documents with promise_status "No" to retrieve

        Returns:
            List[Dict]: List of relevant documents with specified distribution of status values
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

        # Get the corresponding documents
        result = [self.search_data[i] for i, _ in all_selected]

        return result
    
    # def get_relevant_context(self, query: str, top_k: int = 10) -> List[Dict]:
    #     """
    #     Retrieve the top documents related to the query

    #     Args:
    #         query (str): Input query
    #         top_k (int): Number of documents to retrieve

    #     Returns:
    #         List[Dict]: List of relevant documents
    #     """
    #     query_embedding = self.embedder.encode([query])
    #     similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]
    #     top_indices = np.argsort(similarities)[-top_k:][::-1]
    #     return [self.search_data[i] for i in top_indices]

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
        You are an expert in extracting and classifying promise and supporting evidence from corporate texts related to ESG.
        Extract and classify the promise and supporting evidence from the provided test data, and output each content to the corresponding label in the json format.
        Carefully consider the detailed task explanation and reference examples step-by-step before proceeding with the task.
        The content is provided under four tags: <json format>, <the details of the task>, <extraction/classification examples>, and <test data>.
        
                
        <json format>
        Output the results extracted and classified from <test data> according to the json format below.
        <extraction/classification examples> also follow the json format below.
        Put the text of <test data> in the "data".
        
        {{
            "data": str,
            "promise_status": str,
            "promise_string": str or null,
            "verification_timeline": str,
            "evidence_status": str,
            "evidence_string": str or null,
            "evidence_quality": str
        }}:
        
        
        <the details of the task>
        First, understand the details of the steps in the task.
        Then, understand the definitions of each label for extraction and classification, and the thought process at each step.
        
        # Task Steps:
        1. Read the examples in <extraction/classification examples> carefully and learn the characteristics of extraction and classification.
        2. Put the text of <test data> verbatim in the "data" label, and read it carefully.
        3. Classification task (About "promise_status"):
           If <test data> contains the contents that are considered to be promise, it is classified as "Yes".
           If <test data> does not contain the contents that are considered to be promise, it is classified as "No".
        4. Extraction task (About "promise_string"):
           If "promise_status" is "Yes", extract the promise from <test data>. (extract verbatim from the text without changing a single word)
           If "promise_status" is "No", output a blank.
        5. Classification task (About "verification_timeline"):
           If "promise_status" is "Yes", after carefully reading the "promise_string", classify the time when the promise can be verified into one of the four options: "already", "within_2_years", "between_2_and_5_years", or "more_than_5_years".
           If "promise_status" is "No", output "N/A".
        6. Classification task (About "evidence_status"):
           If "promise_status" is "Yes" and there is content in <test data> that is considered to be evidence supporting the content of "promise_string", classify it as "Yes".
           If "promise_status" is "Yes" and there is no content in <test data> that is considered to be evidence supporting the content of "promise_string", classify it as "No".
           If "promise_status" is "No", output "N/A".
        7. Extraction task (About "evidence_string"):
           If "evidence_status" is "Yes", extract the evidence from <test data>. (extract verbatim from the text without changing a single word)
           If "evidence_status" is "No", output a blank.
        8. Classification task (About "evidence_quality"):
           If "evidence_status" is "Yes", after carefully reading the contents of "promise_string" and "evidence_string", consider how well the contents of "evidence_string" support the contents of "promise_string" and classify the relationship between the promise and the evidence as "Clear", "Not Clear", or "Misleading".  
           If "evidence_status"is "No", output "N/A".     
           
        # Definitions of each label and the thought process behind the task:
        1. Read the <extraction/classification examples> carefully and learn what content is considered to be a promise or evidence and.
           In particular, the judgment of "evidence_quality" is the most important and difficult part of this task, so learn how it can be classified thoroughly.
        2. Based on the features learned from the examples in step 1, carefully read the contents of <test data>.
        3, 4. In this task, "promise" is expressed as expressions such as a company's ESG-related "corporate philosophy," "commitments being implemented or planned," "strategies for the future," and "statements for the future."
              Based on the features of the promise learned in step 1, and taking these concepts into account, determine whether <test data> contains the promise and which parts are the contents of the promise.
        5. Based on the features of the promise learned in step 1, think carefully about when the contents of "promise_string" can be verified, following the definition below.
           "already": When the promise have already been applied, or whether or not it is applied, can already be verified.
           "within_2_years": When the promise can be verified within 2 years. (When the promise can be verified in the near future.)
           "between_2_and_5_years": When the promise can be verified in 2 to 5 years. (When the promise can be verified in the not too distant future, though not in the near future.)
           "more_than_5_years: When the promise can be verified in more than 5 years. (When the promsie can be verified in the distant future.)
        6, 7. In this task, "evidence" is expressed as "specific examples of the contents of the promise," "detailed explanation of the contents of the promise," "current status of the contents of the promise," etc.
              Based on the features of the evidence learned in step 1, and taking these concepts into account, determine whether <test data> contains the evidence supporting the promise and which parts are the contents of the evidnece.
        8. Based on the features learned in step 1, think carefully about how well the contents of "evidence_string" support the contents of "promise_string".
           Then, think carefully about which label the quality of the relationship between the promise and the evidence falls into, following the definitions below.
           "Clear": In the content of "evidence_string", there is no lack of information and what is said is intelligible and logical.
           "Not Clear": In the content of "evidence_string", some information is missing or not well described so that what is said may range from intelligible and logical to superficial and/or superfluous.           
           "Misleading": In the content of "evidence_string", it is not suitable to support the promise, or is not relevant to the contents of the promise, or may distract readers, or is untrue.
                
        # Important notes:
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
                            "evidence_string": {"type": "string"},
                            "evidence_quality": {"type": "string", "enum": ["Clear", "Not Clear", "Misleading", "N/A"]}
                        },
                        "required": ["data", "promise_status", "promise_string", "verification_timeline", "evidence_status", "evidence_string", "evidence_quality"]
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
