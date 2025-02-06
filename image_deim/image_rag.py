import openai
from openai import OpenAI
from typing import Optional, List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import base64
import requests
import os

class RAGModel:
    def __init__(self, api_key: str, model_name: str) -> None:
        """
        Initialize the RAG model with API credentials and model configurations.
        テキスト分析は従来の関数呼び出し方式、画像分析は requests 経由で GPT-4o-mini を利用する。
        """
        openai.api_key = api_key
        self.model_name = model_name
        self.embedder = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
        self.client = OpenAI(api_key=openai.api_key)

    def prepare_documents(self, search_data: List[Dict]) -> None:
        """
        Prepare document embeddings for different search scenarios
        """
        self.search_data = search_data
        
        # Step 1の検索用データ
        self.documents = [item['data'] for item in search_data]
        self.doc_embeddings = self.embedder.encode(self.documents)
        
        # Step 2の検索用データ
        self.promise_only_data = [
            item for item in search_data 
            if item.get('promise_status') == 'Yes'
        ]
        self.promise_only_texts = [item['promise_string'] for item in self.promise_only_data]
        self.promise_only_embeddings = self.embedder.encode(self.promise_only_texts)
        
        # Step 3の検索用データ
        self.quality_data = [
            item for item in search_data 
            if item.get('promise_status') == 'Yes' and 
               item.get('evidence_status') == 'Yes' and 
               item.get('evidence_quality') != 'N/A'
        ]
        self.quality_texts = [item['promise_string'] for item in self.quality_data]
        self.quality_embeddings = self.embedder.encode(self.quality_texts)

    def search_step1_promise(self, query: str, yes_with_evidence_count: int = 6, 
                           yes_without_evidence_count: int = 2, no_promise_count: int = 2) -> List[Dict]:
        """
        Step 1: Retrieve documents for promise classification
        """
        query_embedding = self.embedder.encode([query])
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]

        # Create indexed similarities with status
        indexed_similarities = [
            (i, sim, {
                'promise_status': self.search_data[i].get('promise_status', 'No'),
                'evidence_status': self.search_data[i].get('evidence_status', 'N/A')
            }) 
            for i, sim in enumerate(similarities)
        ]

        # Separate by status combinations
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

        # Sort and select
        for category in [yes_with_evidence, yes_without_evidence, no_promise]:
            category.sort(key=lambda x: x[1], reverse=True)

        selected = (
            yes_with_evidence[:yes_with_evidence_count] +
            yes_without_evidence[:yes_without_evidence_count] +
            no_promise[:no_promise_count]
        )

        # Get filtered documents
        result = []
        for i, _ in selected:
            doc = self.search_data[i]
            filtered_doc = {
                'data': doc['data'],
                'promise_status': doc['promise_status'],
                'promise_string': doc.get('promise_string', ''),
                'evidence_status': doc['evidence_status'],
                'evidence_string': doc.get('evidence_string', '')
            }
            result.append(filtered_doc)
            
        print("↓がstep1の参考データ")
        print(f"{result}")
        print("↑がstep1の参考データ")

        return result

    def search_step2_timeline(self, promise_text: str) -> List[Dict]:
        """
        Step 2: Retrieve documents for verification timeline classification
        """
        query_embedding = self.embedder.encode([promise_text])
        similarities = cosine_similarity(query_embedding, self.promise_only_embeddings)[0]
        
        indexed_data = [
            (idx, sim, doc) 
            for idx, (sim, doc) in enumerate(zip(similarities, self.promise_only_data))
        ]
        
        timeline_categories = {
            'already': [],
            'within_2_years': [],
            'between_2_and_5_years': [],
            'more_than_5_years': []
        }
        
        for idx, sim, doc in indexed_data:
            timeline = doc.get('verification_timeline')
            if timeline in timeline_categories:
                timeline_categories[timeline].append((idx, sim, doc))
        
        selected_docs = []
        category_counts = {
            'already': 4,
            'within_2_years': 2,
            'between_2_and_5_years': 2,
            'more_than_5_years': 2
        }
        
        for category, count in category_counts.items():
            category_docs = timeline_categories[category]
            category_docs.sort(key=lambda x: x[1], reverse=True)
            selected_docs.extend([{
                'promise_string': doc['promise_string'],
                'verification_timeline': doc['verification_timeline']
            } for _, _, doc in category_docs[:count]])
            
        print("↓がstep2の参考データ")
        print(f"{selected_docs}")
        print("↑がstep2の参考データ")
        
        return selected_docs

    def search_step3_quality(self, promise_text: str) -> List[Dict]:
        """
        Step 3: Retrieve documents for evidence quality classification
        """
        query_embedding = self.embedder.encode([promise_text])
        similarities = cosine_similarity(query_embedding, self.quality_embeddings)[0]
        
        indexed_data = [
            (idx, sim, doc) 
            for idx, (sim, doc) in enumerate(zip(similarities, self.quality_data))
        ]
        
        quality_categories = {
            'Clear': [],
            'Not Clear': [],
            'Misleading': []
        }
        
        for idx, sim, doc in indexed_data:
            quality = doc.get('evidence_quality')
            if quality in quality_categories:
                quality_categories[quality].append((idx, sim, doc))
        
        selected_docs = []
        category_counts = {
            'Clear': 4,
            'Not Clear': 3,
            'Misleading': 3
        }
        
        for category, count in category_counts.items():
            category_docs = quality_categories[category]
            category_docs.sort(key=lambda x: x[1], reverse=True)
            selected_docs.extend([{
                'promise_string': doc['promise_string'],
                'evidence_string': doc['evidence_string'],
                'evidence_quality': doc['evidence_quality']
            } for _, _, doc in category_docs[:count]])
            
        print("↓がstep3の参考データ")
        print(f"{selected_docs}")
        print("↑がstep3の参考データ")
        
        return selected_docs

    def extract_json_text(self, text: str) -> Optional[str]:
        """Extract JSON text from string"""
        json_pattern = re.compile(r'\{[^{}]*\}')
        matches = json_pattern.findall(text)
        
        if matches:
            try:
                json_obj = json.loads(matches[0])
                return json.dumps(json_obj, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                pass        
        return None

    def classify_step1_promise(self, data: str) -> Dict[str, str]:
        """Step 1: Classify promise status and extract promise string"""
        similar_docs = self.search_step1_promise(data)
        context = "\n".join([json.dumps(doc, ensure_ascii=False, indent=2) for doc in similar_docs])
        
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
            "evidence_status": str,
            "evidence_string": str or null
        }}:
        
        
        <the details of the task>
        
        Task Steps:
        1. Read the examples in <extraction/classification examples> carefully and learn the features of extraction and classification.
        2. Put the text of the test data verbatim in the "data" label, and read it carefully.
        3. Classification task (About "promise_status"):
           If the test data contains the contents that are considered to be promise, it is classified as "Yes".
           If the test data does not contain the contents that are considered to be promise, it is classified as "No".
        4. Extraction task (About "promise_string"):
           If "promise_status" is "Yes", extract the promise from the test data. (extract verbatim from the text without changing a single word)
           If "promise_status" is "No", output a blank.
        5. Classification task (About "evidence_status"):
           If "promise_status" is "Yes" and there is content in the test data that is considered to be evidence supporting the content of "promise_string", classify it as "Yes".
           If "promise_status" is "Yes" and there is no content in the test data that is considered to be evidence supporting the content of "promise_string", classify it as "No".
           If "promise_status" is "No", output "N/A".
        6. Extraction task (About "evidence_string"):
           If "evidence_status" is "Yes", extract the evidence from the test data. (extract verbatim from the text without changing a single word)
           If "evidence_status" is "No", output a blank.
           
        Definitions of each label and the thought process behind the task:
        1. Read the <extraction/classification examples> carefully and learn the features of what content is considered to be promise or evidence.
        2. Based on the features learned from the examples in step 1, carefully read the contents of the test data.
        3, 4. In this task, "promise" is expressed as expressions such as a company's ESG-related "corporate philosophy," "commitments being implemented or planned," "strategies for the future," and "statements for the future."
              Based on the features of the promise learned in the first step, and taking these concepts into account, determine whether the test data contains the promise and which parts are the contents of the promise.
        5, 6. In this task, "evidence" is expressed as "specific examples of the contents of the promise," "detailed explanation of the contents of the promise," "current status of the contents of the promise," etc.
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

        {data}
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in extracting ESG-related promise and their corresponding evidence from corporate reports that describe ESG matters."},
                {"role": "user", "content": prompt}
            ],
            functions=[{
                "name": "analyze_esg_paragraph",
                "description": "Analyze an ESG-related paragraph and extract promise and evidence information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string"},
                        "promise_status": {"type": "string", "enum": ["Yes", "No"]},
                        "promise_string": {"type": "string"},
                        "evidence_status": {"type": "string", "enum": ["Yes", "No", "N/A"]},
                        "evidence_string": {"type": "string"}
                    },
                    "required": ["data", "promise_status", "promise_string", "evidence_status", "evidence_string"]
                }
            }],
            function_call={"name": "analyze_esg_paragraph"},
            temperature=0
        )
        
        result = json.loads(self.extract_json_text(response.choices[0].message.function_call.arguments))
        return result

    def classify_step2_timeline(self, promise_string: str) -> Dict[str, str]:
        """Step 2: Classify verification timeline"""
        similar_docs = self.search_step2_timeline(promise_string)
        context = "\n".join([json.dumps(doc, ensure_ascii=False, indent=2) for doc in similar_docs])
        
        prompt = f"""
        You are an expert in classifying promise from corporate texts related to ESG.
        Classify the verification timing of the promise from the provided test data, and output content to the corresponding label in the json format.
        Carefully consider the detailed task explanation and reference examples step-by-step before proceeding with the task.
        The content is provided under four tags: <json format>, <the details of the task>, <classification examples>, and <test data>.
        
                
        <json format>
        <classification examples> follow the json format below.
        Output the classification results specified in <the details of the task> to the following corresponding labels.
        
        {{
            "promise_string": str,
            "verification_timeline": str
        }}:
        
        
        <the details of the task>
        
        Task Steps:
        1. Read the examples in <classification examples> carefully and learn the features of classification.
        2. Based on the features learned from the examples in step 1, carefully read the contents of the test data.
        3. Classification task (About "verification_timeline"):
           After carefully reading the test data, classify the time when the promise can be verified into one of the four options: "already", "within_2_years", "between_2_and_5_years", or "more_than_5_years".
           
        Definitions of each label and the thought process behind the task:
        1. Read the <classification examples> carefully and learn the classification features of "verification_timeline".
        2. Based on the features learned from the examples in step 1, carefully read the contents of the test data.
        3. Based on the features learned in the first step, think carefully about when the contents of "promise_string" can be verified, following the definition below.
           Make full use of the classification characteristics learned in the first step.
           "already": When the promise have already been applied, or whether or not it is applied, can already be verified.
           "within_2_years": When the promise can be verified within 2 years. (When the promise can be verified in the near future.)
           "between_2_and_5_years": When the promise can be verified in 2 to 5 years. (When the promise can be verified in the not too distant future, though not in the near future.)
           "more_than_5_years: When the promise can be verified in more than 5 years. (When the promsie can be verified in the distant future.)
                
        Important notes:
        You must output the results in the format specified by <json format>, but the thought process described above is carried out step by step using natural language, and then the reasoning results in natural language are output in <json format>.
        Consider the context and logical relationships of the sentences thoroughly. It's important to understand the meaning of the entire paragraph, not just individual sentences.
        Concepts specific to each company or industry may appear in the text, so think carefully about their meaning and appropriately interpret them.
        
        
        <classification examples>
        
        {context}
            
               
        <test data>
    
        {{
            "promise_string": "{promise_string}"
        }}
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in classifying ESG-related promises."},
                {"role": "user", "content": prompt}
            ],
            functions=[{
                "name": "classify_verification_timeline",
                "description": "Classify when the promise can be verified",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "verification_timeline": {
                            "type": "string",
                            "enum": ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years"]
                        }
                    },
                    "required": ["verification_timeline"]
                }
            }],
            function_call={"name": "classify_verification_timeline"},
            temperature=0
        )
        
        result = json.loads(self.extract_json_text(response.choices[0].message.function_call.arguments))
        return result

    def encode_image(self, image_path: str) -> str:
        """
        画像ファイル (PNG) を読み込み、base64 エンコードした文字列を返す。
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8-sig')

    def classify_step3_quality_image(self, promise_string: str, evidence_string: str, image_path: str) -> Dict[str, str]:
        """
        Step3: 根拠のクオリティの分類を、画像情報を付加して実施する。
        従来のテキストのみのプロンプトに、画像 (PNG) を base64 化したものを追加して送信する。
        出力は JSON 形式の辞書として返す。
        """
        similar_docs = self.search_step3_quality(promise_string)
        context = "\n".join([json.dumps(doc, ensure_ascii=False, indent=2) for doc in similar_docs])
        
        prompt = f"""
        You are an expert in analyzing promise and supporting evidence from corporate texts related to ESG.
        Classify the quality of the evidence supporting the promise from the provided test data, and output content to the corresponding label in the json format.
        Carefully consider the detailed task explanation and reference examples step-by-step before proceeding with the task.
        The content is provided under four tags: <json format>, <the details of the task>, <classification examples>, and <test data>.
        And, the attached image contains supplementary information on the promise and evidence to be analyzed, so make sure to understand the content of the image before using it as a reference for analyzing.
        
                
        <json format>
        <classification examples> follow the json format below.
        Output the classification results specified in <the details of the task> to the following corresponding labels.
        
        {{
            "promise_string": str,
            "evidence_string": str,
            "evidence_quality": str
        }}:
        
        
        <the details of the task>
        
        Task Steps:
        1. Read the examples in <classification examples> carefully and learn the features of classification.
        2. Based on the features learned from the examples in step 1, carefully read the contents of the test data.
        3. Classification task (About "evidence_quality"):
           After carefully reading the test data, consider how well the contents of "evidence_string" support the contents of "promise_string" and classify the relationship between the promise and the evidence as "Clear", "Not Clear", or "Misleading".
           
        Definitions of each label and the thought process behind the task:
        1. Read the <classification examples> carefully and learn the classification features of "evidence_quality".
        2. Based on the features learned from the examples in step 1, carefully read the contents of the test data.
           In this task, "promise" is expressed as expressions such as a company's ESG-related "corporate philosophy," "commitments being implemented or planned," "strategies for the future," and "statements for the future."
           In this task, "evidence" is expressed as "specific examples of the contents of the promise," "detailed explanation of the contents of the promise," "current status of the contents of the promise," etc.
        3. Based on the features learned in the first step, think carefully about how well the contents of "evidence_string" support the contents of "promise_string".
           Then, think carefully about which label the quality of the relationship between the promise and the evidence falls into, following the definitions below.
           Make full use of the classification characteristics learned in the first step.
           "Clear": In the content of "evidence_string", there is no lack of information and what is said is intelligible and logical.
           "Not Clear": In the content of "evidence_string", some information is missing or not well described so that what is said may range from intelligible and logical to superficial and/or superfluous.           
           "Misleading": In the content of "evidence_string", it is not suitable to support the promise, or is not relevant to the contents of the promise, or may distract readers, or is untrue.
                
        Important notes:
        You must output the results in the format specified by <json format>, but the thought process described above is carried out step by step using natural language, and then the reasoning results in natural language are output in <json format>.
        Consider the context and logical relationships of the sentences thoroughly. It's important to understand the meaning of the entire paragraph, not just individual sentences.
        The evidence for the promise may not be directly stated, so think carefully.
        Concepts specific to each company or industry may appear in the text, so think carefully about their meaning and appropriately interpret them.
        
        
        <classification examples>
        
        {context}
            
               
        <test data>
        
        {{
            "promise_string": "{promise_string}",
            "evidence_string": "{evidence_string}"
        }}
        """
        base64_image = self.encode_image(image_path)
        
        # headers = {
        #     "Content-Type": "application/json",
        #     "Authorization": f"Bearer {openai.api_key}"
        # }
        # payload = {
        #     "model": "gpt-4o-mini",
        #     "messages": [
        #         {
        #             "role": "user",
        #             "content": [
        #                 {
        #                     "type": "text",
        #                     "text": prompt,
        #                 },
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {
        #                         "url": f"data:image/png;base64,{base64_image}"
        #                     }
        #                 }
        #             ]
        #         }
        #     ],
        #     "max_tokens": 300
        # }
        # response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        # response_json = response.json()
        # result_text = response_json["choices"][0]["message"]["content"]
        # extracted_json = self.extract_json_text(result_text)
        # if extracted_json is None:
        #     extracted_json = json.dumps(json.loads(result_text), ensure_ascii=False, indent=2)
        # return json.loads(extracted_json)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in analyzing ESG-related promise and evidence."},
                {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        # "image_url": {
                        #     "url": f"data:image/png;base64,{self.image_to_base64(image, scale_factor=0.2, quality=95)}"
                        # }
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                ]
                }
            ],
            functions=[{
                "name": "classify_evidence_quality",
                "description": "Classify the evidence_quality",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "evidence_quality": {
                            "type": "string",
                            "enum": ["Clear", "Not Clear", "Misleading"]
                        }
                    },
                    "required": ["evidence_quality"]
                }
            }],
            function_call={"name": "classify_evidence_quality"},
            temperature=0
        )
        
        result = json.loads(self.extract_json_text(response.choices[0].message.function_call.arguments))
        return result

    def analyze_paragraph(self, item: Dict) -> str:
        """
        4段階の分析を実行
        """
        paragraph = item['data']
        result_data = {'data': paragraph}
        
        # Step 1: Promise Status & String
        step1_result = self.classify_step1_promise(paragraph)
        result_data['promise_status'] = step1_result['promise_status']
        result_data['promise_string'] = step1_result['promise_string']
        result_data['evidence_status'] = step1_result['evidence_status']
        result_data['evidence_string'] = step1_result['evidence_string']
        
        # Step 2: Verification Timeline (if promise exists)
        if result_data['promise_status'] == 'Yes':
            step2_result = self.classify_step2_timeline(result_data['promise_string'])
            result_data['verification_timeline'] = step2_result['verification_timeline']
            
            # Step3: 根拠のクオリティの分類（画像入力を用いる；promise かつ evidence が存在する場合）
            if result_data['evidence_status'] == 'Yes':
                image_path = os.path.join("image_deim", "images_experiment", f"{item['id']}.png")
                quality_dict = self.classify_step3_quality_image(result_data['promise_string'], result_data['evidence_string'], image_path)
                result_data['evidence_quality'] = quality_dict['evidence_quality']
            else:
                result_data['evidence_quality'] = 'N/A'
        else:
            result_data['promise_string'] = ''
            result_data['verification_timeline'] = 'N/A'
            result_data['evidence_status'] = 'N/A'
            result_data['evidence_string'] = ''
            result_data['evidence_quality'] = 'N/A'
        
        # 指定された順序でデータを再構成
        ordered_data = {
            'data': result_data['data'],
            'promise_status': result_data['promise_status'],
            'promise_string': result_data['promise_string'],
            'verification_timeline': result_data['verification_timeline'],
            'evidence_status': result_data['evidence_status'],
            'evidence_string': result_data['evidence_string'],
            'evidence_quality': result_data['evidence_quality']
        }
        
        return json.dumps(ordered_data, ensure_ascii=False, indent=2)