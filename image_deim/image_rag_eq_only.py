### 根拠の性質以外は使いまわせるため、ベースラインの結果を利用して、根拠の性質のみイメージベースで分類する

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
        
        # # Step 1の検索用データ
        # self.documents = [item['data'] for item in search_data]
        # self.doc_embeddings = self.embedder.encode(self.documents)
        
        # # Step 2の検索用データ
        # self.promise_only_data = [
        #     item for item in search_data 
        #     if item.get('promise_status') == 'Yes'
        # ]
        # self.promise_only_texts = [item['promise_string'] for item in self.promise_only_data]
        # self.promise_only_embeddings = self.embedder.encode(self.promise_only_texts)
        
        # Step 3の検索用データ
        self.quality_data = [
            item for item in search_data 
            if item.get('promise_status') == 'Yes' and 
               item.get('evidence_status') == 'Yes' and 
               item.get('evidence_quality') != 'N/A'
        ]
        self.quality_texts = [item['promise_string'] for item in self.quality_data]
        self.quality_embeddings = self.embedder.encode(self.quality_texts)

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
        # paragraph = item['data']
        # result_data = {'data': paragraph}
        
        # # Step 1: Promise Status & String
        # step1_result = self.classify_step1_promise(paragraph)
        # result_data['promise_status'] = step1_result['promise_status']
        # result_data['promise_string'] = step1_result['promise_string']
        # result_data['evidence_status'] = step1_result['evidence_status']
        # result_data['evidence_string'] = step1_result['evidence_string']
        
        # Step 2: Verification Timeline (if promise exists)
        if item['promise_status'] == 'Yes':
            # step2_result = self.classify_step2_timeline(result_data['promise_string'])
            # result_data['verification_timeline'] = step2_result['verification_timeline']
            
            # Step3: 根拠のクオリティの分類（画像入力を用いる；promise かつ evidence が存在する場合）
            if item['evidence_status'] == 'Yes':
                image_path = os.path.join("image_deim", "images_experiment", f"{item['id']}.png")
                quality_dict = self.classify_step3_quality_image(item['promise_string'], item['evidence_string'], image_path)
                # result_data['evidence_quality'] = quality_dict['evidence_quality']
            else:
                item['evidence_quality'] = 'N/A'
        # else:
        #     result_data['promise_string'] = ''
        #     result_data['verification_timeline'] = 'N/A'
        #     result_data['evidence_status'] = 'N/A'
        #     result_data['evidence_string'] = ''
        #     result_data['evidence_quality'] = 'N/A'
        
        else:
            quality_dict = {'evidence_quality': 'N/A'}
        
        # 指定された順序でデータを再構成
        ordered_data = {
            'data': item['data'],
            'promise_status': item['promise_status'],
            'promise_string': item['promise_string'],
            'verification_timeline': item['verification_timeline'],
            'evidence_status': item['evidence_status'],
            'evidence_string': item['evidence_string'],
            'evidence_quality': quality_dict['evidence_quality']
        }
        
        return json.dumps(ordered_data, ensure_ascii=False, indent=2)
