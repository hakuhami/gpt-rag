import openai
from openai import OpenAI
from typing import Optional, List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re

class RAGModel:
    def __init__(self, api_key: str, model_name: str) -> None:
        """
        Initialize the RAG model with API credentials and model configurations
        """
        openai.api_key = api_key
        self.model_name = model_name
        self.embedder = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
        self.client = OpenAI(api_key=openai.api_key)

    def prepare_documents(self, search_data: List[Dict]) -> None:
        """
        Prepare document embeddings for both promise-only and promise-evidence cases
        """
        self.search_data = search_data
        
        # データを事前に分類して保持
        self.promise_only_data = [
            item for item in search_data 
            if item.get('promise_status') == 'Yes'
        ]
        
        self.promise_evidence_data = [
            item for item in search_data 
            if item.get('promise_status') == 'Yes' and 
               item.get('evidence_status') == 'Yes' and 
               item.get('evidence_quality') != 'N/A'
        ]
        
        # 各データセットのpromise_stringの埋め込みを計算
        self.promise_only_texts = [item['promise_string'] for item in self.promise_only_data]
        self.promise_only_embeddings = self.embedder.encode(self.promise_only_texts)
        
        self.promise_evidence_texts = [item['promise_string'] for item in self.promise_evidence_data]
        self.promise_evidence_embeddings = self.embedder.encode(self.promise_evidence_texts)

    def search_similar_promise_only(self, promise_text: str) -> List[Dict]:
        """
        promise_status=Yes, evidence_status=Noの場合の類似データ検索
        verification_timelineの分類用の類似データを返す
        """
        print("!!!verification_timelineの分類用の類似データを返す!!!")
        query_embedding = self.embedder.encode([promise_text])
        similarities = cosine_similarity(query_embedding, self.promise_only_embeddings)[0]
        
        # インデックス、類似度、データの組み合わせを作成
        indexed_data = [
            (idx, sim, doc) 
            for idx, (sim, doc) in enumerate(zip(similarities, self.promise_only_data))
        ]
        
        # verification_timelineの各カテゴリごとにデータを分類
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
        
        # 各カテゴリをsimilarityでソートし、指定件数を取得
        selected_docs = []
        category_counts = {
            'already': 3,
            'within_2_years': 2,
            'between_2_and_5_years': 2,
            'more_than_5_years': 3
        }
        
        for category, count in category_counts.items():
            category_docs = timeline_categories[category]
            category_docs.sort(key=lambda x: x[1], reverse=True)
            selected_docs.extend([{
                'promise_string': doc['promise_string'],
                'verification_timeline': doc['verification_timeline']
            } for _, _, doc in category_docs[:count]])
        print("selected_docs_promise_only")
        print(f"{selected_docs}")
        print("selected_docs_promise_only")
        return selected_docs

    def search_similar_promise_evidence(self, promise_text: str) -> List[Dict]:
        """
        promise_status=Yes, evidence_status=Yesの場合の類似データ検索
        verification_timelineとevidence_qualityの分類用の類似データを返す
        """
        print("!!!verification_timelineとevidence_qualityの分類用の類似データを返す!!!")
        query_embedding = self.embedder.encode([promise_text])
        similarities = cosine_similarity(query_embedding, self.promise_evidence_embeddings)[0]
        
        # インデックス、類似度、データの組み合わせを作成
        indexed_data = [
            (idx, sim, doc) 
            for idx, (sim, doc) in enumerate(zip(similarities, self.promise_evidence_data))
        ]
        
        # evidence_qualityの各カテゴリごとにデータを分類
        quality_categories = {
            'Clear': [],
            'Not Clear': [],
            'Misleading': []
        }
        
        for idx, sim, doc in indexed_data:
            quality = doc.get('evidence_quality')
            if quality in quality_categories:
                quality_categories[quality].append((idx, sim, doc))
        
        # 各カテゴリをsimilarityでソートし、指定件数を取得
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
                'verification_timeline': doc['verification_timeline'],
                'evidence_string': doc['evidence_string'],
                'evidence_quality': doc['evidence_quality']
            } for _, _, doc in category_docs[:count]])
        print("selected_docs_promise_evidence")
        print(f"{selected_docs}")
        print("selected_docs_promise_evidence")
        return selected_docs

    def get_relevant_context(self, query: str, yes_with_evidence_count: int = 6, yes_without_evidence_count: int = 2, no_promise_count: int = 2) -> List[Dict]:
        """
        Retrieve documents related to the query, maintaining specific ratios of promise_status and evidence_status values.

        Args:
            query (str): Input query
            yes_with_evidence_count (int): Number of documents with promise_status "Yes" and evidence_status "Yes" to retrieve
            yes_without_evidence_count (int): Number of documents with promise_status "Yes" and evidence_status "No" to retrieve
            no_promise_count (int): Number of documents with promise_status "No" to retrieve

        Returns:
            List[Dict]: List of relevant documents with specified distribution of status values, excluding verification_timeline and evidence_quality
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

        # Get the corresponding documents, excluding verification_timeline and evidence_quality
        result = []
        for i, _ in all_selected:
            doc = self.search_data[i]
            filtered_doc = {
                'data': doc['data'],
                'promise_status': doc['promise_status'],
                'promise_string': doc.get('promise_string', ''),
                'evidence_status': doc.get('evidence_status', 'N/A'),
                'evidence_string': doc.get('evidence_string', '')
            }
            result.append(filtered_doc)

        return result

    def classify_promise_only(self, promise_text: str) -> Dict[str, str]:
        """
        promise_status=Yes, evidence_status=Noの場合の分類を実行
        verification_timelineのみを分類
        """
        similar_docs = self.search_similar_promise_only(promise_text)
        context = "\n".join([json.dumps(doc, ensure_ascii=False, indent=2) for doc in similar_docs])
        
        prompt = f"""
        あなたはESGに関連する企業の約束（公約）を分類する専門家です。
        提供された公約文から、その内容がいつ検証可能になるかを判断し、分類してください。

        <similar_examples>
        {context}

        <promise_to_classify>
        {promise_text}

        検証可能時期の分類基準：
        - "already": すでに検証可能な内容
        - "within_2_years": 2年以内に検証可能
        - "between_2_and_5_years": 2年から5年の間に検証可能
        - "more_than_5_years": 5年以上先でないと検証できない
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

    def classify_promise_evidence(self, promise_text: str, evidence_text: str) -> Dict[str, str]:
        """
        promise_status=Yes, evidence_status=Yesの場合の分類を実行
        verification_timelineとevidence_qualityを分類
        """
        similar_docs = self.search_similar_promise_evidence(promise_text)
        context = "\n".join([json.dumps(doc, ensure_ascii=False, indent=2) for doc in similar_docs])
        
        prompt = f"""
        あなたはESGに関連する企業の約束（公約）と根拠を分類する専門家です。
        提供された公約文と根拠文から、以下の2点を判断し分類してください：
        1. 公約の内容がいつ検証可能になるか
        2. 根拠の内容が公約をどの程度支持できているか

        <similar_examples>
        {context}

        <content_to_classify>
        公約: {promise_text}
        根拠: {evidence_text}

        分類基準：
        1. 検証可能時期（verification_timeline）：
           - "already": すでに検証可能な内容
           - "within_2_years": 2年以内に検証可能
           - "between_2_and_5_years": 2年から5年の間に検証可能
           - "more_than_5_years": 5年以上先でないと検証できない

        2. 根拠の質（evidence_quality）：
           - "Clear": 根拠が完全で、理解しやすく、論理的
           - "Not Clear": 根拠に不足がある、または表面的
           - "Misleading": 根拠が不適切、無関係、または誤解を招く可能性がある
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in classifying ESG-related promises and evidence."},
                {"role": "user", "content": prompt}
            ],
            functions=[{
                "name": "classify_verification_timeline_and_evidence_quality",
                "description": "Classify the verification_timeline and evidence_quality",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "verification_timeline": {
                            "type": "string",
                            "enum": ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years"]
                        },
                        "evidence_quality": {
                            "type": "string",
                            "enum": ["Clear", "Not Clear", "Misleading"]
                        }
                    },
                    "required": ["verification_timeline", "evidence_quality"]
                }
            }],
            function_call={"name": "classify_verification_timeline_and_evidence_quality"},
            temperature=0
        )
        
        result = json.loads(self.extract_json_text(response.choices[0].message.function_call.arguments))
        return result

    def extract_json_text(self, text: str) -> Optional[str]:
        """JSONテキストを抽出"""
        json_pattern = re.compile(r'\{[^{}]*\}')
        matches = json_pattern.findall(text)
        
        if matches:
            try:
                json_obj = json.loads(matches[0])
                return json.dumps(json_obj, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                pass        
        return None

    def analyze_paragraph(self, paragraph: str) -> str:
        """
        段落を分析し、抽出と分類を実行
        """
        # 既存の抽出処理を実行
        relevant_docs = self.get_relevant_context(paragraph)
        context = "\n".join([json.dumps(doc, ensure_ascii=False, indent=2) for doc in relevant_docs])
        
        extraction_prompt = f"""
        [既存の抽出用プロンプト]
        """
        
        # 抽出結果を取得
        extraction_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in extracting ESG-related promise and their corresponding evidence from corporate reports that describe ESG matters."},
                {"role": "user", "content": extraction_prompt}
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
        
        extracted_data = json.loads(self.extract_json_text(extraction_response.choices[0].message.function_call.arguments))
        
        # promise_statusとevidence_statusに応じて分類を実行
        if extracted_data['promise_status'] == 'Yes':
            if extracted_data['evidence_status'] == 'Yes':
                # 両方Yesの場合の分類を実行
                classification_result = self.classify_promise_evidence(
                    extracted_data['promise_string'],
                    extracted_data['evidence_string']
                )
                extracted_data['verification_timeline'] = classification_result['verification_timeline']
                extracted_data['evidence_quality'] = classification_result['evidence_quality']
                
            elif extracted_data['evidence_status'] == 'No':
                # Promiseのみの場合の分類を実行
                classification_result = self.classify_promise_only(
                    extracted_data['promise_string']
                )
                extracted_data['verification_timeline'] = classification_result['verification_timeline']
                extracted_data['evidence_quality'] = 'N/A'
        else:
            extracted_data['verification_timeline'] = 'N/A'
            extracted_data['evidence_status'] = 'N/A'
        
        return json.dumps(extracted_data, ensure_ascii=False, indent=2)