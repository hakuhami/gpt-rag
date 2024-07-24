import openai
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# embedding modelには、多言語に対応したmultilingual-e5-large-instructを使用

class RAGModel:
    def __init__(self, api_key: str, model_name: str = "gpt-4"):
        openai.api_key = api_key
        self.model_name = model_name
        self.embedder = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

    def prepare_documents(self, search_data: List[Dict]) -> None:
        """
        検索用データを準備し、エンコードする

        Args:
            search_data (List[Dict]): 検索用データ
        """
        self.search_data = search_data
        self.documents = [item['paragraph'] for item in search_data]
        self.doc_embeddings = self.embedder.encode(self.documents)

    # とりあえず上位5個のデータを選んだが、これは本当に適当か？動的な選び方の方が良い？
    def get_relevant_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        クエリに関連する上位のドキュメントを取得する

        Args:
            query (str): 入力クエリ
            top_k (int): 取得するドキュメント数

        Returns:
            List[Dict]: 関連するドキュメントのリスト
        """
        query_embedding = self.embedder.encode([query])
        similarities = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.search_data[i] for i in top_indices]

    #プロンプトは英語で書き直す
    def analyze_paragraph(self, paragraph: str) -> Dict:
        """
        パラグラフを分析し、公約と根拠を抽出する

        Args:
            paragraph (str): 分析対象のパラグラフ

        Returns:
            Dict: 分析結果
        """
        relevant_docs = self.get_relevant_context(paragraph)
        context = "\n".join([json.dumps(doc, ensure_ascii=False) for doc in relevant_docs])

        prompt = f"""
        あなたは、ESGに関して記述した企業のレポートから、ESGに関する公約とそれに対応する根拠を抽出する専門家です。
        以下の形式でアノテーションを行ってください。
        なお、結果は以下のJSON形式で出力してください：
        {{
            "promise_status": str,
            "promise_string": str or null,
            "verification_timeline": str or null,
            "evidence_status": str or null,
            "evidence_string": str or null,
            "evidence_quality": str or null
        }}：
        （アノテーションの形式）
        1. パラグラフ中の文章内容が与えられます。
        2. 公約が含まれるかどうかを判断し、含まれる場合はYes、含まれない場合はNoと表記してください。(promise_status)
        3. 公約が含まれる場合（2がYesの場合）、以下の情報も提供してください：
           - 公約の具体的な箇所（文章から一言一句変えずにそのまま抽出）(promise_string)
           - 公約を検証できるタイミング（already, within_2_years, between_2_and_5_years, more_than_5_years）(verification_timeline)
           - 根拠が含まれるかどうか（Yes, No）(evidence_status)
        4. 根拠が含まれる場合evidence_statusがYesの場合）、以下の情報も提供してください：
           - 根拠の箇所（文章から一言一句変えずに直接抽出）(evidence_string)
           - 公約と根拠の関係の質（Clear, Not Clear, Misleading）(evidence_quality)

        以下は、参考となる既存のアノテーション例です：
        {context}

        次の文章を分析し、上記の形式で結果を提供してください：
        {paragraph}
        """

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in extracting commitments and evidence from sustainability reports."},
                {"role": "user", "content": prompt}
            ]
        )

        result = response.choices[0].message['content']
        return json.loads(result)