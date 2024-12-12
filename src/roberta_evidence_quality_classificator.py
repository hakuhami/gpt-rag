import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Optional, Dict, List
import json
from dataclasses import dataclass
from enum import Enum

class QualityLabel(str, Enum):
    CLEAR = "Clear"
    NOT_CLEAR = "Not Clear"
    MISLEADING = "Misleading"
    NOT_APPLICABLE = "N/A"

@dataclass
class ClassificationResult:
    quality_label: QualityLabel
    confidence_score: float
    attention_scores: Optional[Dict[str, float]] = None

class ESGQualityClassifier:
    """
    プロンプトエンジニアリングを活用したESG公約・根拠の質評価分類器
    Waseda RoBERTaモデルを使用して、根拠の質を評価します
    """
    
    def __init__(
        self,
        gpu_required: bool = True,
        model_name: str = "nlp-waseda/roberta-large-japanese",
        max_length: int = 512
    ):
        """
        分類器の初期化
        
        Args:
            gpu_required (bool): GPUが必須かどうか
            model_name (str): 使用するモデルの名前
            max_length (int): 入力テキストの最大長
        """
        if gpu_required and not torch.cuda.is_available():
            raise RuntimeError("GPU is required but not available.")
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.max_length = max_length
        self.prompt_template = self._create_prompt_template()
        self._initialize_model()
        
    def _create_prompt_template(self) -> Dict[str, str]:
        """
        評価に使用するプロンプトテンプレートを定義
        """
        return {
            'context': "以下の2つの文章は、企業のESGレポートから抽出された公約と、それを裏付ける根拠です。",
            'task_description': "根拠の内容が公約の内容をどの程度支持できているか、評価してください。",
            'evaluation_criteria': """
                評価基準:
                - Clear: 根拠が公約を明確に支持し、情報不足も無く述べられている内容は分かりやすくかつ論理的
                - Not Clear: 根拠は公約を支持してはいるが、一部の情報が欠落しているか、十分に説明されていないため、述べられている内容は分かりやすく論理的なものから、表面的または余分なものまで、さまざまなものが含まれる
                - Misleading: 根拠が公約を裏付けるのに不適切であったり、公約の内容と関係がなかったり、読者の注意をそらす恐れがあったり、真実でなかったりする場合
            """,
            'input_format': "公約：{promise}\n根拠：{evidence}",
            'instruction': "上記の根拠の内容が、上記の公約の内容をどの程度支持できているか、評価してください。"
        }
    
    def _initialize_model(self) -> None:
        """
        モデルとトークナイザーの初期化と最適化設定
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=3,
                output_attentions=True,
                torch_dtype=torch.float32  # 明示的に32ビット浮動小数点精度を指定
            ).to(self.device)
            
            # モデルの最適化設定
            self.model.config.use_cache = False
            
            # 演算精度の設定を調整
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False  # ベンチマークを無効化
            torch.backends.cudnn.deterministic = True  # 決定論的な動作を保証
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {e}")
    
    def _format_input_text(self, promise_string: str, evidence_string: str) -> str:
        """
        入力テキストを構造化されたプロンプトに変換
        
        Args:
            promise_string (str): 公約テキスト
            evidence_string (str): 根拠テキスト
            
        Returns:
            str: 構造化されたプロンプト
        """
        template = self.prompt_template
        formatted_text = (
            f"{template['context']}\n"
            f"{template['task_description']}\n"
            f"{template['evaluation_criteria']}\n"
            f"{template['input_format'].format(promise=promise_string, evidence=evidence_string)}\n"
            f"{template['instruction']}"
        )
        return formatted_text
    
    def _get_attention_highlights(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        threshold: float = 0.1
    ) -> Dict[str, float]:
        """
        重要な注目箇所とそのスコアを抽出
        
        Args:
            attention_weights: アテンションの重み
            tokens: 入力トークン
            threshold: 重要と判断する閾値
            
        Returns:
            Dict[str, float]: 重要箇所とそのスコア
        """
        # 最後の層の[CLS]トークンのアテンションスコアを使用
        cls_attention = attention_weights[-1][0][0].mean(dim=0)
        important_tokens = {}
        
        for token, score in zip(tokens, cls_attention):
            if score > threshold:
                important_tokens[token] = float(score)
        
        return dict(sorted(important_tokens.items(), key=lambda x: x[1], reverse=True))
    
    def classify_evidence_quality(self, promise_string: str, evidence_string: str) -> ClassificationResult:
        if not promise_string or not evidence_string:
            return ClassificationResult(
                quality_label=QualityLabel.NOT_APPLICABLE,
                confidence_score=0.0
            )
            
        formatted_text = self._format_input_text(promise_string, evidence_string)
        
        try:
            inputs = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding="max_length",  # パディングを明示的に指定
                return_attention_mask=True
            )
            
            # デバイスへの転送を最適化
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # メモリ効率の良い推論処理
            with torch.cuda.amp.autocast(enabled=False):  # 混合精度を無効化
                with torch.no_grad():
                    # バッチサイズを1に固定
                    outputs = self.model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )
                    
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence_score = probabilities[0][predicted_class].item()
            
            quality_mapping = {
                0: QualityLabel.CLEAR,
                1: QualityLabel.NOT_CLEAR,
                2: QualityLabel.MISLEADING
            }
            
            # 明示的にキャッシュをクリア
            torch.cuda.empty_cache()
            
            return ClassificationResult(
                quality_label=quality_mapping[predicted_class],
                confidence_score=confidence_score
            )
            
        except Exception as e:
            torch.cuda.empty_cache()  # エラー時もキャッシュをクリア
            raise RuntimeError(f"Classification failed: {e}")
  