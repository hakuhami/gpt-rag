import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from typing import Optional, Dict, List
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
        try:
            # モデル設定の最適化
            config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=3,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                classifier_dropout=None
            )
            
            # モデルの初期化を改善
            base_model = AutoModel.from_pretrained(self.model_name)
            
            class CustomRobertaClassifier(nn.Module):
                def __init__(self, base_model, num_labels):
                    super().__init__()
                    self.roberta = base_model
                    self.dropout = nn.Dropout(0.1)
                    self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)
                    
                    # 分類層の重みを適切に初期化
                    torch.nn.init.xavier_normal_(self.classifier.weight)
                    self.classifier.bias.data.zero_()
                
                def forward(self, input_ids, attention_mask):
                    outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
                    pooled_output = outputs.last_hidden_state[:, 0, :]
                    pooled_output = self.dropout(pooled_output)
                    logits = self.classifier(pooled_output)
                    return logits
            
            self.model = CustomRobertaClassifier(base_model, num_labels=3).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
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
                padding="max_length",
                return_attention_mask=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.cuda.amp.autocast(enabled=False):
                with torch.no_grad():
                    # モデルからの出力を直接取得
                    logits = self.model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                    )
                    
                    # logitsテンソルに対して直接softmaxを適用
                    probabilities = torch.softmax(logits, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence_score = probabilities[0][predicted_class].item()
            
                quality_mapping = {
                    0: QualityLabel.CLEAR,
                    1: QualityLabel.NOT_CLEAR,
                    2: QualityLabel.MISLEADING
                }
                
                torch.cuda.empty_cache()
                
                return ClassificationResult(
                    quality_label=quality_mapping[predicted_class],
                    confidence_score=confidence_score
                )
                
        except Exception as e:
            torch.cuda.empty_cache()
            raise RuntimeError(f"Classification failed: {e}")
  