import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_annotated_data_from_excel
from src.rag_model import RAGModel
import yaml

def run_analysis(config_path, input_text):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load annotated data
    annotated_data = load_annotated_data_from_excel(config['data_path'])

    # Initialize RAG model
    rag_model = RAGModel(api_key=config['openai_api_key'])
    rag_model.prepare_documents(annotated_data)

    # Extract commitment and evidence
    result = rag_model.extract_commitment_and_evidence(input_text)
    print(result)

if __name__ == "__main__":
    config_path = 'config/config.yml'
    input_text = "ここにサステナビリティレポートの文章を入力してください。"
    run_analysis(config_path, input_text)