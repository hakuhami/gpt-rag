import openai
from openai import OpenAI
import json
from typing import Dict, Optional
import yaml

class EvidenceQualityExplainer:
    def __init__(self, api_key: str, model_name: str):
        """
        Initialize the explainer with OpenAI credentials and model
        
        Args:
            api_key (str): OpenAI API key
            model_name (str): Name of the GPT model to use
        """
        openai.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def generate_explanation(self, data: Dict) -> Optional[str]:
        """
        Generate explanation for evidence quality if applicable
        
        Args:
            data (Dict): Input data containing promise, evidence and quality
            
        Returns:
            Optional[str]: Generated explanation or None if not applicable
        """
        if data.get('evidence_quality') == 'N/A':
            return None
            
        prompt = f"""
        Analyze the following ESG-related promise and evidence pair, and explain why the evidence quality is classified as {data['evidence_quality']}.
        
        Promise: {data['promise_string']}
        Evidence: {data['evidence_string']}
        Quality Classification: {data['evidence_quality']}
        
        Provide a detailed explanation of why this evidence quality classification is appropriate.
        """
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in analyzing ESG reports and evaluating the quality of evidence supporting corporate promises."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        return response.choices[0].message.content.strip()

    def process_file(self, input_path: str, output_path: str) -> None:
        """
        Process input file and add explanations where applicable
        
        Args:
            input_path (str): Path to input JSON file
            output_path (str): Path to output JSON file
        """
        # Read input data
        with open(input_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        # Process each entry
        for entry in data:
            if entry.get('evidence_quality') != 'N/A':
                explanation = self.generate_explanation(entry)
                if explanation:
                    entry['eq_explanation'] = explanation
                    print(f"\n{explanation}\n")
        
        # Write output data
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize explainer
    explainer = EvidenceQualityExplainer(
        api_key=config['openai_api_key'],
        model_name=config['model_name']
    )
    
    # Process files
    explainer.process_file(
        input_path=config['search_data_path'],
        output_path=config['eq_explanation_added_path']
    )

if __name__ == "__main__":
    main()