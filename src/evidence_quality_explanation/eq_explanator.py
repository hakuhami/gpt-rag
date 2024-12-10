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
        You are an expert in generating ESG-related documents and reading corporate ESG-related documents.
        Do document generation task for ESG-related corporate texts.
        Carefully consider the detailed task explanation and reference examples step-by-step before proceeding with the task.
        The content is provided under five tags: <task description>, <task steps>, <promise_string>, <evidence_string>, and <evidence_quality>.
        
        <task description>
        You will be given two extracted texts from a section of company documentation: an ESG promise statement and an evidence statement.
        The promise statement is given in the <promise_string> below, and the evidence statement is provided in the <evidence_string> below.
        Additionally, a evaluation item will be given that evaluates the extent to which the contents of the <evidence_string> supports the contents of the <promise_string>.
        This evaluation item is given as the <evidence_quality> below.
        The <evidence_quality> item has three values: Clear, Not Clear, and Misleading.
        The definition of each value is as follows:
        "Clear": In the content of "evidence_string", there is no lack of information and what is said is intelligible and logical.
        "Not Clear": In the content of "evidence_string", some information is missing or not well described so that what is said may range from intelligible and logical to superficial and/or superfluous.           
        "Misleading": In the content of "evidence_string", it is not suitable to support the promise, or is not relevant to the contents of the promise, or may distract readers, or is untrue.
        According to <task steps> below, generate a sentence explaining why <evidence_quality> has the value it does.
        Output only the sentences specified in the <task steps> below, and do not output other strings.
        
        <task steps>
        Follow the steps below to do the task step by step.
        Use natural language to think at each step.
        Step1: Carefully read and understand each of the sentences in <promise_string> and <evidence_string>.
        Step2: Considering the sentences in <promise_string> and <evidence_string>, think carefully about why the value of <evidence_quality> is evaluated to this value.
        Step3: Think of a logical and easy-to-understand explanation of why <evidence_quality> has the value it does.
        Step4: Output the explanation you thought of in the Step 3.
        
        <promise_string>
        {data['promise_string']}
        
        <evidnece_string>
        {data['evidence_string']}
        
        <evidence_quality>
        {data['evidence_quality']}
        """
                
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in analyzing ESG reports and evaluating the quality of evidence supporting corporate promises."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        print(f"Generating explanation for promise: {data['promise_string']}/// and evidence: {data['evidence_string']}/// with quality: {data['evidence_quality']}")
        print(f"Response: {response.choices[0].message.content.strip()}")
        
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