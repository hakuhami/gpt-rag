import openai
from openai import OpenAI
import json
from typing import Dict, Optional
import yaml

class AllRabelExplainer:
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
        Generate explanation for all train rabel
        
        Args:
            data (Dict): Input data
            
        Returns:
            Optional[str]: Generated explanation or None if not applicable
        """
            
        prompt = f"""
        You are an expert in reading and generating ESG-related documents.
        Read the provided test data that extracts and classifies promise and evidence from ESG-related corporate documents, and then explain the features of extraction and classification according to the specified format.
        Carefully consider the detailed task explanation and step-by-step before proceeding with the task.
        The content is provided under four tags: <input json format>, <output json format>, <the details of the task>, and <test data>.
        
        
        <input json format>
        The test data is in the json format below.
        
        {{
            "data": str,
            "promise_status": str,
            "promise_string": str or null,
            "verification_timeline": str,
            "evidence_status": str,
            "evidence_string": str or null,
            "evidence_quality": str
        }}:
        
        Explanation of each label:
        "data": Text that is target to extraction and classification task.
        "promise_status": Classification result of whether the text contains the contents that are considered to be promise. (Yes/No)
        "promise_string": Extraction result of the promise from the text. (verbatim from the text)
        "verification_timeline": Classification result of the time when the promise can be verified. (already/within_2_years/between_2_and_5_years/more_than_5_years/N/A)
        "evidence_status": Classification result of whether the text contains the contents that are considered to be evidence supporting the promise. (Yes/No)
        "evidence_string": Extraction result of the evidence supporting the promise from the text. (verbatim from the text)
        "evidence_quality": Classification result of the relationship between the promise and the evidence. (Clear/Not Clear/Misleading/N/A)
        
        
        <output json format>
        Output explanations of the features of extraction and classification according to the json format below.
        Be sure to explain step by step in logical Japanese sentences.
        
        {{
            "promise_explanation": str,
            "verification_timeline_explanation": str,
            "evidence_explanation": str,
            "evidence_quality_explanation": str
        }}:
        
        Explanation of each label:
        "promise_explanation": Explanation of the features of the classification result of "promise_status" and the extraction result of "promise_string".
        "verification_timeline_explanation": Explanation of the features of the classification result of "verification_timeline".
        "evidence_explanation": Explanation of the features of the classification result of "evidence_status" and the extraction result of "evidence_string".
        "evidence_quality_explanation": Explanation of the features of the classification result of "evidence_quality".
        
        
        <the details of the task>
        Task Steps:
        1. Carefully read the text in the "data" in the test data that is target of extraction and classification.
        2. About "promise_explanation":
           If "promise_status" is "Yes", explain why it can be judged that a promise exists and why "promise_string" is the text.
           If "promise_status" is "No", explain why it can be judged that a promise does not exist.
        3. About "verification_timeline_explanation":
           If "promise_status" is "Yes", after carefully reading the contents of "promise_string", explain why "verification_timeline" is classified as that value.
           If "promise_status" is "No" ("verification_timeline" is "N/A"), output a blank.
        4. About "evidence_explanation":
           If "evidence_status" is "Yes", explain why it can be judged that a evidence exists and why "evidence_string" is the text.
           If "evidence_status" is "No", explain why it can be judged that a evidence does not exist.
           If "evidence_status" is "N/A", output a blank.
        5. About "evidence_quality_explanation":
           If "evidence_status" is "Yes", after carefully reading the contents of "promise_string" and "evidence_string", explain why "evidence_quality" is classified as that value.
           If "evidence_status" is "No" or "N/A", output a blank.
           
        Definitions of each label and the thought process behind the task:
        1. After carefully considering whether a promise exists and, if so, whether there is evidence, carefully read and understand the text in the "data".
        2. In this task, "promise" is expressed as expressions such as a company's ESG-related "corporate philosophy," "commitments being implemented or planned," "strategies for the future," and "statements for the future."
           Based on what you learned in step 1, and taking these concepts into account, think carefully about the reasons for the classification results of "promise_status" and the extraction results of "promise_string".
           Then explain those reasons step by step in logical Japanese sentences.
        3. In this task, the value of "verification_timeline" follows the definitions below.
           Based on what you learned in step 1 and the definitions below, think carefully about the reasons for the classification results of "verification_timeline".
           Then explain those reasons step by step in logical Japanese sentences.
           <the definitions>
           "already": When the promise have already been applied, or whether or not it is applied, can already be verified.
           "within_2_years": When the promise can be verified within 2 years. (When the promise can be verified in the near future.)
           "between_2_and_5_years": When the promise can be verified in 2 to 5 years. (When the promise can be verified in the not too distant future, though not in the near future.)
           "more_than_5_years: When the promise can be verified in more than 5 years. (When the promsie can be verified in the distant future.)
        4. In this task, "evidence" is expressed as "specific examples of the contents of the promise," "detailed explanation of the contents of the promise," "current status of the contents of the promise," etc.
           Based on what you learned in step 1, and taking these concepts into account, think carefully about the reasons for the classification results of "evidence_status" and the extraction results of "evidence_string".
           Then explain those reasons step by step in logical Japanese sentences.
        5. In this task, the value of "evidence_quality" follows the definitions below.
           Based on what you learned in step 1 and the definitions below, think carefully about the reasons for the classification results of "evidence_quality".
           Then explain those reasons step by step in logical Japanese sentences.
           <the definitions>
           "Clear": In the content of "evidence_string", there is no lack of information and what is said is intelligible and logical.
           "Not Clear": In the content of "evidence_string", some information is missing or not well described so that what is said may range from intelligible and logical to superficial and/or superfluous.           
           "Misleading": In the content of "evidence_string", it is not suitable to support the promise, or is not relevant to the contents of the promise, or may distract readers, or is untrue.
          
        Important notes:
        You must output the results in the format specified by <output json format>, but the thought process described above is carried out step by step using natural language, and then the reasoning results in natural language are output in <output json format>.
        Consider the context and logical relationships of the sentences thoroughly. It's important to understand the meaning of the entire paragraph, not just individual sentences.
        Concepts specific to each company or industry may appear in the text, so think carefully about their meaning and appropriately interpret them.
        
        
        <test_data>
        
        {data}
        """
                
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in reading and generating ESG-related documents."},
                {"role": "user", "content": prompt}
            ],
            functions=[
                {
                    "name": "explain_esg_paragraph",
                    "description": "Explanation the features of extraction and classification",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "promise_explanation": {"type": "string"},
                            "verification_timeline_explanation": {"type": "string"},
                            "evidence_explanation": {"type": "string"},
                            "evidence_quality_explanation": {"type": "string"}
                        },
                        "required": ["promise_explanation", "verification_timeline_explanation", "evidence_explanation", "evidence_quality_explanation"]
                    }
                }
            ],
            function_call={"name": "explain_esg_paragraph"},
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
        
        # # Process each entry
        # for entry in data:
        #     if entry.get('evidence_quality') != 'N/A':
        #         explanation = self.generate_explanation(entry)
        #         if explanation:
        #             entry['eq_explanation'] = explanation
        #             print(f"\n{explanation}\n")
        
        # Process each entry
        for entry in data:
            print(f"入力のjsonデータ:::\n{entry}\n")
            explanation = self.generate_explanation(entry)

            if entry['promise_status'] == 'Yes':
                if entry['evidence_status'] == 'Yes':
                  # 公約も根拠もある場合
                    entry['explanation'] = explanation
                    print(f"出力の説明文:::\n{explanation}\n")
                elif entry['evidence_status'] == 'No':
                  # 公約はあるが根拠が無い場合
                    del explanation['evidence_quality_explanation']
                    entry['explanation'] = explanation
                    print(f"出力の説明文:::\n{explanation}\n")
            elif entry['evidence_status'] == 'No':
              # 公約も根拠もない場合
                del explanation['verification_timeline_explanation']
                del explanation['evidence_explanation']
                del explanation['evidence_quality_explanation']
                entry['explanation'] = explanation
                print(f"出力の説明文:::\n{explanation}\n")           
        
        # Write output data
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    # Load configuration
    with open('config/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize explainer
    explainer = AllRabelExplainer(
        api_key=config['openai_api_key'],
        model_name=config['model_name']
    )
    
    # Process files
    explainer.process_file(
        input_path=config['search_data_path'],
        output_path=config['all_rabel_explanation_added_path']
    )

if __name__ == "__main__":
    main()
