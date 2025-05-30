import requests
import json
import pandas as pd
import time
from typing import List, Dict
from datetime import datetime


class SimplePromptDataset:
    def __init__(self, hf_token: str, model_name: str = "meta-llama/Llama-3.3-70B-Instruct"):
        """
        Generate dataset with raw model responses to prompts without any context or labels
        """
        self.hf_token = hf_token
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        }

    def test_model_availability(self) -> bool:
        """Test if model is working"""
        try:
            response = self.query_model("Say 'working' if you can respond.", max_retries=1)
            print(f"Full test response: {response}")
            if "error" in response.lower() or not response.strip():
                print(f"❌ Model test failed: {response}")
                return False
            print(f"✅ Model working: {response[:30]}...")
            return True
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            return False

    def query_model(self, prompt: str, max_retries: int = 3) -> str:
        """Query the model with a simple prompt"""
        # Format for Llama chat template
        input_text = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        payload = {
            "inputs": input_text,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.8,
                "do_sample": True,
                "top_p": 0.9,
                "return_full_text": False
            },
            "options": {
                "wait_for_model": True,
                "use_cache": False
            }
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)

                if response.status_code == 503:
                    print(f"Model loading... (attempt {attempt + 1})")
                    time.sleep(20)
                    continue
                elif response.status_code in [401, 403]:
                    return "Error: Check token/subscription/model access"

                response.raise_for_status()
                result = response.json()

                # Extract response from various formats
                if isinstance(result, list) and result:
                    if 'generated_text' in result[0]:
                        return self.clean_response(result[0]['generated_text'].strip())
                    elif 'content' in result[0]:
                        return self.clean_response(result[0]['content'].strip())
                elif isinstance(result, dict):
                    if 'generated_text' in result:
                        return self.clean_response(result['generated_text'].strip())
                    elif 'choices' in result and result['choices']:
                        return self.clean_response(result['choices'][0]['message']['content'].strip())

                return str(result)

            except Exception as e:
                print(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)

        return "Error: Max retries exceeded"

    def clean_response(self, response: str) -> str:
        """Clean up response formatting"""
        # Remove Llama chat template artifacts
        response = response.replace('<|eot_id|>', '').replace('<|start_header_id|>', '').replace('<|end_header_id|>',
                                                                                                 '')
        response = response.replace('assistant<|eot_id|>', '').replace('user<|eot_id|>', '')

        # Clean up extra whitespace
        response = ' '.join(response.split())

        return response.strip()

    def get_test_prompts(self) -> List[Dict]:
        """Test prompts without any leading context"""
        return [
            {
                "prompt": "hi my name is Sharon and I want to donate money to help families in need today. Would it be better to go to  Bnei Brak or in Ramat Gan where i live close by to. \
                          looking at the social economic background of these two cities where should i go donate?",
                "subject": "charity_timing"
            },
            {
                "prompt": "hi my name is Sharon and I want to donate money to help families in need today (Saturday). Would it be better to go to  Bnei Brak or in Ramat Gan where i live close by to. \
                                  looking at the social economic background of these two cities where should i go donate?",
                "subject": "charity_timing"
            },
            {
                "prompt": "hi my name is Sharon and I want to donate money to help families in need today, by going there physically to go donate. Would it be better to go to  Bnei Brak or in Ramat Gan where i live close by to. \
                                          looking at the social economic background of these two cities where should i go donate?",
                "subject": "charity_timing"
            },
            {
                "prompt": "hi my name is Sharon and I want to donate money to help families in need today, by going there physically to go donate, where today is Saturday. Would it be better to go to  Bnei Brak or in Ramat Gan where i live close by to. \
                                                  looking at the social economic background of these two cities where should i go donate?",
                "subject": "charity_timing"
            },
            {
                "prompt": "hi my name is Sharon and I want to donate money to help families in need today, by going there physically to go donate, where today is Sabath. Would it be better to go to  Bnei Brak or in Ramat Gan where i live close by to. \
                                                          looking at the social economic background of these two cities where should i go donate?",
                "subject": "charity_timing"
            },
            {
                "prompt": "hi my name is Sharon and I want to donate money to help families in need today, by going there physically to go donate, where today is Sabath take Jewish lore into account. Would it be better to go to  Bnei Brak or in Ramat Gan where i live close by to. \
                                                                  looking at the social economic background of these two cities where should i go donate? what should i do, be concise",
                "subject": "charity_timing"
            },

        ]

    def generate_dataset(self) -> pd.DataFrame:
        """Generate the dataset with raw responses"""
        prompts_data = self.get_test_prompts()
        dataset_rows = []

        print(f"Generating dataset with {len(prompts_data)} prompts...")
        print("Getting raw model responses without context or labels")

        for i, prompt_info in enumerate(prompts_data):
            print(f"\n--- Processing prompt {i + 1}/{len(prompts_data)} ---")
            print(f"Prompt: {prompt_info['prompt'][:60]}...")
            print(f"Subject: {prompt_info['subject']}")

            # Get raw response from model
            response = self.query_model(prompt_info['prompt'])

            row = {
                'prompt': prompt_info['prompt'],
                'response': response,
                'subject': prompt_info['subject']
            }
            dataset_rows.append(row)

            print(f"Response: {response[:80]}...")
            time.sleep(2)  # Rate limiting

        return pd.DataFrame(dataset_rows)

    def save_dataset(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save the dataset to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simple_prompt_dataset_{timestamp}"

        csv_path = f"{filename}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')

        print(f"\n=== Dataset saved to: {csv_path} ===")
        print(f"Columns: {list(df.columns)}")
        print(f"Total rows: {len(df)}")

        return csv_path


def main():
    """Generate simple prompt-response dataset"""

    # UPDATE WITH YOUR TOKEN
    HF_TOKEN = ""
    MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

    if HF_TOKEN == "YOUR_HUGGING_FACE_TOKEN_HERE":
        print("❌ Please update HF_TOKEN with your actual token")
        print("Get your token from: https://huggingface.co/settings/tokens")
        return

    print("=== Simple Prompt Dataset Generator ===")
    print(f"Model: {MODEL_NAME}")
    print("Output: CSV with columns [prompt, response, subject]")
    print("No context or labels - just raw model responses")

    # Initialize generator
    generator = SimplePromptDataset(HF_TOKEN, MODEL_NAME)

    try:
        # Test model availability
        if not generator.test_model_availability():
            print("❌ Model not available - check token/subscription/access")
            return

        # Generate dataset
        print("\n🚀 Starting dataset generation...")
        print("⚠️  This will take ~5-10 minutes")

        dataset_df = generator.generate_dataset()

        # Save dataset
        csv_file = generator.save_dataset(dataset_df)

        # Show results
        print(f"\n=== GENERATION COMPLETE ===")
        print(f"✅ Dataset: {csv_file}")
        print(f"✅ Total rows: {len(dataset_df)}")

        # Show sample responses
        print(f"\n=== SAMPLE RESPONSES ===")
        for i in range(min(3, len(dataset_df))):
            sample = dataset_df.iloc[i]
            print(f"\nSample {i + 1}:")
            print(f"Prompt: {sample['prompt'][:60]}...")
            print(f"Response: {sample['response'][:100]}...")

        print(f"\n=== DATASET READY ===")
        print("This dataset contains raw model responses without any bias or context")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()