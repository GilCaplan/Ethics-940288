#!/usr/bin/env python3
"""
Simple script to run Jewish/Israeli Legal NER Analysis
No imports needed - everything is self-contained
"""

# Step 1: Copy the main analysis code here (or save it as jewish_legal_ner.py)
# Step 2: Just run this script with your CSV file

import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
from typing import List, Dict, Any
import json
import warnings
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')
from entities import *

class JewishLegalNERAnalyzer:
    def __init__(self, hf_token=None):
        self.models = {}
        self.spacy_model = None
        ALL_ENTITIES = {**JEWISH_LEGAL_ENTITIES, **ADDITIONAL_LIABILITY_ENTITIES, **LIABILITY_ENHANCING_PATTERNS}
        self.entity_categories = ALL_ENTITIES

        # Get token from parameter or environment variable
        self.hf_token = hf_token or os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
        if self.hf_token:
            print("✓ Using Hugging Face token from environment/parameter")

        self.setup_models()

    def setup_models(self):
        """Download and setup NER models from Hugging Face with progress tracking"""
        print("Setting up NER models...")
        print("These models are PUBLIC and should not require a Hugging Face token")
        print("📦 First run will download ~500MB-1GB of models (cached for future use)")

        # PUBLIC NER models that don't require tokens
        model_configs = [
            ("basic_ner", "dslim/bert-base-NER", "~400MB"),
            ("multilingual_ner", "dbmdz/bert-large-cased-finetuned-conll03-english", "~1.2GB"),
        ]

        for model_name, model_identifier, size_info in model_configs:
            try:
                print(f"\n📥 Downloading {model_name} ({size_info})")
                print(f"   Source: {model_identifier}")

                kwargs = {}
                if self.hf_token:
                    kwargs['use_auth_token'] = self.hf_token

                print("   ⬇️  Downloading tokenizer...")
                with tqdm(desc="Tokenizer", unit="file") as pbar:
                    tokenizer = AutoTokenizer.from_pretrained(model_identifier, **kwargs)
                    pbar.update(1)

                print("   ⬇️  Downloading model weights...")
                try:
                    with tqdm(desc="Model", unit="file") as pbar:
                        model = AutoModelForTokenClassification.from_pretrained(model_identifier, **kwargs)
                        pbar.update(1)
                    print(f"   ✅ {model_name} downloaded successfully")
                except Exception as model_error:
                    print(f"   ❌ {model_name} is not a NER model: {model_error}")
                    continue

                print("   🔧 Creating NER pipeline...")
                with tqdm(desc="Pipeline", unit="step") as pbar:
                    self.models[model_name] = pipeline(
                        "ner",
                        model=model,
                        tokenizer=tokenizer,
                        aggregation_strategy="simple",
                        device=0 if torch.cuda.is_available() else -1
                    )
                    pbar.update(1)

                print(f"   ✅ {model_name} ready for analysis!")
                break

            except Exception as e:
                print(f"   ❌ Failed to load {model_name}: {e}")
                continue

        if not self.models:
            try:
                print("\n📥 Attempting to use spaCy as fallback...")
                import spacy
                with tqdm(desc="Loading spaCy", unit="model") as pbar:
                    nlp = spacy.load("en_core_web_sm")
                    pbar.update(1)
                self.spacy_model = nlp
                print("   ✅ spaCy model loaded as fallback")
            except Exception as spacy_error:
                print("   ⚠️  No NER models available. Will use regex-only analysis.")
                self.spacy_model = None

    def custom_entity_matching(self, text: str) -> Dict[str, List[str]]:
        """Match custom Jewish/Israeli legal entities using regex"""
        found_entities = {}
        text_lower = text.lower()

        for category, terms in self.entity_categories.items():
            found_entities[category] = []
            for term in terms:
                pattern = r'\b' + re.escape(term.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    found_entities[category].append(term)

        found_entities = {k: v for k, v in found_entities.items() if v}
        return found_entities

    def calculate_liability_risk(self, custom_entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Calculate civil liability risk based on detected entities"""
        liability_score = 0
        risk_factors = []

        scoring_rules = {
            'LIABILITY_INDICATORS': 35,
            'AI_LIABILITY': 30,
            'JEWISH_LAW': 20,
            'SABBATH_CONCEPTS': 20,
            'ISRAELI_LAW': 25,
            'CONSTITUTIONAL_RIGHTS': 20,
            'DAMAGES_TYPES': 25,
            'RELIGIOUS_PLACES': 15,
            'ISRAELI_LOCATIONS': 15,
            'PROFESSIONAL_STANDARD': 20
        }

        for category, score_value in scoring_rules.items():
            if custom_entities.get(category):
                liability_score += score_value
                risk_factors.append(f"{category.replace('_', ' ').title()}: {', '.join(custom_entities[category][:3])}")

        liability_score = min(liability_score, 100)

        if liability_score >= 70:
            risk_level = 'HIGH'
        elif liability_score >= 40:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'

        return {
            'liability_score': liability_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'entity_count': sum(len(entities) for entities in custom_entities.values())
        }

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Complete analysis of a text string"""
        if not text or pd.isna(text):
            return {
                'liability_score': 0,
                'risk_level': 'LOW',
                'risk_factors': [],
                'custom_entities': {},
                'entity_count': 0
            }

        custom_entities = self.custom_entity_matching(text)
        liability_analysis = self.calculate_liability_risk(custom_entities)

        return {
            'liability_score': liability_analysis['liability_score'],
            'risk_level': liability_analysis['risk_level'],
            'risk_factors': liability_analysis['risk_factors'],
            'custom_entities': custom_entities,
            'entity_count': liability_analysis['entity_count']
        }

    def analyze_csv_dataset(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        """Analyze the Sharon dataset CSV file with progress tracking"""
        print(f"\n📂 Loading dataset from: {csv_path}")

        try:
            print("   📄 Reading CSV file...")
            with tqdm(desc="Loading CSV", unit="file") as pbar:
                df = pd.read_csv(csv_path, encoding='cp1252')
                pbar.update(1)
        except UnicodeDecodeError:
            try:
                with tqdm(desc="Loading CSV (UTF-8)", unit="file") as pbar:
                    df = pd.read_csv(csv_path, encoding='utf-8')
                    pbar.update(1)
            except:
                with tqdm(desc="Loading CSV (Latin1)", unit="file") as pbar:
                    df = pd.read_csv(csv_path, encoding='latin1')
                    pbar.update(1)

        print(f"   ✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

        estimated_time = len(df) * 2
        print(f"   ⏱️  Estimated processing time: ~{estimated_time // 60}m {estimated_time % 60}s")

        results = []
        print(f"\n🔍 Analyzing {len(df)} rows for civil liability and religious concepts...")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing rows", unit="row"):
            current_subject = str(row.get('subject', 'Unknown'))[:30]
            tqdm.write(f"   Processing: {current_subject}...")

            prompt_analysis = self.analyze_text(str(row.get('prompt', '')))
            response_analysis = self.analyze_text(str(row.get('response', '')))

            combined_score = max(prompt_analysis['liability_score'], response_analysis['liability_score'])
            combined_risk = 'HIGH' if combined_score >= 70 else 'MEDIUM' if combined_score >= 40 else 'LOW'

            result = {
                'row_index': idx,
                'subject': row.get('subject', ''),
                'llm_status': row.get('LLM status', ''),
                'prompt_liability_score': prompt_analysis['liability_score'],
                'prompt_risk_level': prompt_analysis['risk_level'],
                'prompt_entity_count': prompt_analysis['entity_count'],
                'prompt_risk_factors': '; '.join(prompt_analysis['risk_factors'][:3]),
                'response_liability_score': response_analysis['liability_score'],
                'response_risk_level': response_analysis['risk_level'],
                'response_entity_count': response_analysis['entity_count'],
                'response_risk_factors': '; '.join(response_analysis['risk_factors'][:3]),
                'combined_liability_score': combined_score,
                'combined_risk_level': combined_risk,
                'total_entity_count': prompt_analysis['entity_count'] + response_analysis['entity_count'],
                'prompt_entities': json.dumps(prompt_analysis['custom_entities']),
                'response_entities': json.dumps(response_analysis['custom_entities'])
            }
            results.append(result)

        print(f"\n💾 Processing results...")
        results_df = pd.DataFrame(results)
        final_df = df.copy()
        for col in results_df.columns:
            if col not in ['row_index']:
                final_df[col] = results_df[col]

        if output_path is None:
            base_name = os.path.splitext(csv_path)[0]
            output_path = f"{base_name}_analyzed.csv"

        print(f"   💾 Saving results to: {output_path}")
        final_df.to_csv(output_path, index=False)
        print(f"   ✅ Results saved successfully!")

        # Print summary
        self.print_summary(results_df)
        return final_df

    def print_summary(self, results_df: pd.DataFrame):
        """Print analysis summary"""
        print("\n" + "=" * 60)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"Total rows analyzed: {len(results_df)}")

        risk_counts = results_df['combined_risk_level'].value_counts()
        print(f"\n🎯 Risk Level Distribution:")
        for risk, count in risk_counts.items():
            percentage = count / len(results_df) * 100
            print(f"  {risk}: {count} ({percentage:.1f}%)")

        avg_combined_score = results_df['combined_liability_score'].mean()
        print(f"\n⚖️ Average Combined Liability Score: {avg_combined_score:.1f}/100")

        high_risk_cases = results_df[results_df['combined_risk_level'] == 'HIGH']
        if len(high_risk_cases) > 0:
            print(f"\n⚠️  HIGH RISK CASES ({len(high_risk_cases)} found):")
            for idx, row in high_risk_cases.head(3).iterrows():
                subject = str(row.get('subject', 'Unknown'))[:40]
                score = row.get('combined_liability_score', 0)
                print(f"  • Row {idx}: {subject} (Score: {score})")


def main():
    """Main function - just change the CSV path below"""

    # CHANGE THIS TO YOUR CSV FILE PATH
    csv_file_path = "prompt_buildup_Sharon_2.csv"

    print("🚀 Starting Jewish/Israeli Legal NER Analysis")
    print("=" * 50)

    if not os.path.exists(csv_file_path):
        print(f"❌ Error: File not found: {csv_file_path}")
        print("📝 Make sure your CSV file is in the same folder as this script")
        return None

    # Initialize analyzer
    analyzer = JewishLegalNERAnalyzer()

    # Analyze dataset
    results = analyzer.analyze_csv_dataset(csv_file_path)

    if results is not None:
        print(f"\n🎉 Analysis Complete!")
        print(f"📄 Results saved as: {csv_file_path.replace('.csv', '_analyzed.csv')}")
        print(f"📊 You can now use the analyzed CSV for your A/B testing research")

    return results


if __name__ == "__main__":
    results = main()