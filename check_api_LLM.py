import requests
import json
import time
from huggingface_hub import HfApi, login

# Your token
HF_TOKEN = ""


def test_basic_connection():
    """Test basic connection to Hugging Face"""
    print("=== Testing Basic Connection ===")

    try:
        # Login to HF
        login(HF_TOKEN)
        print("✅ Successfully logged in to Hugging Face")

        # Test API connection
        api = HfApi(token=HF_TOKEN)
        user_info = api.whoami()
        print(f"✅ Connected as: {user_info['name']}")

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

    return True


def test_simple_model(model_name: str):
    """Test a simple API call to a model"""
    print(f"\n=== Testing Model: {model_name} ===")

    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    # Simple payload
    payload = {
        "inputs": "Hello, how are you?",
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.7
        },
        "options": {
            "wait_for_model": True
        }
    }

    try:
        print(f"Making request to: {api_url}")
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS! Response: {json.dumps(result, indent=2)}")
            return True

        elif response.status_code == 401:
            print("❌ UNAUTHORIZED - Token issue")

        elif response.status_code == 403:
            print("🔒 FORBIDDEN - Need to request access")

        elif response.status_code == 404:
            print("❌ NOT FOUND - Model doesn't exist")

        elif response.status_code == 503:
            print("⏳ MODEL LOADING - Waiting 30 seconds...")
            time.sleep(30)
            return test_simple_model(model_name)  # Retry once

        else:
            print(f"⚠️  UNKNOWN STATUS: {response.status_code}")
            print(f"Response text: {response.text[:500]}")

    except requests.exceptions.Timeout:
        print("⏰ TIMEOUT - Request took too long")

    except Exception as e:
        print(f"❌ ERROR: {e}")

    return False


def test_model_info(model_name: str):
    """Get model information"""
    print(f"\n=== Model Info: {model_name} ===")

    try:
        api = HfApi(token=HF_TOKEN)
        model_info = api.model_info(model_name)

        print(f"✅ Model exists!")
        print(f"   ID: {model_info.id}")
        print(f"   Downloads: {getattr(model_info, 'downloads', 'Unknown')}")
        print(f"   Pipeline: {getattr(model_info, 'pipeline_tag', 'Unknown')}")
        print(f"   Gated: {getattr(model_info, 'gated', False)}")

        return True

    except Exception as e:
        print(f"❌ Cannot access model info: {e}")
        return False


def main():
    """Test multiple models to find working ones"""

    # Test basic connection first
    if not test_basic_connection():
        print("❌ Basic connection failed. Check your token.")
        return

    # Models to test (from most likely to work to least likely)
    models_to_test = [
        "microsoft/DialoGPT-small",  # Very likely to work
        "microsoft/DialoGPT-medium",  # Likely to work
        "google/flan-t5-small",  # Should work
        "facebook/blenderbot-400M-distill",  # Should work
        "HuggingFaceH4/zephyr-7b-beta",  # Might need access
        "mistralai/Mistral-7B-Instruct-v0.1",  # Might need access
        "meta-llama/Llama-3.2-1B-Instruct",  # Likely needs access
        "meta-llama/Llama-2-7b-chat-hf",  # Definitely needs access
    ]

    working_models = []

    for model in models_to_test:
        print("\n" + "=" * 60)

        # First check if we can get model info
        if test_model_info(model):
            # Then test actual inference
            if test_simple_model(model):
                working_models.append(model)
                print(f"🎉 {model} is WORKING!")
            else:
                print(f"⚠️  {model} exists but inference failed")
        else:
            print(f"❌ {model} is not accessible")

    print("\n" + "=" * 60)
    print("=== SUMMARY ===")

    if working_models:
        print("✅ WORKING MODELS:")
        for model in working_models:
            print(f"   - {model}")
        print(f"\n🎯 RECOMMENDED: Use '{working_models[0]}' in your script")
    else:
        print("❌ No models are working. Possible issues:")
        print("   - Token permissions")
        print("   - Network connection")
        print("   - All models require special access")


if __name__ == "__main__":
    main()