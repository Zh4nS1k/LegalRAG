import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv('.env')

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_API_KEY not found in .env")
    exit(1)

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
response = requests.get(url)

if response.status_code != 200:
    print(f"Error fetching models: {response.status_code} {response.text}")
    exit(1)

models_data = response.json()
models = models_data.get('models', [])

output_file = "google_models_test_results.txt"

print(f"Found {len(models)} models. Starting tests...")

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(f"Total models available: {len(models)}\n\n")
    
    for model in models:
        model_name = model['name']
        print(f"Testing {model_name}...")
        f.write(f"Model: {model_name}\n")
        f.write(f"Display Name: {model.get('displayName', 'N/A')}\n")
        f.write(f"Description: {model.get('description', 'N/A')}\n")
        
        # Test model if it supports generateContent
        supported_methods = model.get('supportedGenerationMethods', [])
        if 'generateContent' in supported_methods:
            test_url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={api_key}"
            payload = {
                "contents": [{
                    "parts": [{"text": "Say 'hello world' and nothing else."}]
                }]
            }
            try:
                res = requests.post(test_url, json=payload, headers={'Content-Type': 'application/json'})
                if res.status_code == 200:
                    answer = res.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No text in response')
                    f.write(f"Test Status: Success\n")
                    f.write(f"Test Output: {answer.strip()}\n")
                else:
                    f.write(f"Test Status: Failed (HTTP {res.status_code})\n")
                    # Try to parse JSON error message
                    try:
                        err_msg = res.json().get('error', {}).get('message', res.text)
                        f.write(f"Error Details: {err_msg}\n")
                    except:
                        f.write(f"Error Details: {res.text}\n")
            except Exception as e:
                f.write(f"Test Status: Failed (Exception)\n")
                f.write(f"Error Details: {str(e)}\n")
        else:
            f.write(f"Test Status: Skipped (generateContent not supported)\n")
            
        f.write("-" * 50 + "\n")

print(f"Tests complete. Results saved to {output_file}")
