import os
import requests

key = os.popen("cat .env | grep OPENROUTER_API_KEY | cut -d '=' -f2").read().strip()
resp = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {key}"},
    json={"model": "openai/gpt-5.5", "messages": [{"role": "user", "content": "hello"}]},
    timeout=10
)
print("Status:", resp.status_code)
print("Response:", resp.text)
