import re

with open("LLM_MODELS_TEST/legal_rag_tester/.env", "r") as f:
    content = f.read()

content = re.sub(r'PINECONE_API_KEY=.*\n', '', content)
content = re.sub(r'EMBEDDING_MODEL=.*\n', '', content)
content = re.sub(r'GROQ_API_KEY=.*\n', '', content)
content = re.sub(r'PINECONE_INDEX_NAME=.*\n', '', content)
content = re.sub(r'PINECONE_NAMESPACE=.*\n', '', content)
content = re.sub(r'PINECONE_TOP_K=.*\n', '', content)
content = re.sub(r'PINECONE_FINAL_K=.*\n', '', content)
content = re.sub(r'PINECONE_SCORE_THRESHOLD=.*\n', '', content)
content = re.sub(r'EMBEDDING_PREFIX=.*\n', '', content)

if "AI_SERVICE_URL" not in content:
    content += "\nAI_SERVICE_URL=http://localhost:8000\nAI_SERVICE_TIMEOUT=120\n"

with open("LLM_MODELS_TEST/legal_rag_tester/.env", "w") as f:
    f.write(content)
