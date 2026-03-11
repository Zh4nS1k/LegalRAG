
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.rag_chain import _vector_retriever, invoke_qa
from langchain_core.documents import Document

query = "Что будет если я украду яблоко?"

print(f"--- Testing Retrieval for: '{query}' ---")
try:
    docs = _vector_retriever.invoke(query)
    print(f"Retrieved {len(docs)} documents.")
    found_188 = False
    for i, doc in enumerate(docs):
        meta = doc.metadata
        art_num = str(meta.get("article_number", ""))
        if "188" in art_num:
            found_188 = True
            print(f"\n--- FOUND ARTICLE 188 ---")
            print(f"Metadata: {meta}")
            print(f"Content:\n{doc.page_content}\n-------------------------\n")
    
    if found_188:
        print("\nSUCCESS: Article 188 found!")
    else:
        print("\nFAILURE: Article 188 NOT found.")

    print("\n--- Testing Invoke QA ---")
    res = invoke_qa(query)
    print("Result:", res["result"])
    
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error: {e}")
