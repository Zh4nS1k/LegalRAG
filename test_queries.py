import asyncio
import logging
import sys

# Configure logging to capture output
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

from engine.retrieval.agentic_workflow import invoke_agentic_qa
from engine.retrieval.intent_router import detect_question_type

async def run_tests():
    questions = [
        "Что будет за кражу яблока?",
        "Какое наказание за кражу в крупном размере?",
        "Каковы права работника при увольнении?",
        "Статья 188 УК РК",
        "Куда подать жалобу на работодателя?"
    ]
    
    for q in questions:
        print("="*60)
        print(f"QUESTION: {q}")
        print("="*60)
        
        qtype = detect_question_type(q)
        print(f"\n1. Тип вопроса (Определен роутером): {qtype}")
        
        trace_id = f"test_intent_{questions.index(q)}"
        result = await invoke_agentic_qa(query=q, trace_id=trace_id)
        
        ans = result.get("result", "")
        print("\n2. Полный текст ответа:")
        print(ans)
        
        # Check if answer contains quotes « » or ""
        has_quote = "«" in ans or "»" in ans or '"' in ans
        print(f"\n3. Цитата из статьи: {'Присутствует' if has_quote else 'ОТСУТСТВУЕТ!'}")
        
        print("\n--- CITED SOURCES IN CONTEXT ---")
        for doc in result.get("source_documents", []):
            code = doc.metadata.get("code_ru", "Unknown Code")
            art = doc.metadata.get("article_number", "Unknown Article")
            print(f"- {code}, Статья {art}")
            
        print("\n\n")

if __name__ == "__main__":
    asyncio.run(run_tests())
