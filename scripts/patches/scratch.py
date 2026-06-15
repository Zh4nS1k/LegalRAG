from engine.retrieval import agentic_workflow as aw
from langchain_core.documents import Document

query = "Действует ли статья 50 на 2024 год?"
doc = Document(
    page_content="Статья 50. Общие положения.",
    metadata={
        "code_ru": "Закон РК о публичной службе",
        "article_number": "50",
        "status": "действует",
    },
)

evaluation = aw._evaluate_context_quality(query, [doc], [0.31])
print(evaluation)
