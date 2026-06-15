from pathlib import Path
from xml.sax.saxutils import escape
import zipfile


OUTPUT = Path("docs/retrieval_iterations_summary.docx")


CONTENT = """
Полный разбор работ по benchmark и улучшению retrieval

1. Сначала улучшили сам benchmark, а не retrieval
Идея была простая: прежде чем поднимать retrieval, нужно начать мерить его нормально.

Что изменили:
- В engine/utils/retrieval_quality_benchmark.py расширили метрики.
- В engine/tests/test_retrieval_quality_benchmark.py добавили тесты.

Что добавили в benchmark:
- strict/soft precision
- strict/soft recall
- strict/soft f1
- map_strict, map_soft
- relevant_count, predicted_count, *_match_count
- summary по языку, сложности и тегам
- latency avg и p95
- comparison не только по hit@k, но и по mrr и остальным метрикам

Что это дало:
- benchmark перестал быть одной цифрой
- стало видно, где retrieval ломается: multi-article, range queries, разные языки, compound/legal-narrative вопросы

2. Потом добавили автоматическую разметку сложности запросов
Чтобы потом видеть регрессии не в среднем по больнице, а по типам запросов.

Что сделали:
- В benchmark появился _classify_query(...)
- Появились теги:
- lang:ru
- lang:kz
- single_article
- multi_article
- range_query
- compound_issue
- penalty_focused
- code_lookup

Результат:
- В summary["by_tag"] стало видно, на каких типах запросов retrieval проседает сильнее всего.

3. Затем проверили XLSX 642_questions_with_citations.xlsx против gold_citations
Цель была: не просто оценивать релевантно / нерелевантно, а смотреть, совпадают ли retrieved статьи с gold citations.

Сначала был сделан офлайн baseline:
- engine/utils/gold_citations_local_bm25_benchmark.py

Результат на 642:
- strict_hit = 0.0545 -> 35/642
- soft_hit = 0.1745 -> 112/642
- strict_mrr = 0.0237
- soft_mrr = 0.0843

Вывод:
- Чистый локальный BM25 слабый
- Даже soft совпадения низкие
- Для реальной оценки нужен Pinecone

4. После этого сделали direct Pinecone benchmark
Чтобы померить текущий retrieval без лишних догадок.

Сделали файл:
- engine/utils/gold_citations_pinecone_direct_benchmark.py

Результат на 642:
- strict_hit = 0.1059 -> 68/642
- soft_hit = 0.2882 -> 185/642
- strict_mrr = 0.0499
- soft_mrr = 0.1361
- avg_elapsed_sec = 1.235

Это стало baseline.

Главные наблюдения:
- Pinecone сильно лучше BM25
- Но exact match мало, много soft-only и очень много full_miss

5. Потом начали пытаться поднять retrieval через hybrid-подход
Идея была:
- dense retrieval из Pinecone
- lexical retrieval через BM25
- объединение через fusion
- law/article-aware boosts

Сделали:
- engine/utils/gold_citations_pinecone_hybrid_benchmark.py

6. Первая итерация hybrid
Компоненты:
- Pinecone + BM25
- RRF fusion
- бусты по code и article

На first100:
- baseline direct: strict_hit=0.11, soft_hit=0.38, strict_mrr=0.049
- hybrid v1: strict_hit=0.13, soft_hit=0.34, strict_mrr=0.087

Вывод:
- точные попадания выросли
- но soft просел
- идея рабочая, но fusion слишком агрессивный

7. Вторая итерация: hybrid v2
Добавили:
- decomposition длинных запросов
- более активный multi-query retrieval
- чуть более сложную сборку результатов

На first100:
- strict_hit=0.16
- soft_hit=0.36
- strict_mrr=0.091

Против baseline:
- strict_hit +0.05
- strict_mrr +0.0416
- soft_hit -0.02

На полном 642:
- strict_hit = 0.1246 -> 80/642
- soft_hit = 0.2648 -> 170/642
- strict_mrr = 0.0553
- soft_mrr = 0.1065
- avg_elapsed_sec = 2.705

Вывод:
- exact retrieval стал лучше
- но soft ощутимо испортился
- latency ухудшился примерно в 2.2 раза
- как финальный retriever это было ещё сыро

8. Третья итерация: hybrid v3
Что поменяли по сравнению с v2:
- decomposition сделали мягче
- BM25 влияние ослабили
- баланс сместили в сторону Pinecone
- агрессию fusion уменьшили

На first100:
- baseline direct: strict_hit=0.11, soft_hit=0.38, strict_mrr=0.049, soft_mrr=0.165
- hybrid_v2: strict_hit=0.16, soft_hit=0.36, strict_mrr=0.091, soft_mrr=0.144
- hybrid_v3: strict_hit=0.16, soft_hit=0.37, strict_mrr=0.083, soft_mrr=0.152

На полном 642:
- baseline direct:
- strict_hit = 0.1059 -> 68/642
- soft_hit = 0.2882 -> 185/642
- strict_mrr = 0.0499
- soft_mrr = 0.1361
- avg_elapsed_sec = 1.235
- hybrid_v2:
- strict_hit = 0.1246 -> 80/642
- soft_hit = 0.2648 -> 170/642
- strict_mrr = 0.0553
- soft_mrr = 0.1065
- avg_elapsed_sec = 2.705
- hybrid_v3:
- strict_hit = 0.1293 -> 83/642
- soft_hit = 0.2710 -> 174/642
- strict_mrr = 0.0536
- soft_mrr = 0.1070
- avg_elapsed_sec = 2.038

Главный вывод по benchmark-скрипту:
- hybrid_v3 стал лучшим компромиссом
- дал +15 exact hits к baseline
- потерял -11 soft hits
- стал примерно в 1.65x медленнее
- это был лучший из протестированных вариантов

9. После этого начали переносить hybrid_v3 в основной retriever
Задача была: не держать улучшение только в benchmark-скрипте, а встроить его в production retrieval stack.

Что встроили:
- engine/retrieval/rag_chain.py
- engine/core/config.py
- engine/tests/test_retrieval_improvements.py

Что именно добавили:
- feature flag EXPERIMENTAL_HYBRID_V3_RETRIEVAL
- feature flag EXPERIMENTAL_NEIGHBOR_EXPANSION
- decomposition длинных и сложных legal queries
- neighbor expansion для соседних статей
- новый _HybridV3Retriever поверх уже существующей chain

Логика была аккуратная:
- не ломать default retriever
- включать новый режим только флагами
- переиспользовать существующую law-aware инфраструктуру в rag_chain.py

Тесты прошли:
- py_compile
- прямой прогон лёгких test-функций

10. Потом попытались проверить уже интегрированный retriever через реальный benchmark
Использовали:
- engine/utils/gold_citations_retrieval_benchmark.py

Цель:
- не просто мерить экспериментальный скрипт
- а посмотреть, что даёт уже основной retriever после интеграции

На локальной машине полный прогон оказался слишком тяжёлым.
Мы дополнительно проверили единичный retrieval call через основной retriever и увидели:
- один вызов занимал около 44.6s

Отсюда оценка:
- полный прогон 642 вопросов через основной retriever в таком виде занял бы около 8 часов и больше

Вывод:
- интеграция по качеству живая
- но по latency implementation в основном retriever пока слишком тяжёлый

11. После этого был прогон в Colab
Сначала были инфраструктурные проблемы:

Проблема 1:
- ModuleNotFoundError: No module named engine
Решение:
- запускать с PYTHONPATH=/content/LegalRAG

Проблема 2:
- config требовал GROQ_API_KEY
Решение:
- задать GROQ_API_KEY="dummy"

Проблема 3:
- ModuleNotFoundError: No module named langchain_pinecone
Решение:
- установить langchain-pinecone

После этого smoke-test через Colab уже пошёл с реальным Pinecone.

12. Результат smoke-теста интегрированного main retriever в Colab
На 5 вопросах:
- strict_hit = 0.4
- soft_hit = 0.8
- strict_mrr = 0.4
- soft_mrr = 0.6286
- avg_elapsed_sec = 219.9344

По отдельным вопросам:
- 701.528s
- 18.642s
- 45.461s
- 329.237s
- 4.804s

Это показало две вещи:
- retrieval quality есть
- latency абсолютно нерабочая для full 642

Оценка:
- 642 * 220s ~= 39 часов

13. Финальный технический вывод
Мы пришли к двум важным результатам.

Первый:
- как экспериментальный benchmark-runner hybrid_v3 работает и реально поднимает strict_hit

Второй:
- как интегрированный слой внутри rag_chain.py он пока слишком дорогой по времени

То есть сейчас состояние такое:

Лучший benchmark-only вариант:
- hybrid_v3 в отдельном benchmark-скрипте

Текущая проблема production-like интеграции:
- слишком много повторных retrieval steps
- слишком много subqueries
- слишком много Pinecone вызовов через law-aware fallback
- neighbor/code-filter expansion резко раздувает latency

14. Что именно создано и изменено по ходу работы
Основные benchmark/util файлы:
- engine/utils/retrieval_quality_benchmark.py
- engine/utils/gold_citations_local_bm25_benchmark.py
- engine/utils/gold_citations_pinecone_direct_benchmark.py
- engine/utils/gold_citations_pinecone_hybrid_benchmark.py
- engine/utils/gold_citations_retrieval_benchmark.py

Основные интеграционные изменения:
- engine/retrieval/rag_chain.py
- engine/core/config.py
- engine/tests/test_retrieval_improvements.py
- engine/tests/test_retrieval_quality_benchmark.py

Результаты benchmark-ов:
- benchmark_results/642_questions_vs_gold_pinecone_direct.json
- benchmark_results/642_questions_vs_gold_pinecone_hybrid_v2.json
- benchmark_results/642_questions_vs_gold_pinecone_hybrid_v3.json

15. Самый честный итог всей работы
Если в одной фразе:
- benchmark мы сильно улучшили
- retrieval quality подняли в экспериментальном hybrid_v3
- но production-style интеграция этого режима в основной retriever пока слишком медленная и требует отдельной оптимизации

16. Что логично делать дальше
Следующий разумный шаг такой:
1. Не гнать full 642 через текущий main retriever.
2. Сначала оптимизировать latency:
- урезать decomposition
- сократить число subqueries
- ослабить fallback code-filter search
- включать neighbor expansion ещё более избирательно
3. Потом снова сравнить:
- direct baseline
- hybrid_v3 benchmark-runner
- integrated main retriever
""".strip()


def _paragraph(text: str) -> str:
    runs = []
    for line in text.split("\n"):
        escaped = escape(line)
        if not escaped:
            runs.append("<w:r/>")
        else:
            runs.append(
                "<w:r><w:t xml:space=\"preserve\">"
                + escaped
                + "</w:t></w:r>"
            )
        runs.append("<w:r><w:br/></w:r>")
    if runs:
        runs.pop()
    return "<w:p>" + "".join(runs) + "</w:p>"


def build_docx(content: str, output_path: Path) -> None:
    body = _paragraph(content)
    document_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:wpc=\"http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas\" "
        "xmlns:mc=\"http://schemas.openxmlformats.org/markup-compatibility/2006\" "
        "xmlns:o=\"urn:schemas-microsoft-com:office:office\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        "xmlns:m=\"http://schemas.openxmlformats.org/officeDocument/2006/math\" "
        "xmlns:v=\"urn:schemas-microsoft-com:vml\" "
        "xmlns:wp14=\"http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing\" "
        "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
        "xmlns:w10=\"urn:schemas-microsoft-com:office:word\" "
        "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
        "xmlns:w14=\"http://schemas.microsoft.com/office/word/2010/wordml\" "
        "xmlns:wpg=\"http://schemas.microsoft.com/office/word/2010/wordprocessingGroup\" "
        "xmlns:wpi=\"http://schemas.microsoft.com/office/word/2010/wordprocessingInk\" "
        "xmlns:wne=\"http://schemas.microsoft.com/office/2006/wordml\" "
        "xmlns:wps=\"http://schemas.microsoft.com/office/word/2010/wordprocessingShape\" "
        "mc:Ignorable=\"w14 wp14\">"
        "<w:body>"
        + body
        + "<w:sectPr><w:pgSz w:w=\"11906\" w:h=\"16838\"/><w:pgMar w:top=\"1440\" w:right=\"1440\" "
        "w:bottom=\"1440\" w:left=\"1440\" w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/></w:sectPr>"
        "</w:body></w:document>"
    )

    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""

    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""

    core_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Retrieval Iterations Summary</dc:title>
  <dc:creator>OpenAI Codex</dc:creator>
</cp:coreProperties>"""

    app_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>OpenAI Codex</Application>
</Properties>"""

    document_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("docProps/core.xml", core_xml)
        zf.writestr("docProps/app.xml", app_xml)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/_rels/document.xml.rels", document_rels_xml)


if __name__ == "__main__":
    build_docx(CONTENT, OUTPUT)
    print(OUTPUT)
