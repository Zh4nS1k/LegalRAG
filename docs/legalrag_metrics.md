# Метрики для LegalRAG (RK Law)

В юридическом RAG “примерно похожий по смыслу” ответ часто равен провалу: ошибка в одной цифре статьи/пункта меняет правовой вывод. Поэтому метрики должны измерять **точность извлечённых норм**, **отсутствие галлюцинаций** и **качество правового применения**.

## RAG Triad (Триада RAG)

### 1) Context Precision (Точность контекста)
Насколько релевантные chunks/документы извлечены retrieval’ом.

В репо считается как сравнение `retrieved_topk` (статья+кодекс/закон из metadata документов) с `gold_citations` из бенчмарка:
- **strict**: совпадает и статья, и кодекс/закон
- **soft**: совпадает хотя бы статья

Практический смысл: если по `ст. 272 ГК РК` retrieval приносит нерелевантные нормы — precision падает.

### 2) Faithfulness (Верность источнику)
Насколько ответ основан **только** на найденном контексте (анти-галлюцинации).

В репо считается LLM-judge’ом по паре `(answer, retrieved_context)` в режиме `full`:
- `faithfulness ∈ [0,1]`

Важно: для корректной проверки нужен тот же контекст, который реально использовался при генерации ответа.

### 3) Answer Relevance (Релевантность ответа)
Насколько ответ действительно решает вопрос пользователя (не уходит в общие слова).

В репо считается LLM-judge’ом по паре `(query, answer)` в режиме `full`:
- `answer_relevance ∈ [0,1]`

## Специфические юридические метрики (Legal-RAG-Bench style)

### 4) Legal Reasoning (Юридическая логика)
Способность не просто цитировать норму, а **применять** её к ситуации:
условия, исключения, квалификация, вывод (например сроки давности, субсидиарная ответственность).

В репо считается LLM-judge’ом по паре `(query, answer)` в режиме `full`:
- `legal_reasoning ∈ [0,1]`

### 5) Citation Accuracy (Точность ссылок)
Наличие и правильность ссылок на пункты/части/статьи.

В репо есть две проверки:
1) **vs gold**: цитаты, извлечённые из текста ответа, сравниваются с `gold_citations` из XLSX:
   - `citation_vs_gold.strict_precision/strict_recall/strict_mrr`
2) (опционально) **vs context**: можно расширить, чтобы проверять, что каждая цитата в ответе реально присутствует среди retrieved документов (защита от “правдоподобной” подмены статьи).

## Как запустить

### Бенчмарк по вашему XLSX (по умолчанию `Полный бенчмарк-3.reviewed8.xlsx`)
- Только citation-metrics (без retrieval и judge):
  - `python -m engine.utils.legal_rag_bench_xlsx --mode offline`
- Добавить Context Precision/Recall (retrieval vs gold):
  - `python -m engine.utils.legal_rag_bench_xlsx --mode retrieval --top-k 10`
- Добавить Faithfulness/Answer Relevance/Legal Reasoning (LLM-judge):
  - `python -m engine.utils.legal_rag_bench_xlsx --mode full --top-k 10`

Результат сохраняется в `benchmark_results/` в JSON (с `summary` и построчными `results`).

## Практический формат для GRATA (коммерческое направление)

Чтобы сфокусировать анализ под коммерческий сектор (гражданское право), рекомендуется:
- сократить набор до самых качественных вопросов по гражданско-правовым кейсам;
- тестировать только 1 флагманскую модель от каждого глобального вендора;
- отдельно фиксировать 5 казахстанских AI-решений для сравнительного анализа интеграции.

### Глобальный shortlist (по 1 продвинутой модели)
- DeepSeek: `deepseek_v3_2`
- Meta: `llama_4_maverick`
- OpenAI: `gpt_5_4_pro`
- Google: `gemini_3_pro`
- Anthropic: `claude_4_6_opus`

### 5 казахстанских AI-решений для сравнения
- Zakon AI
- BeeFree (AI Legal Assistant)
- SmartGov AI (юридический модуль)
- Aisulu / Цифровой помощник предпринимателя
- Локальные RAG-системы (Astana Hub, нишевые LegalRAG-проекты)

### Как посчитать бюджет для 50/100 вопросов

Скрипт:
- `python scripts/calculate_commercial_shortlist_budget.py --question-counts 50,100`

Сохранить JSON-отчет:
- `python scripts/calculate_commercial_shortlist_budget.py --question-counts 50,100 --output-json tests/benchmarks/commercial_shortlist_budget.json`

Этот расчет использует `tests/benchmarks/llm_budget_estimates.market_2026.json` и считает:
- стоимость на 1 запрос;
- стоимость для сценариев на 50 и 100 вопросов;
- суммарный бюджет, если прогонять все 5 глобальных моделей.

### Авто-фильтр вопросов под гражданское/коммерческое направление

Скрипт фильтрации:
- `python scripts/filter_benchmark_for_commercial_civil.py`

Сделать выборку ровно на 50 или 100 вопросов:
- `python scripts/filter_benchmark_for_commercial_civil.py --max-rows 50 --output tests/benchmarks/kz_benchmark_commercial_civil.50.xlsx`
- `python scripts/filter_benchmark_for_commercial_civil.py --max-rows 100 --output tests/benchmarks/kz_benchmark_commercial_civil.100.xlsx`

Дальше бюджет считается этим же shortlist-скриптом:
- `python scripts/calculate_commercial_shortlist_budget.py --question-counts 50,100`

### Precision-расчет только для качественных моделей

Если нужен формат отчета "как для презентации" (средний токен вопроса, total input/output, цена за запрос и итог по датасету), используйте:
- `python scripts/compute_quality_models_budget.py --output-json tests/benchmarks/quality_models_budget.json`

Скрипт:
- считает токены вопросов из `kz_benchmark_gold_final.filtered.xlsx` (tokenizer `cl100k_base`);
- применяет формулу `per_request` и `dataset_total` с допущениями `+2500` input RAG и `500` output;
- оставляет только качественные модели (DeepSeek V4 Pro, GPT-5.4, Claude Sonnet 4.6, Gemini 3 Pro, Claude Opus 4.7, GPT-5.5).

