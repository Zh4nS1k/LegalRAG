"""Pipeline orchestration for the Legal RAG Tester."""
import os
import sys
import signal
import time
from datetime import datetime
from tqdm import tqdm
from logger import pipeline_logger
from timer import StepTimer
from token_counter import TokenCounter
import excel_io
from llm_client import LLMClient
from models.schemas import LLMResult, TestRow
from config import settings
from answer_scorer import AnswerScorer
from checkpoint import CheckpointManager


def _get_provider(model: str) -> str:
    """Classify model by which API key / rate-limit pool it uses."""
    if model.startswith("models/"):
        return "google"
    if model.startswith("gpt-5") or model.startswith("gpt-4"):
        return "openai"
    return "groq"  # everything else (llama, qwen, deepseek, openai/gpt-oss-*)


def _inter_request_delay(prev_model: str, next_model: str) -> None:
    """Sleep using the SLOWER of the two adjacent providers' configured delays."""
    delays = {
        "groq":   float(os.getenv("DELAY_GROQ",   "3.0")),
        "google": float(os.getenv("DELAY_GOOGLE", "6.0")),
        "openai": float(os.getenv("DELAY_OPENAI", "5.0")),
    }
    prev_delay = delays.get(_get_provider(prev_model), 3.0)
    next_delay = delays.get(_get_provider(next_model), 3.0)
    time.sleep(max(prev_delay, next_delay))


class TestPipeline:
    """Orchestrates the entire evaluation process."""

    def __init__(self, override_models: str = None):
        """Initializes the pipeline components."""
        self.llm_client = LLMClient()
        self.token_counter = TokenCounter()
        self.answer_scorer = AnswerScorer()

        self.models_to_test = settings.llm_models
        if override_models:
            self.models_to_test = [m.strip() for m in override_models.split(",") if m.strip()]

    def _check_engine(self) -> bool:
        import requests
        try:
            r = requests.get(
                f"{os.getenv('AI_SERVICE_URL', 'http://localhost:8000')}/health",
                timeout=5
            )
            return r.status_code < 500
        except Exception:
            return False

    def run(self, limit: int = None, dry_run: bool = False,
            skip_low_limit: bool = False, resume: bool = False):
        """Runs the complete RAG test pipeline with crash-safe checkpointing."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint = CheckpointManager(
            output_dir=settings.output_dir,
            session_id=session_id
        )

        # ── Graceful signal handlers ────────────────────────────────────
        def _signal_handler(sig, frame):
            pipeline_logger.log_warning(
                f"⌨️  Interrupted! Saving {len(checkpoint.rows)} rows..."
            )
            checkpoint.finalize()
            sys.exit(0)

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        if not self._check_engine():
            pipeline_logger.log_error("pipeline", "health", "❌ engine не запущен. Запусти: uvicorn engine.api.api:app --port 8000")
            sys.exit(1)

        try:
            questions = excel_io.read_questions()
            if limit:
                questions = questions[:limit]

            if not questions:
                pipeline_logger.log_warning("No questions found. Exiting.")
                return []

            maverick = "meta-llama/llama-4-scout-17b-16e-instruct"
            if maverick in self.models_to_test and (len(questions) > 400 or skip_low_limit):
                pipeline_logger.log_simple_info(f"Skipping {maverick} due to low limit constraints.")
                self.models_to_test.remove(maverick)

            # ── Resume: skip already-completed questions ────────────────
            if resume:
                ckpt_file = CheckpointManager.find_latest_checkpoint(settings.output_dir)
                if ckpt_file:
                    import pandas as pd
                    already_done = set(pd.read_excel(ckpt_file)["id"].astype(str).unique())
                    before = len(questions)
                    questions = [q for q in questions if str(q.id) not in already_done]
                    pipeline_logger.log_simple_info(
                        f"▶️  Resuming from {ckpt_file.name}: "
                        f"{before - len(questions)} done, {len(questions)} remaining"
                    )
                    # Load existing rows into the checkpoint so the final file is complete
                    existing_rows = excel_io.read_results_from_checkpoint(ckpt_file)
                    if existing_rows:
                        checkpoint.rows.extend(existing_rows)
                else:
                    pipeline_logger.log_warning("--resume requested but no checkpoint found. Starting fresh.")

            pipeline_logger.log_pipeline_start(len(questions), self.models_to_test)

            for q_index, question in enumerate(tqdm(questions, desc="Processing questions"), 1):
                pipeline_logger.log_question(question.id, question.text, q_index, len(questions))

                if dry_run:
                    continue

                prompt_toks = 0

                total_llm_ms = 0.0
                model_results = []

                prev_model = None

                for i, model_name in enumerate(self.models_to_test):
                    # ── Inter-request delay ─────────────────────────────
                    if prev_model:
                        _inter_request_delay(prev_model, model_name)
                    prev_model = model_name

                    with StepTimer("llm_call") as t_llm:
                        try:
                            result = self.llm_client.call(
                                model_name, question.text, q_id=question.id
                            )
                            # ---- AUTO-REPLY LOGIC ----
                            if "нужны уточнения:" in result.answer or "Ответьте, пожалуйста," in result.answer:
                                pipeline_logger.log_simple_info(f"🔄 [{model_name}] Запросил уточнения. Отвечаем автоматически: 'Ответьте с имеющимися данными'")
                                
                                history = [
                                    {"role": "user", "content": question.text},
                                    {"role": "assistant", "content": result.answer}
                                ]
                                follow_up_text = "Ответьте с имеющимися данными. Если данных не хватает, дайте частичный анализ или опишите возможные варианты развития событий."
                                
                                second_result = self.llm_client.call(
                                    model_name, 
                                    follow_up_text, 
                                    q_id=question.id,
                                    history=history
                                )
                                # Update result with the final answer
                                result.answer = second_result.answer
                                result.answer_raw = second_result.answer_raw
                                result.latency_ms += second_result.latency_ms
                                result.retrieve_ms += second_result.retrieve_ms
                                result.embed_ms += second_result.embed_ms
                                # --------------------------
                        except KeyboardInterrupt:
                            raise
                        except Exception as e:
                            pipeline_logger.log_error(question.id, model_name, f"unhandled: {e}")
                            result = LLMResult(model=model_name, answer="", error=str(e),
                                               latency_ms=0, chunks_used=0)

                    total_llm_ms += t_llm.elapsed_ms

                    result.llm_ms = t_llm.elapsed_ms
                    result.total_ms = result.latency_ms

                    # Sanitize
                    def sanitize_answer(ans: str) -> str:
                        NO_ANSWER = "Контекстте жауап жоқ."
                        if ans.strip().startswith(NO_ANSWER):
                            return NO_ANSWER
                        ans = ans.replace(NO_ANSWER, "").strip()
                        return ans if ans else NO_ANSWER

                    if result.answer:
                        result.answer = sanitize_answer(result.answer)

                    comp_toks = result.completion_tokens or self.token_counter.count(result.answer_raw)

                    if t_llm.elapsed_ms > 10000:
                        tok_s = comp_toks / (result.llm_ms / 1000.0) if result.llm_ms > 0 else 0.0
                        pipeline_logger.log_warning(
                            f"⚠️  [{model_name}] Q#{question.id} slow: {t_llm.elapsed_ms:.0f}ms "
                            f"({comp_toks} tokens, {tok_s:.1f} tok/s)"
                        )

                    total_toks = prompt_toks + comp_toks
                    result.prompt_tokens = prompt_toks
                    result.completion_tokens = comp_toks

                    # Auto-scoring + scorer RPM protection
                    score_val, reason = self.answer_scorer.score(
                        question.text, question.text, result.answer, question.id
                    )
                    result.quality_score = score_val
                    result.quality_reason = reason
                    time.sleep(0.5)  # scorer shares Groq key — count against RPM

                    result.tokens_per_sec = comp_toks / (result.llm_ms / 1000.0) if result.llm_ms > 0 else 0.0
                    self.token_counter.record(model_name, prompt_toks, comp_toks)

                    model_results.append(TestRow(question=question, result=result))

                    # ✅ Print immediately
                    pipeline_logger.log_model_result(i + 1, len(self.models_to_test), model_name, result)

                    # Verbose logs
                    if not result.error:
                        pipeline_logger.log_request(question.id, model_name, "", question.text, prompt_toks)
                        pipeline_logger.log_response(question.id, model_name, result.answer, result.llm_ms)
                        pipeline_logger.log_tokens(question.id, model_name, prompt_toks, comp_toks, total_toks, result.llm_ms)

                # ── Rank models for this question ───────────────────────
                model_results_sorted = sorted(model_results, key=lambda r: r.result.quality_score, reverse=True)
                for rank, row in enumerate(model_results_sorted, 1):
                    row.result.quality_rank = rank

                pipeline_logger.log_question_best(
                    question.id, [r.result for r in model_results],
                    total_llm_ms
                )

                # ── 💾 Save after EVERY question ────────────────────────
                checkpoint.add_question_results(model_results)

            # ── Clean completion ────────────────────────────────────────
            checkpoint.finalize()
            pipeline_logger.log_pipeline_summary(checkpoint.rows)
            return checkpoint.rows

        except Exception as e:
            pipeline_logger.log_error("pipeline", "pipeline", str(e))
            pipeline_logger.log_warning("💾 Saving partial results before exit...")
            checkpoint.finalize()
            raise