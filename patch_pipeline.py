import re

with open("llm_models_test/legal_rag_tester/pipeline.py", "r", encoding="utf-8") as f:
    content = f.read()

# We need to add imports
if "from concurrent.futures" not in content:
    content = content.replace("import time\n", "import time\nimport threading\nfrom concurrent.futures import ThreadPoolExecutor, as_completed\n")

# We want to replace the `for i, model_name in enumerate(self.models_to_test):` block.
# Since it's large, we can define the inner function inside `run` or as a helper.
# Actually, the easiest is to define a worker function right before the model loop.

worker_func = """
                def process_model(args):
                    i, model_name = args
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
                                result.answer = second_result.answer
                                result.answer_raw = second_result.answer_raw
                                result.latency_ms += second_result.latency_ms
                                result.retrieve_ms += second_result.retrieve_ms
                                result.embed_ms += second_result.embed_ms
                        except KeyboardInterrupt:
                            raise
                        except Exception as e:
                            pipeline_logger.log_error(question.id, model_name, f"unhandled: {e}")
                            result = LLMResult(model=model_name, answer="", error=str(e),
                                               latency_ms=0, chunks_used=0)

                    result.llm_ms = t_llm.elapsed_ms
                    result.total_ms = result.latency_ms

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

                    with scorer_lock:
                        score_val, reason = self.answer_scorer.score(
                            question.text, question.text, result.answer, question.id
                        )
                        time.sleep(0.5)

                    result.quality_score = score_val
                    result.quality_reason = reason
                    result.tokens_per_sec = comp_toks / (result.llm_ms / 1000.0) if result.llm_ms > 0 else 0.0
                    
                    with counter_lock:
                        self.token_counter.record(model_name, prompt_toks, comp_toks)

                    row = TestRow(question=question, result=result)
                    
                    with print_lock:
                        pipeline_logger.log_model_result(i + 1, len(self.models_to_test), model_name, result)
                        if not result.error:
                            pipeline_logger.log_request(question.id, model_name, "", question.text, prompt_toks)
                            pipeline_logger.log_response(question.id, model_name, result.answer, result.llm_ms)
                            pipeline_logger.log_tokens(question.id, model_name, prompt_toks, comp_toks, total_toks, result.llm_ms)
                            
                    return row, t_llm.elapsed_ms

                scorer_lock = threading.Lock()
                counter_lock = threading.Lock()
                print_lock = threading.Lock()
                
                with ThreadPoolExecutor(max_workers=len(self.models_to_test)) as executor:
                    futures = [executor.submit(process_model, (i, m)) for i, m in enumerate(self.models_to_test)]
                    for future in as_completed(futures):
                        try:
                            row, t_ms = future.result()
                            model_results.append(row)
                            total_llm_ms += t_ms
                        except Exception as e:
                            pipeline_logger.log_error("pipeline", "worker", str(e))
"""

# Find the loop
start_idx = content.find("for i, model_name in enumerate(self.models_to_test):")
end_idx = content.find("# ── Rank models for this question ───────────────────────")

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + worker_func + "\n                " + content[end_idx:]
    with open("llm_models_test/legal_rag_tester/pipeline.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Patched successfully")
else:
    print("Could not find start or end index")
