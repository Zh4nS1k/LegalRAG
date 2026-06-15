"""Logging module replacing loguru using rich."""
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import time
from typing import List
from models.schemas import TestRow, Chunk
from config import settings

class PipelineLogger:
    """Rich-based logging replacing loguru."""
    def __init__(self, verbose: bool = False):
        self.console = Console()
        self.verbose = verbose

    def log_pipeline_start(self, total_questions: int, models: List[str]):
        text = f"Questions: {total_questions}\nModels: {', '.join(models)}\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        text += f"Pinecone Index: {settings.pinecone_index_name}\nEmbedding Model: {settings.embedding_model}"
        panel = Panel(text, title="🚀 Legal RAG Testing Pipeline", border_style="bold blue")
        self.console.print(panel)

    def log_question(self, q_id: str, text: str, index: int, total: int):
        self.console.print(f"🚀 [bold cyan][Q {index}/{total}][/bold cyan] #{q_id} — {text[:100]}...")

    def log_retrieval(self, q_id: str, chunks: List[Chunk], elapsed_ms: float):
        if not self.verbose: return
        self.console.print(f"🔍 Retrieved {len(chunks)} chunks in {elapsed_ms:.0f}ms")
        for i, chunk in enumerate(chunks, 1):
            text_preview = chunk.text[:120].replace('\n', ' ')
            self.console.print(f"📄   [Chunk {i}] score={chunk.score:.3f} | {text_preview}...")

    def log_request(self, q_id: str, model: str, system: str, user: str, token_count: int):
        if not self.verbose: return
        self.console.print(f"📤 [{model}] Sending request — {token_count} prompt tokens")
        sys_preview = system[:200] + "..." if len(system) > 200 else system
        usr_preview = user[:400] + "..." if len(user) > 400 else user
        content = f"**System Prompt:**\n{sys_preview}\n\n**User Prompt:**\n{usr_preview}"
        panel = Panel(content, title=f"REQUEST → {model}", border_style="blue")
        self.console.print(panel)

    def log_response(self, q_id: str, model: str, answer: str, elapsed_ms: float):
        if not self.verbose: return
        self.console.print(f"📥 [{model}] Response received in {elapsed_ms:.0f}ms")
        color = "yellow" if "No answer found in context" in answer or "Контекстте жауап жоқ" in answer else "green"
        panel = Panel(answer, title=f"RESPONSE ← {model}", border_style=color)
        self.console.print(panel)

    def log_tokens(self, q_id: str, model: str, prompt_tokens: int, completion_tokens: int, total_tokens: int, elapsed_ms: float):
        if not self.verbose: return
        tokens_per_sec = completion_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
        self.console.print(f"🧮   prompt={prompt_tokens} | completion={completion_tokens} | total={total_tokens} | speed={tokens_per_sec:.1f} tok/s")

    def log_timing(self, step_name: str, elapsed_ms: float):
        if not self.verbose: return
        self.console.print(f"⏱️  Timing for {step_name}: {elapsed_ms:.0f}ms")

    def log_warning(self, message: str):
        self.console.print(f"⚠️  [yellow]{message}[/yellow]")

    def log_error(self, q_id: str, model: str, error: str):
        self.console.print(f"❌  [red][{model}] Q#{q_id} FAILED — {error}[/red]")
        
    def log_simple_error(self, message: str):
        self.console.print(f"❌  [red]{message}[/red]")
        
    def log_simple_info(self, message: str):
        if not self.verbose: return
        self.console.print(f"ℹ️  {message}")
        
    def log_simple_success(self, message: str):
        self.console.print(f"✅  [green]{message}[/green]")

    def log_question_done(self, q_id: str, elapsed_ms: float, models_count: int):
        self.console.print(f"✅  Q#{q_id} completed ({models_count} models) in {elapsed_ms:.0f}ms total")

    def log_model_result(self, model_idx: int, total_models: int,
                         model: str, result) -> None:
        """Print one model result line immediately after it completes."""
        model_short = model.split("/")[-1][:32]

        if result.error and result.error not in ("truncated",):
            icon = "❌"
            style = "red"
        elif result.answer.strip() == "Контекстте жауап жоқ.":
            icon = "🔶"
            style = "yellow"
        else:
            icon = "✅"
            style = "default"

        answer_preview = result.answer.replace("\n", " ")[:80] if result.answer else ""
        branch = "└─" if model_idx == total_models else "├─"
        score_val = result.quality_score if result.quality_score is not None else "?"
        score_str = f"score={score_val}"
        tok_s = result.tokens_per_sec if result.tokens_per_sec else 0.0

        line = (
            f"  {branch} [{model_idx}/{total_models}] "
            f"{model_short:<32} "
            f"{icon}  "
            f"{result.latency_ms:>7.0f}ms | "
            f"{tok_s:>6.1f} tok/s | "
            f"{score_str:<8} | "
            f"{answer_preview}..."
        )
        self.console.print(line, style=style)

    def log_question_best(self, q_id: str, model_results: list,
                          total_ms: float) -> None:
        """Print best model summary after all models for one question."""
        answered = [
            r for r in model_results
            if not r.error
            and r.answer.strip() != "Контекстте жауап жоқ."
            and r.quality_score
        ]
        if answered:
            best = max(answered, key=lambda r: r.quality_score)
            best_str = f"best: {best.model.split('/')[-1]} (score={best.quality_score})"
        else:
            best_str = "no answered results"

        self.console.print(
            f"  ✅ Q#{q_id} done — {len(model_results)} models | "
            f"{best_str} | {total_ms:.0f}ms total\n"
        )


    def log_pipeline_summary(self, results: List[TestRow]):
        table = Table(title="🏁 Pipeline Summary", show_header=True, header_style="bold magenta", show_lines=True, min_width=110)
        table.add_column("Model", style="cyan", min_width=46, no_wrap=True)
        table.add_column("Questions", style="white", min_width=10, justify="right")
        table.add_column("Answered", style="green", min_width=10, justify="right")
        table.add_column("No-answer", style="yellow", min_width=10, justify="right")
        table.add_column("Errors", style="red", min_width=8, justify="right")
        table.add_column("Avg latency ms", style="blue", min_width=15, justify="right")
        table.add_column("Avg tok/s", style="magenta", min_width=11, justify="right")
        table.add_column("Total tokens", style="white", min_width=13, justify="right")
        table.add_column("Avg quality", style="yellow", min_width=11, justify="right")

        summary = {}
        for row in results:
            m = row.result.model
            if m not in summary:
                summary[m] = {"q": 0, "ans": 0, "no_ans": 0, "err": 0, "lat": 0.0, "tok_s": [], "tot_tok": 0, "qual": []}
            
            s = summary[m]
            s["q"] += 1
            if row.result.error:
                s["err"] += 1
            else:
                ans = row.result.answer
                if ans and ans.strip() == "Контекстте жауап жоқ.":
                    s["no_ans"] += 1
                else:
                    s["ans"] += 1
                
                s["lat"] += row.result.llm_ms
                if row.result.tokens_per_sec is not None:
                    s["tok_s"].append(row.result.tokens_per_sec)
                s["tot_tok"] += (row.result.prompt_tokens + row.result.completion_tokens)
                if row.result.quality_score is not None:
                    s["qual"].append(row.result.quality_score)
            
        for m, s in summary.items():
            avg_lat = s["lat"] / s["q"] if s["q"] > 0 else 0
            avg_tok_s = sum(s["tok_s"]) / len(s["tok_s"]) if s["tok_s"] else 0
            
            if s["qual"]:
                avg_qual_str = f"{sum(s['qual']) / len(s['qual']):.1f}"
            else:
                avg_qual_str = "—"
            
            table.add_row(
                m,
                str(s["q"]),
                str(s["ans"]),
                str(s["no_ans"]),
                str(s["err"]),
                f"{avg_lat:.0f}",
                f"{avg_tok_s:.1f}",
                str(s["tot_tok"]),
                avg_qual_str
            )
            
        # Add total row
        if summary:
            tot_q = sum(s["q"] for s in summary.values())
            tot_ans = sum(s["ans"] for s in summary.values())
            tot_no_ans = sum(s["no_ans"] for s in summary.values())
            tot_err = sum(s["err"] for s in summary.values())
            tot_lat = sum(s["lat"] for s in summary.values())
            tot_tok_s = [tok_s for s in summary.values() for tok_s in s["tok_s"]]
            avg_tot_tok_s = sum(tot_tok_s) / len(tot_tok_s) if tot_tok_s else 0
            avg_tot_lat = tot_lat / tot_q if tot_q > 0 else 0
            tot_tokens = sum(s["tot_tok"] for s in summary.values())
            tot_qual = [q for s in summary.values() for q in s["qual"]]
            avg_tot_qual = f"{sum(tot_qual) / len(tot_qual):.1f}" if tot_qual else "—"
            
            table.add_row(
                "TOTAL", str(tot_q), str(tot_ans), str(tot_no_ans), str(tot_err),
                f"{avg_tot_lat:.0f}", f"{avg_tot_tok_s:.1f}", str(tot_tokens), avg_tot_qual,
                style="bold cyan"
            )
            
        self.console.print(table)

pipeline_logger = PipelineLogger(verbose=settings.verbose)

def update_logger_verbose(verbose: bool):
    pipeline_logger.verbose = verbose