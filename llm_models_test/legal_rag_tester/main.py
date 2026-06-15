"""Entry point for the Legal RAG Tester CLI."""
import argparse
import sys
from logger import update_logger_verbose
from pipeline import TestPipeline
from config import settings


def run_coverage_check(questions, retriever, embedder):
    """Test all questions and report which have 0 chunks (KB gap detection)."""
    from tqdm import tqdm
    zero_chunk_questions = []
    low_score_questions = []

    for q in tqdm(questions, desc="Coverage check"):
        embedding = embedder.embed(q.text)
        chunks = retriever.query_raw(embedding, threshold=0.30)
        if not chunks:
            zero_chunk_questions.append(q)
        elif max(c.score for c in chunks) < 0.60:
            low_score_questions.append((q, max(c.score for c in chunks)))

    good = len(questions) - len(zero_chunk_questions) - len(low_score_questions)

    print(f"\n📊 COVERAGE REPORT")
    print(f"  Total questions   : {len(questions)}")
    print(f"  ❌ Zero matches   : {len(zero_chunk_questions)}  (topics NOT in KB)")
    print(f"  🔶 Low confidence : {len(low_score_questions)}  (score < 0.60 — benefits from rewriting)")
    print(f"  ✅ Good coverage  : {good}  (score >= 0.60)")

    if zero_chunk_questions:
        print(f"\n  Questions with NO KB coverage (first 10):")
        for q in zero_chunk_questions[:10]:
            print(f"    {q.id}: {q.text[:100]}...")

    if low_score_questions:
        print(f"\n  Low-confidence questions (first 5):")
        for q, score in low_score_questions[:5]:
            print(f"    {q.id} [{score:.3f}]: {q.text[:100]}...")


def main():
    """Parses arguments and runs the pipeline."""
    parser = argparse.ArgumentParser(description="Legal RAG Tester Pipeline")
    parser.add_argument("--input", type=str, help="Override INPUT_EXCEL")
    parser.add_argument("--output-dir", type=str, help="Override OUTPUT_DIR")
    parser.add_argument("--models", type=str, help="Override LLM_MODELS (comma-separated)")
    parser.add_argument("--limit", type=int, help="Process only first N questions")
    parser.add_argument("--dry-run", action="store_true", help="Embed and retrieve only, skip LLM calls")
    parser.add_argument("--skip-low-limit", action="store_true", help="Skip low-limit models like llama-4-maverick")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--diagnose", action="store_true", help="Run Pinecone diagnostic tool")
    parser.add_argument("--coverage", action="store_true", help="Run KB coverage check (use with --diagnose)")

    args = parser.parse_args()

    if args.input:
        settings.input_excel = args.input
    if args.output_dir:
        settings.output_dir = args.output_dir

    verbose = args.verbose or settings.verbose
    update_logger_verbose(verbose)

    if args.diagnose:
        import excel_io
        from embedder import Embedder
        from retriever import PineconeRetriever

        questions = excel_io.read_questions()
        if not questions:
            print("No questions found!")
            return

        embedder = Embedder()
        retriever = PineconeRetriever(settings)

        if args.coverage:
            run_coverage_check(questions, retriever, embedder)
            return

        # Single-question deep diagnostic
        q = questions[0]
        embedding = embedder.embed(q.text)
        print(f"🔍 Query embedding dim: {len(embedding)}")
        stats = retriever.index.describe_index_stats()
        print(f"📊 Index stats: dimension={stats.dimension}, "
              f"total_vectors={stats.total_vector_count}, "
              f"namespaces={list(stats.namespaces.keys())}")
        response = retriever.index.query(
            namespace=settings.pinecone_namespace,
            vector=embedding,
            top_k=5,
            include_values=False,
            include_metadata=True
        )
        print("📄 Top 5 raw matches (no threshold filter):")
        for i, match in enumerate(response.get("matches", []), 1):
            meta_keys = list(match.metadata.keys()) if match.metadata else []
            text_preview = match.metadata.get(settings.chunk_text_field, "")[:80] if match.metadata else ""
            print(f"   [{i}] id={match.id}  score={match.score:.3f}  keys={meta_keys}")
            print(f"       text: \"{text_preview}...\"")
        return

    pipeline = TestPipeline(override_models=args.models)
    pipeline.run(
        limit=args.limit,
        dry_run=args.dry_run,
        skip_low_limit=args.skip_low_limit,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()