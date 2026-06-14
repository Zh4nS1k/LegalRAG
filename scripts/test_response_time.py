#!/usr/bin/env python3
"""
Performance Testing Script for LegalRAG
Measures response times before and after optimizations.
"""

import asyncio
import json
import time
import statistics
from typing import List, Dict, Any
import requests
import concurrent.futures

# Test queries covering different types
TEST_QUERIES = [
    # Simple article lookups (should be fastest)
    "Что такое статья 136 УК РК?",
    "Статья 122 Трудового кодекса",
    "Ст. 105 Гражданского кодекса",

    # Definition queries
    "Что такое трудовой договор?",
    "Определение недвижимого имущества",
    "Что означает подмена ребенка?",

    # Complex legal questions
    "Какие права у работника при увольнении?",
    "Что грозит за кражу в крупном размере?",
    "Как оформить договор купли-продажи квартиры?",

    # Procedural questions
    "Как подать иск в суд?",
    "Сроки рассмотрения административного дела",
    "Порядок обжалования решения суда",
]


class PerformanceTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.results = []

    async def test_single_query(self, query: str, query_id: int) -> Dict[str, Any]:
        """Test a single query and return timing metrics."""
        url = f"{self.base_url}/api/v1/internal-chat"
        payload = {
            "query": query,
            "history": []
        }

        start_time = time.perf_counter()

        try:
            response = requests.post(url, json=payload, timeout=30)
            end_time = time.perf_counter()

            response_time_ms = (end_time - start_time) * 1000

            if response.status_code == 200:
                data = response.json()

                # Extract timing metrics from trace report
                trace_report = data.get('trace_report', {})
                metrics_ms = trace_report.get('metrics_ms', {})

                result = {
                    'query_id': query_id,
                    'query': query,
                    'success': True,
                    'total_time_ms': response_time_ms,
                    'response_length': len(data.get('result', '')),
                    'has_sources': len(data.get('source_documents', [])) > 0,
                    'metrics': metrics_ms,
                    'error': None
                }

                # Add breakdown if available
                if 'breakdown' in metrics_ms:
                    for key, value in metrics_ms['breakdown'].items():
                        result[f'{key}_ms'] = value

            else:
                result = {
                    'query_id': query_id,
                    'query': query,
                    'success': False,
                    'total_time_ms': response_time_ms,
                    'error': f"HTTP {response.status_code}: {response.text[:100]}"
                }

        except requests.exceptions.Timeout:
            result = {
                'query_id': query_id,
                'query': query,
                'success': False,
                'total_time_ms': 30000,  # 30 second timeout
                'error': 'Request timeout'
            }
        except Exception as e:
            result = {
                'query_id': query_id,
                'query': query,
                'success': False,
                'total_time_ms': 0,
                'error': str(e)
            }

        print(f"Query {query_id}: {query[:40]}... → {result['total_time_ms']:.0f}ms {'✓' if result['success'] else '✗'}")
        return result

    async def run_concurrent_tests(self, queries: List[str], concurrent_workers: int = 3) -> List[Dict[str, Any]]:
        """Run tests with concurrent workers."""
        print(f"\n🚀 Starting performance test with {len(queries)} queries ({concurrent_workers} concurrent)...")

        # Use ThreadPoolExecutor for concurrent HTTP requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
            loop = asyncio.get_event_loop()
            futures = []

            for i, query in enumerate(queries):
                future = loop.run_in_executor(executor, lambda q=query, idx=i: asyncio.run(self.test_single_query(q, idx)))
                futures.append(future)

            results = await asyncio.gather(*futures)

        self.results = results
        return results

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        successful = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r['success']]

        if not successful:
            return {"error": "No successful tests"}

        times = [r['total_time_ms'] for r in successful]

        # Categorize by query type
        simple_times = []
        definition_times = []
        complex_times = []
        procedural_times = []

        for r in successful:
            query = r['query'].lower()
            if any(keyword in query for keyword in ['статья', 'ст.', 'статьи']):
                simple_times.append(r['total_time_ms'])
            elif any(keyword in query for keyword in ['что такое', 'определение', 'означает']):
                definition_times.append(r['total_time_ms'])
            elif any(keyword in query for keyword in ['права', 'грозит', 'оформить']):
                complex_times.append(r['total_time_ms'])
            else:
                procedural_times.append(r['total_time_ms'])

        report = {
            'summary': {
                'total_queries': len(self.results),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': len(successful) / len(self.results) * 100,
            },
            'timing': {
                'avg_ms': statistics.mean(times),
                'median_ms': statistics.median(times),
                'p95_ms': statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times),
                'p99_ms': max(times),
                'min_ms': min(times),
                'max_ms': max(times),
                'std_dev_ms': statistics.stdev(times) if len(times) > 1 else 0,
            },
            'by_category': {
                'simple_article_lookups': {
                    'count': len(simple_times),
                    'avg_ms': statistics.mean(simple_times) if simple_times else 0,
                },
                'definition_queries': {
                    'count': len(definition_times),
                    'avg_ms': statistics.mean(definition_times) if definition_times else 0,
                },
                'complex_legal_questions': {
                    'count': len(complex_times),
                    'avg_ms': statistics.mean(complex_times) if complex_times else 0,
                },
                'procedural_questions': {
                    'count': len(procedural_times),
                    'avg_ms': statistics.mean(procedural_times) if procedural_times else 0,
                }
            },
            'failures': [
                {
                    'query': r['query'],
                    'error': r.get('error', 'Unknown')
                }
                for r in failed
            ],
            'recommendations': self._generate_recommendations(times)
        }

        # Add breakdown metrics if available
        if successful and 'metrics' in successful[0]:
            breakdown_keys = set()
            for r in successful:
                if 'breakdown' in r.get('metrics', {}):
                    breakdown_keys.update(r['metrics']['breakdown'].keys())

            if breakdown_keys:
                report['breakdown'] = {}
                for key in breakdown_keys:
                    values = [r['metrics']['breakdown'].get(key, 0) for r in successful if 'breakdown' in r.get('metrics', {})]
                    if values:
                        report['breakdown'][key] = {
                            'avg_ms': statistics.mean(values),
                            'percentage_of_total': statistics.mean(values) / report['timing']['avg_ms'] * 100
                        }

        return report

    def _generate_recommendations(self, times: List[float]) -> List[str]:
        """Generate optimization recommendations based on results."""
        avg_time = statistics.mean(times)
        recommendations = []

        if avg_time > 8000:
            recommendations.append("CRITICAL: Response time >8s. Prioritize LLM optimization and context reduction.")
        elif avg_time > 5000:
            recommendations.append("HIGH: Response time >5s. Enable caching and optimize retrieval parameters.")
        elif avg_time > 3000:
            recommendations.append("MEDIUM: Response time >3s. Consider lighter embedding model and parallel processing.")
        else:
            recommendations.append("GOOD: Response time <3s. Maintain current configuration.")

        # Check for high variance
        if len(times) > 1:
            cv = statistics.stdev(times) / avg_time * 100
            if cv > 50:
                recommendations.append("High variance in response times. Implement query classification for consistent performance.")

        # Check specific thresholds
        if any(t > 15000 for t in times):
            recommendations.append("Some queries >15s. Implement circuit breakers and timeout handling.")

        return recommendations

    def save_report(self, filename: str = "performance_report.json"):
        """Save report to JSON file."""
        report = self.generate_report()

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n📊 Report saved to {filename}")

        # Print summary
        print("\n" + "="*60)
        print("PERFORMANCE REPORT SUMMARY")
        print("="*60)
        print(f"Total Queries: {report['summary']['total_queries']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"Average Time: {report['timing']['avg_ms']:.0f}ms")
        print(f"Median Time: {report['timing']['median_ms']:.0f}ms")
        print(f"P95 Time: {report['timing']['p95_ms']:.0f}ms")
        print(f"Best: {report['timing']['min_ms']:.0f}ms | Worst: {report['timing']['max_ms']:.0f}ms")

        if 'breakdown' in report:
            print("\nTime Breakdown:")
            for key, stats in report['breakdown'].items():
                print(f"  {key}: {stats['avg_ms']:.0f}ms ({stats['percentage_of_total']:.0f}%)")

        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")


async def main():
    """Main test execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Test LegalRAG performance')
    parser.add_argument('--url', default='http://localhost:8000', help='AI service URL')
    parser.add_argument('--queries', type=int, default=12, help='Number of queries to test')
    parser.add_argument('--concurrent', type=int, default=3, help='Concurrent workers')
    parser.add_argument('--output', default='performance_report.json', help='Output file')
    parser.add_argument('--custom-queries', help='JSON file with custom queries')

    args = parser.parse_args()

    tester = PerformanceTester(args.url)

    # Load queries
    if args.custom_queries:
        with open(args.custom_queries, 'r', encoding='utf-8') as f:
            custom_queries = json.load(f)
        queries_to_test = custom_queries[:args.queries]
    else:
        queries_to_test = TEST_QUERIES[:args.queries]

    # Run tests
    await tester.run_concurrent_tests(queries_to_test, args.concurrent)

    # Generate and save report
    tester.save_report(args.output)


if __name__ == "__main__":
    asyncio.run(main())