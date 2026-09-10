"""Performance Benchmarking Script (Module 16)
Profiles end-to-end cataloging pipeline latency and memory usage on CPU/GPU.
Computes:
- Average processing time
- P50 (median) latency
- P95 latency
- Peak memory usage (MB)
- Success rate (%)
"""

import sys
import os
import time
import psutil
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import ProductImagePipeline

TEST_IMAGES_DIR = Path(__file__).parent.parent / "tests" / "images"
SAMPLE_IMAGE = TEST_IMAGES_DIR / "pottery_normal.jpg"


def run_benchmark(iterations: int = 5):
    print("=" * 65)
    print("  ARTISAN COMPUTER VISION PIPELINE BENCHMARK (SIH26090)")
    print("=" * 65)

    if not SAMPLE_IMAGE.exists():
        print(f"Error: Sample test image not found at {SAMPLE_IMAGE}")
        return

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    pipeline = ProductImagePipeline()

    latencies = []
    successes = 0

    print(f"Running {iterations} iterations on {SAMPLE_IMAGE.name}...")
    for i in range(iterations):
        t0 = time.perf_counter()
        try:
            res = pipeline.process(
                source=SAMPLE_IMAGE,
                background="white",
                aspect_ratio="1:1",
                enhancement="auto",
                shadow_mode="professional",
                target_dim=1080,
                save_files=False,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            if res["status"] == "success":
                successes += 1
            print(f"  Iteration {i + 1}: {elapsed_ms:.1f} ms | Status: {res['status']}")
        except Exception as e:
            print(f"  Iteration {i + 1} FAILED: {str(e)}")

    mem_after = process.memory_info().rss / (1024 * 1024)
    peak_mem_mb = mem_after - mem_before

    avg_time = float(np.mean(latencies))
    p50_time = float(np.percentile(latencies, 50))
    p95_time = float(np.percentile(latencies, 95))
    success_rate = (successes / iterations) * 100.0

    print("\n" + "-" * 65)
    print("  BENCHMARK SUMMARY RESULTS")
    print("-" * 65)
    print(f"  Total Iterations        : {iterations}")
    print(f"  Success Rate            : {success_rate:.1f}%")
    print(f"  Average Processing Time : {avg_time:.1f} ms ({avg_time / 1000.0:.2f} s)")
    print(f"  P50 (Median) Latency    : {p50_time:.1f} ms")
    print(f"  P95 Latency             : {p95_time:.1f} ms")
    print(f"  Process Memory Usage    : {mem_after:.1f} MB (Peak Delta: {max(0, peak_mem_mb):.1f} MB)")
    print(f"  Hardware Target         : CPU (ONNX Runtime Multithreaded)")
    print("=" * 65)


if __name__ == "__main__":
    run_benchmark(iterations=5)
