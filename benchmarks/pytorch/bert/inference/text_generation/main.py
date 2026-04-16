#!/usr/bin/env python3
"""
BERT Text Classification Inference Benchmark for PyTorch
(Located under text_generation/ for framework compatibility, but BERT is an
encoder-only model so the actual task is text classification / sentiment analysis.)
"""

import sys
from pathlib import Path

# Import and re-use the text_classification benchmark since BERT
# is not a generative model — this path exists only so the benchmark
# runner can dispatch to it without errors.
text_cls_dir = str(Path(__file__).resolve().parent.parent / "text_classification")
if text_cls_dir not in sys.path:
    sys.path.insert(0, text_cls_dir)

from main import main  # noqa: E402

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
