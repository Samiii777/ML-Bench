#!/usr/bin/env python3
"""PyTorch BERT Text Classification Inference Benchmark"""

import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve()
for parent in project_root.parents:
    if (parent / "benchmark.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import torch

from core.harness import InferenceHarness
from core.schema import MetricEntry, BenchmarkMeta
from core.args import build_base_parser
from core.validation import ResultValidator

BENCHMARK_META = BenchmarkMeta(
    framework="pytorch",
    model_family="bert",
    supported_models=[
        "bert", "bert-base-uncased", "bert-base-cased",
        "bert-large-uncased", "bert-large-cased", "bert-sentiment",
    ],
    supported_precisions=["fp32", "fp16", "mixed"],
    mode="inference",
    use_case="text_classification",
)


def get_bert_model_name(model_arg):
    """Map model argument to actual Hugging Face model name for classification or use directly if in HF format"""
    if "/" in model_arg:
        print(f"Using HuggingFace model directly: {model_arg}")
        return model_arg

    bert_models = {
        "bert": "nlptown/bert-base-multilingual-uncased-sentiment",
        "bert-base-uncased": "nlptown/bert-base-multilingual-uncased-sentiment",
        "bert-base-cased": "nlptown/bert-base-multilingual-uncased-sentiment",
        "bert-large-uncased": "nlptown/bert-base-multilingual-uncased-sentiment",
        "bert-large-cased": "nlptown/bert-base-multilingual-uncased-sentiment",
        "bert-sentiment": "nlptown/bert-base-multilingual-uncased-sentiment",
    }
    return bert_models.get(model_arg, "nlptown/bert-base-multilingual-uncased-sentiment")


def get_sample_texts():
    """Get diverse sample texts for sentiment classification benchmarking"""
    return [
        # Positive sentiment
        "I absolutely love this product! It's amazing and works perfectly.",
        "This is the best experience I've ever had. Highly recommended!",
        "Fantastic quality and excellent customer service. Very satisfied!",
        "Outstanding performance and great value for money. Love it!",
        "Incredible results! This exceeded all my expectations.",
        # Negative sentiment
        "This is terrible quality and completely broke after one day.",
        "Worst purchase ever. Don't waste your money on this garbage.",
        "Extremely disappointed with this product. Poor build quality.",
        "This doesn't work at all and customer service is unhelpful.",
        "Overpriced and underdelivered. Completely unsatisfied.",
        # Neutral sentiment
        "The product works as described. Nothing special but does the job.",
        "Average quality for the price. Could be better, could be worse.",
        "It's okay, meets basic expectations but nothing more.",
        "Standard functionality, typical for this type of product.",
        "Decent enough, though there's room for improvement.",
        # Mixed/Complex sentiment
        "Great design but poor durability. Mixed feelings about this.",
        "Love the concept but execution could be better.",
        "Good features but the price is too high for what you get.",
        "Works well most of the time but occasionally has issues.",
        "Beautiful aesthetics but functionality is lacking.",
    ]


class BertTextClassificationBenchmark(InferenceHarness):

    @property
    def use_case(self):
        return "text_classification"

    def load_model(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        name = get_bert_model_name(self.args.model or "bert")
        self._hf_model_name = name
        print(f"Loading model: {name}")

        torch_dtype = torch.float32
        if self.args.precision in ("fp16", "mixed"):
            torch_dtype = torch.float16

        self._tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(
            name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device.type == "cuda" else None,
        )
        if self.device.type != "cuda":
            model = model.to(self.device)
        model.eval()
        return model

    def prepare_inputs(self):
        sample_texts = get_sample_texts()
        batch_texts = [sample_texts[j % len(sample_texts)] for j in range(self.args.batch_size)]
        inputs = self._tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def run_step(self, model, inputs):
        return model(**inputs)

    def get_extra_metrics(self, model, inputs, outputs):
        import torch.nn.functional as F

        predictions = F.softmax(outputs.logits.float(), dim=-1)
        predicted_classes = torch.argmax(predictions, dim=-1)
        preds_np = predicted_classes.cpu().numpy()
        unique_classes, class_counts = np.unique(preds_np, return_counts=True)

        self._num_unique_classes = len(unique_classes)

        print("\nPrediction Distribution:")
        for cls, count in zip(unique_classes, class_counts):
            pct = (count / len(preds_np)) * 100
            print(f"  Class {cls}: {count} samples ({pct:.1f}%)")

        return [
            MetricEntry("num_unique_classes", float(self._num_unique_classes), "count", "higher_is_better"),
        ]

    def validate_result(self, model, inputs, outputs, validator: ResultValidator):
        validator.expect_greater_than("num_unique_classes", self._num_unique_classes, 1)


if __name__ == "__main__":
    parser = build_base_parser("PyTorch BERT Text Classification Inference Benchmark")
    parser.set_defaults(model="bert")
    args = parser.parse_args()

    try:
        benchmark = BertTextClassificationBenchmark(args)
        benchmark.run()
        print("Benchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
