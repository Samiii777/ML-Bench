#!/usr/bin/env python3
"""Encoder (BERT-base) text classification inference.

Model: a BERT-base checkpoint fine-tuned for 1-5 star review sentiment, so the
outputs can be checked. Every sequence is padded to exactly --seq_len tokens,
making throughput depend only on batch size and sequence length.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "benchmark.py").exists())))

import torch

from core.args import build_base_parser
from core.harness import InferenceHarness, run_main

CHECKPOINTS = {"bert-base": "nlptown/bert-base-multilingual-uncased-sentiment"}
POSITIVE = "I absolutely love this product. It works perfectly and the quality is outstanding."
NEGATIVE = "Terrible. It broke after one day, support never answered, and I want my money back."


class BertClassification(InferenceHarness):
    use_case = "text_classification"

    def load_model(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        name = CHECKPOINTS.get(self.args.model, self.args.model)
        self._tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name, dtype=self.dtype)
        return model.eval().to(self.device)

    def prepare_inputs(self):
        texts = [POSITIVE if i % 2 == 0 else NEGATIVE for i in range(self.batch_size)]
        enc = self._tok(texts, padding="max_length", truncation=True, max_length=self.args.seq_len,
                        return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    def run_step(self, model, inputs):
        return model(**inputs).logits

    def validate_result(self, model, inputs, logits, validator):
        stars = logits.float().argmax(dim=-1).cpu() + 1  # labels are 1..5 stars
        print(f"Predicted stars (first 4): {stars[:4].tolist()}")
        validator.expect_greater_than("positive_stars", int(stars[0]), 3)
        if self.batch_size > 1:
            validator.expect_in_range("negative_stars", int(stars[1]), 1, 2)

    def get_config(self):
        return {"seq_len": self.args.seq_len, "checkpoint": CHECKPOINTS.get(self.args.model, self.args.model)}


if __name__ == "__main__":
    parser = build_base_parser("BERT text classification inference")
    parser.add_argument("--seq_len", "--seq-len", type=int, default=128, help="tokens per sequence (default: 128)")
    parser.set_defaults(model="bert-base")
    run_main(BertClassification, parser)
