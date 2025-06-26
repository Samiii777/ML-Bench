"""
Text generation inference benchmark for PyTorch
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.metrics import MetricsTracker
from utils.config import get_vram_requirement

class TextGenerationBenchmark:
    def __init__(self, model_name, precision="fp32", batch_size=1, device="cuda"):
        self.model_name = model_name
        self.precision = precision
        self.batch_size = batch_size
        self.device = device
        self.metrics = MetricsTracker()
        
        try:
            # Initialize model and tokenizer
            print(f"Loading model {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if precision in ["fp16", "mixed"] else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            
            # Set precision
            if precision == "fp16":
                self.model = self.model.half()
            elif precision == "mixed":
                self.model = self.model.half()
                
            if device == "cuda":
                self.model = self.model.to(device)
            self.model.eval()
            
            # Sample prompts for benchmarking
            self.prompts = [
                "Once upon a time",
                "The quick brown fox",
                "In a world where",
                "The future of AI",
                "Machine learning is"
            ]
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise
        
    def run_inference(self, num_runs=100):
        """Run inference benchmark"""
        print(f"\nRunning text generation inference benchmark:")
        print(f"Model: {self.model_name}")
        print(f"Precision: {self.precision}")
        print(f"Batch size: {self.batch_size}")
        print(f"Device: {self.device}")
        
        try:
            # Warmup
            print("\nWarming up...")
            for _ in range(3):
                prompt = self.prompts[0]
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    _ = self.model.generate(**inputs, max_length=50)
            
            # Benchmark
            print("\nRunning benchmark...")
            total_tokens = 0
            total_time = 0
            
            for i in range(num_runs):
                prompt = self.prompts[i % len(self.prompts)]
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                start_time = time.time()
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=50,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                end_time = time.time()
                
                # Calculate metrics
                generation_time = end_time - start_time
                num_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
                
                total_tokens += num_tokens
                total_time += generation_time
                
                # Record metrics
                self.metrics.record_metric("generation_time", generation_time)
                self.metrics.record_metric("tokens_per_second", num_tokens / generation_time)
                
                if (i + 1) % 10 == 0:
                    print(f"Run {i+1}/{num_runs}")
            
            # Calculate and print results
            avg_time = total_time / num_runs
            avg_tokens = total_tokens / num_runs
            tokens_per_second = total_tokens / total_time
            
            print("\nResults:")
            print(f"Average generation time: {avg_time:.3f} seconds")
            print(f"Average tokens generated: {avg_tokens:.1f}")
            print(f"Tokens per second: {tokens_per_second:.1f}")
            
            # Get VRAM usage
            vram_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"VRAM used: {vram_used:.1f} GB")
            
            # Record final metrics
            self.metrics.record_metric("avg_generation_time", avg_time)
            self.metrics.record_metric("avg_tokens", avg_tokens)
            self.metrics.record_metric("tokens_per_second", tokens_per_second)
            self.metrics.record_metric("vram_used_gb", vram_used)
            
            return self.metrics.get_metrics()
            
        except Exception as e:
            print(f"Error during benchmark: {str(e)}")
            raise

def run_benchmark(model_name, precision="fp32", batch_size=1, device="cuda"):
    """Run the text generation benchmark"""
    try:
        benchmark = TextGenerationBenchmark(
            model_name=model_name,
            precision=precision,
            batch_size=batch_size,
            device=device
        )
        return benchmark.run_inference()
    except Exception as e:
        print(f"Benchmark failed: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    run_benchmark("meta-llama/Llama-3.1-8B", precision="fp16") 