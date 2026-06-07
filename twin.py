#!/usr/bin/env python3
import argparse
import sys

# Standard library safe alternative to PyYAML for standalone script distribution
def parse_simple_yaml(filepath):
    """Parses flat/nested structure without external dependencies."""
    data = {}
    current_section = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line and not line.endswith(':'):
                k, v = line.split(':', 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                # Parse types
                if v.isdigit(): v = int(v)
                elif v.replace('.', '', 1).isdigit(): v = float(v)
                
                if current_section:
                    data[current_section][k] = v
                else:
                    data[k] = v
            elif line.endswith(':'):
                current_section = line[:-1].strip()
                data[current_section] = {}
    return data

class FakeLLMScaler:
    def __init__(self, config):
        self.config = config

    def calculate_throughput(self, gpu_count, gpu_tflops):
        m = self.config["model"]["parameters_billion"]
        batch = self.config["inference"]["batch_size"]
        # Simplified operational throughput formula
        return ((gpu_count * gpu_tflops * 40) / m) * (batch ** 0.6) * (2000 / (2000 + self.config["inference"]["sequence_length"]))

    def simulate(self):
        m = self.config["model"]["parameters_billion"]
        g = self.config["hardware"]["gpu_count"]
        mem = self.config["hardware"]["gpu_memory_gb"]
        tflops = self.config["hardware"]["gpu_tflops"]
        batch = self.config["inference"]["batch_size"]
        seq = self.config["inference"]["sequence_length"]

        # Base weights calculation (assuming default FP16 = 2 bytes/param)
        weight_memory = m * 2
        
        # High-fidelity KV Cache calculation proxy per token
        kv_cache_per_token_gb = (m * 1.5e-6 + 4e-5)
        kv_cache_memory = kv_cache_per_token_gb * seq * batch
        
        total_required_memory = weight_memory + kv_cache_memory
        total_gpu_memory = g * mem
        memory_utilization = (total_required_memory / total_gpu_memory) * 100

        throughput = self.calculate_throughput(g, tflops)
        latency = (seq * m) / (g * 2500)
        gpu_utilization = min(98.0, max(15.0, (batch * 0.4) + (seq / 100)))

        return {
            "model_weights_gb": weight_memory,
            "kv_cache_demand_gb": kv_cache_memory,
            "total_memory_needed_gb": total_required_memory,
            "memory_utilization_pct": memory_utilization,
            "throughput_tokens_sec": throughput,
            "latency_ms": latency,
            "gpu_compute_utilization_pct": gpu_utilization,
            "fits_memory": total_required_memory < total_gpu_memory,
        }

def print_report(config, metrics):
    print("\n" + "=" * 65)
    print(" 🚀  LLM INFRASTRUCTURE SIMULATION REPORT")
    print("=" * 65)
    print(f"  [Model]      : {config['model']['name'].upper()} ({config['model']['parameters_billion']}B)")
    print(f"  [Hardware]   : {config['hardware']['gpu_count']}x {config['hardware']['gpu_type']} ({config['hardware']['gpu_memory_gb']}GB VRAM)")
    print(f"  [Workload]   : Batch Size {config['inference']['batch_size']} | Seq Len {config['inference']['sequence_length']}")
    print("-" * 65)
    print(f"  • Model Weight VRAM    : {metrics['model_weights_gb']:.1f} GB")
    print(f"  • Context KV Cache VRAM: {metrics['kv_cache_demand_gb']:.1f} GB")
    print(f"  • Total VRAM Required  : {metrics['total_memory_needed_gb']:.1f} GB / {config['hardware']['gpu_count'] * config['hardware']['gpu_memory_gb']} GB")
    print(f"  • Memory Utilization   : {metrics['memory_utilization_pct']:.1f}%")
    print(f"  • Est. System Output   : {metrics['throughput_tokens_sec']:.1f} tokens/sec")
    print(f"  • Computed Latency     : {metrics['latency_ms']:.1f} ms")
    print(f"  • GPU Load Proxy       : {metrics['gpu_compute_utilization_pct']:.1f}%")
    print("-" * 65)
    if metrics["fits_memory"]:
        print("  ✓ [STATUS: PASSED] Hardware configuration is stable for this workload.")
    else:
        print("  ❌ [STATUS: CRITICAL OOM] System configuration will cause a CUDA Out-Of-Memory error.")
    print("=" * 65 + "\n")

def main():
    parser = argparse.ArgumentParser(description="gpusim CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sim = sub.add_parser("simulate")
    sim.add_argument("config", help="Path to YAML topology configuration")
    
    args = parser.parse_args()
    if args.command == "simulate":
        try:
            cfg = parse_simple_yaml(args.config)
            scaler = FakeLLMScaler(cfg)
            metrics = scaler.simulate()
            print_report(cfg, metrics)
        except Exception as e:
            print(f"Execution Error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()