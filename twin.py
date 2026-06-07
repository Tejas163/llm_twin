#!/usr/bin/env python3
import argparse
import sys
import os

# Pure-Python YAML Parser to avoid pip install issues in clean environments
def parse_simple_yaml(filepath):
    data = {}
    current_section = None
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ':' in line and not line.endswith(':'):
                    k, v = line.split(':', 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
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
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

class FakeLLMScaler:
    def __init__(self, config):
        self.config = config

    def calculate_throughput(self, gpu_count, gpu_tflops):
        m = self.config["model"]["parameters_billion"]
        batch = self.config["inference"]["batch_size"]
        return ((gpu_count * gpu_tflops * 40) / m) * (batch ** 0.6) * (2000 / (2000 + self.config["inference"]["sequence_length"]))

    def simulate(self):
        m = self.config["model"]["parameters_billion"]
        g = self.config["hardware"]["gpu_count"]
        mem = self.config["hardware"]["gpu_memory_gb"]
        tflops = self.config["hardware"]["gpu_tflops"]
        batch = self.config["inference"]["batch_size"]
        seq = self.config["inference"]["sequence_length"]

        weight_memory = m * 2
        kv_cache_per_token_gb = (m * 1.5e-6 + 4e-5)
        kv_cache_memory = kv_cache_per_token_gb * seq * batch
        
        total_required_memory = weight_memory + kv_cache_memory
        total_gpu_memory = g * mem

        throughput = self.calculate_throughput(g, tflops)
        latency = (seq * m) / (g * 2500)

        return {
            "model": self.config["model"]["name"],
            "topology": f"{g}x {self.config['hardware']['gpu_type']}",
            "total_mem_gb": total_gpu_memory,
            "required_mem_gb": total_required_memory,
            "mem_util_pct": (total_required_memory / total_gpu_memory) * 100,
            "throughput_tok_sec": throughput,
            "latency_ms": latency,
            "fits": total_required_memory < total_gpu_memory
        }

def generate_markdown_matrix(config_dir):
    if not os.path.isdir(config_dir):
        print(f"Error: Directory '{config_dir}' does not exist.")
        sys.exit(1)

    results = []
    files = [f for f in os.listdir(config_dir) if f.endswith('.yaml') or f.endswith('.yml')]
    
    for file in sorted(files):
        path = os.path.join(config_dir, file)
        cfg = parse_simple_yaml(path)
        if cfg and "model" in cfg and "hardware" in cfg:
            metrics = FakeLLMScaler(cfg).simulate()
            metrics["filename"] = file
            results.append(metrics)

    md = []
    md.append("### 🚦 `llm-twin` Architectural Simulation Matrix")
    md.append("Automated performance validation run across target infrastructure configurations.\n")
    md.append("| Profile Name | Target Model | Cluster Setup | VRAM Allocation | Throughput | Est. Latency | Status |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for r in results:
        vram_string = f"{r['required_mem_gb']:.1f} / {r['total_mem_gb']} GB ({r['mem_util_pct']:.1f}%)"
        status = "✅ PASSED" if r["fits"] else "❌ CRITICAL OOM"
        throughput_str = f"{r['throughput_tok_sec']:,.1f} tok/s"
        latency_str = f"{r['latency_ms']:.1f} ms"
        md.append(f"| `{r['filename']}` | {r['model'].upper()} | {r['topology']} | {vram_string} | {throughput_str} | {latency_str} | {status} |")
    
    md.append("\n*Generated automatically by `llm-twin` continuous integration gatekeeper.*")
    return "\n".join(md)

def main():
    parser = argparse.ArgumentParser(description="llm-twin orchestration engine")
    
    # CRITICAL FIX: dest="command" must be explicitly set here 
    # so that args.command becomes accessible.
    sub = parser.add_subparsers(dest="command", required=True)
    
    # 1. Single Simulation Subcommand
    sim = sub.add_parser("simulate")
    sim.add_argument("config", help="Path to a single YAML configuration file")
    
    # 2. Batch Matrix Subcommand
    matrix = sub.add_parser("matrix")
    matrix.add_argument("directory", help="Path to directory containing multiple YAML configurations")

    args = parser.parse_args()

    # Safety fall-through check
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "matrix":
        print(generate_markdown_matrix(args.directory))
    elif args.command == "simulate":
        cfg = parse_simple_yaml(args.config)
        if cfg:
            metrics = FakeLLMScaler(cfg).simulate()
            print(f"Throughput: {metrics['throughput_tok_sec']:.2f} tokens/sec | Fits Memory: {metrics['fits']}")

if __name__ == "__main__":
    main()
