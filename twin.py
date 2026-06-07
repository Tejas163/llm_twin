#!/usr/bin/env python3
import argparse
import sys
import os
import json

# --- Simple YAML Parser from Phase 2 ---
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
                    if current_section: data[current_section][k] = v
                    else: data[k] = v
                elif line.endswith(':'):
                    current_section = line[:-1].strip()
                    data[current_section] = {}
        return data
    except Exception as e:
        print(f"Error parsing YAML {filepath}: {e}")
        return None

# --- Phase 3: Discrete-Event Trace Emulator Engine ---
class TraceEmulator:
    def __init__(self, trace_data, hardware_config):
        self.trace = trace_data
        self.hw = hardware_config
        self.virtual_clock_ms = 0.0
        self.execution_logs = []

    def run(self):
        """Processes execution kernels line-by-line using a virtual clock."""
        tflops = self.hw.get("hardware", {}).get("gpu_tflops", 312)
        gpus = self.hw.get("hardware", {}).get("gpu_count", 1)
        interconnect = self.hw.get("hardware", {}).get("interconnect_type", "NVLink")
        
        # Determine communication speed overhead penalty (in ms)
        comm_latency = 0.002 if interconnect == "NVLink" else 0.050

        for event in self.trace.get("traceEvents", []):
            event_type = event.get("type")
            name = event.get("name", "unknown_kernel")
            
            if event_type == "COMPUTE":
                # Scale execution duration based on simulated TFLOPS capability vs baseline trace
                baseline_tflops = event.get("baseline_tflops", 312)
                raw_duration = event.get("duration_ms", 1.0)
                simulated_duration = raw_duration * (baseline_tflops / (tflops * gpus))
                
                self.virtual_clock_ms += simulated_duration
                self.execution_logs.append(f"[{self.virtual_clock_ms:.3f} ms] Executed compute kernel: {name} ({simulated_duration:.3f} ms)")
                
            elif event_type == "COMMUNICATION":
                # Model collective communications penalty (e.g., AllReduce across nodes)
                data_size_mb = event.get("data_size_mb", 10)
                transfer_delay = (data_size_mb / 1000.0) + comm_latency
                
                self.virtual_clock_ms += transfer_delay
                self.execution_logs.append(f"[{self.virtual_clock_ms:.3f} ms] Synchronized GPU cluster via {interconnect}: {name} ({transfer_delay:.3f} ms)")

        return {
            "total_simulated_time_ms": self.virtual_clock_ms,
            "timeline": self.execution_logs
        }

# --- Phase 2: Micro-Architectural Analytical Scaling Layer ---
class FakeLLMScaler:
    def __init__(self, config):
        self.config = config

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

        # Compute topology configurations
        tp = g if g <= 8 else 8
        pp = max(1, g // 8)
        
        comm_efficiency = max(0.4, 0.95 - (0.03 * tp) - ((pp - 1) / max(1, batch / 2)))

        base_throughput = ((g * tflops * 40) / m) * (batch ** 0.6) * (2000 / (2000 + seq))
        throughput = base_throughput * comm_efficiency
        latency = ((seq * m) / (g * 2500)) / comm_efficiency

        return {
            "model": self.config["model"]["name"],
            "topology": f"{g}x {self.config['hardware']['gpu_type']}",
            "comm_efficiency_pct": comm_efficiency * 100,
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

    md = ["### 🚦 `llm-twin` Architectural Simulation Matrix", 
          "Automated performance validation running Phase 2 & 3 Emulation Pipeline.\n",
          "| Profile Name | Cluster Setup | Comm Efficiency | VRAM Allocation | Throughput | Status |",
          "| :--- | :--- | :--- | :--- | :--- | :--- |"]
    
    for r in results:
        vram_string = f"{r['required_mem_gb']:.1f} / {r['total_mem_gb']} GB ({r['mem_util_pct']:.1f}%)"
        status = "✅ PASSED" if r["fits"] else "❌ CRITICAL OOM"
        md.append(f"| `{r['filename']}` | {r['topology']} | {r['comm_efficiency_pct']:.1f}% | {vram_string} | {r['throughput_tok_sec']:,.1f} tok/s | {status} |")
    
    return "\n".join(md)

def main():
    parser = argparse.ArgumentParser(description="llm-twin orchestration engine")
    sub = parser.add_subparsers(dest="command", required=True)
    
    # Core Subcommands
    sim = sub.add_parser("simulate")
    sim.add_argument("config")
    
    matrix = sub.add_parser("matrix")
    matrix.add_argument("directory")

    # New Phase 3 Trace Ingestion Subcommand
    trace_cmd = sub.add_parser("emulate-trace")
    trace_cmd.add_argument("--trace", required=True, help="Path to JSON execution trace file")
    trace_cmd.add_argument("--hardware", required=True, help="Target topology configuration YAML")

    args = parser.parse_args()

    if args.command == "matrix":
        print(generate_markdown_matrix(args.directory))
    elif args.command == "simulate":
        cfg = parse_simple_yaml(args.config)
        if cfg:
            m = FakeLLMScaler(cfg).simulate()
            print(f"Throughput Output: {m['throughput_tok_sec']:.2f} tok/s | VRAM Fits: {m['fits']}")
    elif args.command == "emulate-trace":
        hw_cfg = parse_simple_yaml(args.hardware)
        try:
            with open(args.trace, 'r') as f:
                trace_json = json.load(f)
        except Exception as e:
            print(f"Error reading trace file: {e}")
            sys.exit(1)
            
        if hw_cfg and trace_json:
            report = TraceEmulator(trace_json, hw_cfg).run()
            print(f"\n🎯 --- Trace-Driven Simulation Complete ---")
            print(f"Total Virtual Execution Latency: {report['total_simulated_time_ms']:.4f} ms\n")
            print("🕒 Timeline Step Breakdown:")
            for log in report["timeline"][:10]: # Print first 10 steps for terminal neatness
                print(f"  {log}")

if __name__ == "__main__":
    main()