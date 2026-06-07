#!/usr/bin/env python3
import argparse
import sys
import os
import json

# --- Pure-Python Lightweight YAML Parser ---
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

# --- Phase 3: Advanced Discrete-Event Trace Emulator ---
class TraceEmulator:
    def __init__(self, trace_data, hardware_config):
        self.trace = trace_data
        self.hw = hardware_config
        self.virtual_clock_ms = 0.0
        self.execution_logs = []
        self.max_vram_observed = 0.0
        self.initial_timestamp_us = None

    def run(self):
        """Emulates production Chrome Trace / PyTorch Profiler json structures."""
        target_tflops = self.hw.get("hardware", {}).get("gpu_tflops", 312)
        target_gpus = self.hw.get("hardware", {}).get("gpu_count", 1)
        total_vram_limit = target_gpus * self.hw.get("hardware", {}).get("gpu_memory_gb", 80)
        
        baseline_gpu_tflops = 312 
        events = self.trace.get("traceEvents", [])
        
        for event in events:
            name = event.get("name", "unnamed_event")
            category = event.get("cat", "")
            phase = event.get("ph", "")
            ts_us = event.get("ts", None)

            if ts_us is not None and self.initial_timestamp_us is None:
                self.initial_timestamp_us = ts_us

            if phase == "X":
                raw_duration_us = event.get("dur", 0)
                raw_duration_ms = raw_duration_us / 1000.0
                
                if ts_us is not None:
                    relative_start_ms = (ts_us - self.initial_timestamp_us) / 1000.0
                    if relative_start_ms > self.virtual_clock_ms:
                        self.virtual_clock_ms = relative_start_ms

                if category == "gpu":
                    simulated_duration = raw_duration_ms * (baseline_gpu_tflops / (target_tflops * target_gpus))
                else:
                    simulated_duration = raw_duration_ms

                self.virtual_clock_ms += simulated_duration
                self.execution_logs.append(
                    f"[{self.virtual_clock_ms:.3f} ms] Completed [{category.upper()}] Kernel: {name} (Simulated Duration: {simulated_duration:.3f} ms)"
                )
            
            elif phase == "C":
                args = event.get("args", {})
                value = args.get("value", 0)
                
                if ts_us is not None:
                    relative_start_ms = (ts_us - self.initial_timestamp_us) / 1000.0
                    if relative_start_ms > self.virtual_clock_ms:
                        self.virtual_clock_ms = relative_start_ms

                if name == "GPU_Memory_Used_GB":
                    if value > self.max_vram_observed:
                        self.max_vram_observed = value
                    
                    status_flag = "🚨 OVERFLOW FAILURE" if value > total_vram_limit else "STABLE"
                    self.execution_logs.append(
                        f"[{self.virtual_clock_ms:.3f} ms] [COUNTER] {name} changed to {value} GB / {total_vram_limit} GB -> Capacity: {status_flag}"
                    )

        return {
            "total_simulated_time_ms": self.virtual_clock_ms,
            "peak_vram_tracked_gb": self.max_vram_observed,
            "vram_limit_gb": total_vram_limit,
            "timeline": self.execution_logs
        }

# --- Phase 2: Micro-Architectural Scaling Engine ---
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

        # --- Explicit Phase 2 Auto-Parallelism Heuristics ---
        tp = self.config.get("hardware", {}).get("tensor_parallel_size", 0)
        pp = self.config.get("hardware", {}).get("pipeline_parallel_size", 0)
        
        if tp == 0 or pp == 0:
            if g <= 8:
                tp = g
                pp = 1
            else:
                tp = 8  # Scale out to full single-node NVLink boundary first
                pp = max(1, g // 8)
        
        dp = max(1, g // (tp * pp))
        
        # Interconnect Overhead Modeling
        interconnect = self.config.get("hardware", {}).get("interconnect_type", "NVLink" if tp <= 8 else "InfiniBand")
        comm_efficiency = 0.95
        if tp > 1:
            comm_efficiency -= (0.03 * tp)
        if pp > 1:
            bubble_penalty = (pp - 1) / max(1, (batch / dp))
            comm_efficiency -= min(0.3, bubble_penalty)
            
        comm_efficiency = max(0.4, comm_efficiency)

        base_throughput = ((g * tflops * 40) / m) * (batch ** 0.6) * (2000 / (2000 + seq))
        throughput = base_throughput * comm_efficiency
        latency = ((seq * m) / (g * 2500)) / comm_efficiency

        return {
            "model": self.config["model"]["name"],
            "topology": f"{g}x {self.config['hardware']['gpu_type']}",
            "tp": tp,
            "pp": pp,
            "dp": dp,
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

    # --- Phase 2 Columns Visibly Added to GitHub PR Reports ---
    md = [
        "### 🚦 `llm-twin` Architectural Simulation Matrix", 
        "Automated performance validation running Phase 2 Distributed Parallelism & Phase 3 Emulation Pipeline.\n",
        "| Profile Name | Cluster Setup | Parallel Strategy | Comm Eff | VRAM Allocation | Throughput | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    
    for r in results:
        vram_string = f"{r['required_mem_gb']:.1f} / {r['total_mem_gb']} GB ({r['mem_util_pct']:.1f}%)"
        strategy_str = f"TP:{r['tp']} \| PP:{r['pp']} \| DP:{r['dp']}"
        status = "✅ PASSED" if r["fits"] else "❌ CRITICAL OOM"
        md.append(f"| `{r['filename']}` | {r['topology']} | {strategy_str} | {r['comm_efficiency_pct']:.1f}% | {vram_string} | {r['throughput_tok_sec']:,.1f} tok/s | {status} |")
    
    return "\n".join(md)

def main():
    parser = argparse.ArgumentParser(description="llm-twin orchestration engine")
    sub = parser.add_subparsers(dest="command", required=True)
    
    sim = sub.add_parser("simulate")
    sim.add_argument("config")
    
    matrix = sub.add_parser("matrix")
    matrix.add_argument("directory")

    trace_cmd = sub.add_parser("emulate-trace")
    trace_cmd.add_argument("--trace", required=True, help="Path to JSON trace file")
    trace_cmd.add_argument("--hardware", required=True, help="Target configuration YAML")

    args = parser.parse_args()

    if args.command == "matrix":
        print(generate_markdown_matrix(args.directory))
    elif args.command == "simulate":
        cfg = parse_simple_yaml(args.config)
        if cfg:
            m = FakeLLMScaler(cfg).simulate()
            print(f"\n📈 --- Phase 2 Distributed Simulation Details ---")
            print(f"Topology Strategy : TP={m['tp']}, PP={m['pp']}, DP={m['dp']}")
            print(f"Interconnect Eff  : {m['comm_efficiency_pct']:.2f}%")
            print(f"Throughput Output : {m['throughput_tok_sec']:.2f} tok/s")
            print(f"VRAM Memory Status: {'Stable' if m['fits'] else 'OOM Failure'}")
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
            print(f"Total Virtual Execution Latency: {report['total_simulated_time_ms']:.4f} ms")
            print(f"Peak Observed VRAM Context    : {report['peak_vram_tracked_gb']} GB / {report['vram_limit_gb']} GB")
            print("\n🕒 Timeline Step Breakdown:")
            for log in report["timeline"]:
                print(f"  {log}")

if __name__ == "__main__":
    main()