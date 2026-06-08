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

# --- Phase 2 & 4: Analytical Hardware & FinOps Broker Engine ---
class FakeLLMScaler:
    def __init__(self, config):
        self.config = config
        
        # Real Market Pricing Matrix Engine (2026 Reference Data)
        self.pricing_matrix = {
            "A100_40GB": {"hyperscaler": 3.25, "specialized": 1.35, "reserved": 0.95},
            "A100": {"hyperscaler": 4.00, "specialized": 1.75, "reserved": 1.30},
            "H100": {"hyperscaler": 5.10, "specialized": 2.45, "reserved": 1.95},
            "H200": {"hyperscaler": 6.85, "specialized": 3.75, "reserved": 2.95}
        }

    def simulate(self):
        m = self.config["model"]["parameters_billion"]
        g = self.config["hardware"]["gpu_count"]
        mem = self.config["hardware"]["gpu_memory_gb"]
        gpu_type = self.config["hardware"]["gpu_type"]
        tflops = self.config["hardware"]["gpu_tflops"]
        batch = self.config["inference"]["batch_size"]
        seq = self.config["inference"]["sequence_length"]

        # Real-world structural hidden dimension mapping (Llama style scales)
        hidden_size = 8192 if m >= 70 else (12288 if m >= 400 else 4096)

        weight_memory = m * 2
        kv_cache_per_token_gb = (m * 1.5e-6 + 4e-5)
        kv_cache_memory = kv_cache_per_token_gb * seq * batch
        total_required_memory = weight_memory + kv_cache_memory
        total_gpu_memory = g * mem

        # --- Phase 2 & 6 Auto-Parallelism Strategy Engine ---
        tp = self.config.get("hardware", {}).get("tensor_parallel_size", 0)
        pp = self.config.get("hardware", {}).get("pipeline_parallel_size", 0)
        
        if tp == 0 or pp == 0:
            if g <= 8:
                tp = g
                pp = 1
            else:
                tp = 8
                pp = max(1, g // 8)
        
        dp = max(1, g // (tp * pp))
        
        # --- Phase 6: High-Fidelity Network Fabric Simulation Math ---
        NVLINK_BANDWIDTH = 900.0      
        INFINIBAND_BANDWIDTH = 50.0   
        
        # Calculate true activation payloads
        tp_factor = (tp - 1) / max(1, tp)
        activation_payload_bytes = 2 * tp_factor * batch * seq * hidden_size * 2
        payload_gb = activation_payload_bytes / 1e9

        # FIX: Check if the strategy or the total cluster size overflows a single 8-GPU node
        if tp <= 8 and (g <= 8 or (pp == 1 and dp == 1)):
            # Communication stays purely inside the ultra-fast internal node
            effective_bandwidth = NVLINK_BANDWIDTH
            fabric_type = "NVLink Mesh"
            
            # Standard high-speed scaling efficiency
            compute_time_ms = (seq * m * 2) / (tflops * 1e3)
            network_latency_ms = (payload_gb / effective_bandwidth) * 1000.0
            comm_efficiency = compute_time_ms / (compute_time_ms + network_latency_ms)
        else:
            # INTERCONNECT BOTTLENECK: Workload crosses nodes over the network interface cards
            effective_bandwidth = INFINIBAND_BANDWIDTH
            fabric_type = "InfiniBand Switch Network"
            
            # Calculate cross-node latency stalls
            compute_time_ms = (seq * m * 2) / (tflops * 1e3)
            network_latency_ms = (payload_gb / effective_bandwidth) * 1000.0
            
            # Apply a harsh network communication penalty for inter-node scaling delays
            comm_efficiency = compute_time_ms / (compute_time_ms + (network_latency_ms * 3.5))
            if pp > 1:
                comm_efficiency *= 0.70  # Model pipeline bubble stalls across the wires

        comm_efficiency = max(0.05, min(0.99, comm_efficiency))
        # Severe penalty if pipeline bubbles are introduced by bad splits across nodes
        if pp > 1 and fabric_type == "InfiniBand Switch Network":
            comm_efficiency *= 0.75 
        
        comm_efficiency = max(0.1, min(0.99, comm_efficiency))

        # Re-map standard outputs using our new high-fidelity fabric parameters
        base_throughput = ((g * tflops * 40) / m) * (batch ** 0.6) * (2000 / (2000 + seq))
        throughput = base_throughput * comm_efficiency
        latency = ((seq * m) / (g * 2500)) / comm_efficiency

        # --- Phase 4 FinOps Pricing Engine ---
        econ = self.config.get("economics", {})
        provider = econ.get("provider_type", "specialized")
        billing = econ.get("billing_model", "on-demand")

        self.pricing_matrix = {
            "A100_40GB": {"hyperscaler": 3.25, "specialized": 1.35, "reserved": 0.95},
            "A100": {"hyperscaler": 4.00, "specialized": 1.75, "reserved": 1.30},
            "H100": {"hyperscaler": 5.10, "specialized": 2.45, "reserved": 1.95},
            "H200": {"hyperscaler": 6.85, "specialized": 3.75, "reserved": 2.95}
        }

        lookup_key = f"{gpu_type}_{mem}GB" if f"{gpu_type}_{mem}GB" in self.pricing_matrix else gpu_type
        rates = self.pricing_matrix.get(lookup_key, {"hyperscaler": 4.0, "specialized": 2.0, "reserved": 1.5})

        if billing == "reserved":
            hourly_gpu_rate = rates["reserved"]
        else:
            hourly_gpu_rate = rates["hyperscaler"] if provider == "hyperscaler" else rates["specialized"]

        hourly_cluster_cost = g * hourly_gpu_rate
        tokens_per_hour = throughput * 3600
        cost_per_m_tokens = (hourly_cluster_cost / tokens_per_hour) * 1000000 if tokens_per_hour > 0 else 0.0

        return {
            "model": self.config["model"]["name"],
            "topology": f"{g}x {gpu_type}",
            "tp": tp, "pp": pp, "dp": dp,
            "comm_efficiency_pct": comm_efficiency * 100,
            "total_mem_gb": total_gpu_memory,
            "required_mem_gb": total_required_memory,
            "mem_util_pct": (total_required_memory / total_gpu_memory) * 100,
            "throughput_tok_sec": throughput,
            "latency_ms": latency,
            "fits": total_required_memory < total_gpu_memory,
            "hourly_cost": hourly_cluster_cost,
            "cost_per_m_tokens": cost_per_m_tokens,
            "provider": provider,
            "billing": billing,
            "fabric_type": fabric_type,
            "payload_gb": payload_gb
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

    # --- Expanded Phase 4 FinOps Matrix Report Layout ---
    md = [
        "### 🚦 `llm-twin` Architectural & FinOps Simulation Matrix", 
        "Automated resource governance validation running Phase 2 Splits, Phase 3 Traces, and Phase 4 Cloud Economic Brokerage.\n",
        "| Profile Name | Cluster Setup | Parallel Strategy | Hourly Bill | Cost / M Tokens | Throughput | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    
    for r in results:
        strategy_str = f"TP:{r['tp']} | PP:{r['pp']} | DP:{r['dp']}"
        cost_str = f"${r['hourly_cost']:.2f}/hr ({r['billing']})"
        m_tokens_str = f"${r['cost_per_m_tokens']:.4f}"
        status = "✅ PASSED" if r["fits"] else "❌ CRITICAL OOM"
        md.append(f"| `{r['filename']}` | {r['topology']} ({r['provider']}) | {strategy_str} | {cost_str} | **{m_tokens_str}** | {r['throughput_tok_sec']:,.1f} tok/s | {status} |")
    
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
            print(f"\n💰 --- Phase 4 FinOps Cost Optimization Broker ---")
            print(f"Provider Profile  : {m['provider'].upper()} ({m['billing'].upper()})")
            print(f"Hourly Run Rate   : ${m['hourly_cost']:.2f} / hr")
            print(f"Efficiency Target : {m['cost_per_m_tokens']:.4f} per Million Tokens")
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

if __name__ == "__main__":
    main()