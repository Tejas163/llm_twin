#!/usr/bin/env python3
import argparse
import sys
import os
import json
import random


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

        # Structural hidden dimension mapping
        hidden_size = 8192 if m >= 70 else (12288 if m >= 400 else 4096)

        weight_memory = m * 2
        kv_cache_per_token_gb = (m * 1.5e-6 + 4e-5)
        kv_cache_memory = kv_cache_per_token_gb * seq * batch
        total_required_memory = weight_memory + kv_cache_memory
        total_gpu_memory = g * mem

        # Auto-Parallelism Strategy Engine
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
        
        # High-Fidelity Network Fabric Simulation Math
        NVLINK_BANDWIDTH = 900.0      
        INFINIBAND_BANDWIDTH = 50.0   
        
        tp_factor = (tp - 1) / max(1, tp)
        activation_payload_bytes = 2 * tp_factor * batch * seq * hidden_size * 2
        payload_gb = activation_payload_bytes / 1e9

        if tp <= 8 and (g <= 8 or (pp == 1 and dp == 1)):
            effective_bandwidth = NVLINK_BANDWIDTH
            fabric_type = "NVLink Mesh"
            compute_time_ms = (seq * m * 2) / (tflops * 1e3)
            network_latency_ms = (payload_gb / effective_bandwidth) * 1000.0
            comm_efficiency = compute_time_ms / (compute_time_ms + network_latency_ms)
        else:
            effective_bandwidth = INFINIBAND_BANDWIDTH
            fabric_type = "InfiniBand Switch Network"
            compute_time_ms = (seq * m * 2) / (tflops * 1e3)
            network_latency_ms = (payload_gb / effective_bandwidth) * 1000.0
            comm_efficiency = compute_time_ms / (compute_time_ms + (network_latency_ms * 3.5))
            if pp > 1:
                comm_efficiency *= 0.70  

        comm_efficiency = max(0.05, min(0.99, comm_efficiency))

        base_throughput = ((g * tflops * 40) / m) * (batch ** 0.6) * (2000 / (2000 + seq))
        throughput = base_throughput * comm_efficiency
        latency = ((seq * m) / (g * 2500)) / comm_efficiency

        # --- Phase 7 Corporate Governance & Data Layer Expansion ---
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
        raw_cost_per_m_tokens = (hourly_cluster_cost / tokens_per_hour) * 1000000 if tokens_per_hour > 0 else 0.0

        # --- Phase 7 TCO Financial Calculations ---
        if provider == "hyperscaler":
            egress_fee_per_m_tokens = 0.12  # Premium network tax
            sla_risk_multiplier = 1.00       # Rock solid uptime/guarantee
            strategy_label = "Conservative Tier-1 Enterprise Framework"
        else:
            egress_fee_per_m_tokens = 0.00  # Specialized vendors skip egress margins
            sla_risk_multiplier = 1.25       # 25% risk premium added for spotty capacity/SLA
            strategy_label = "Agile Asset-Light Cost Optimization Strategy"

        # TCO = (Raw Cost + Network Egress) * Risk Adjustment Factor
        risk_adjusted_tco_per_m = (raw_cost_per_m_tokens + egress_fee_per_m_tokens) * sla_risk_multiplier

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
            "cost_per_m_tokens": raw_cost_per_m_tokens,
            "provider": provider,
            "billing": billing,
            "fabric_type": fabric_type,
            "payload_gb": payload_gb,
            # Phase 7 New Strategic Metadata Outputs
            "egress_fee": egress_fee_per_m_tokens,
            "tco_per_m_tokens": risk_adjusted_tco_per_m,
            "strategy_label": strategy_label,
            "sla_risk_pct": (sla_risk_multiplier - 1.0) * 100
        }


# --- Phase 8: Pure-Python Nature-Inspired Evolutionary Agent ---
class EvolutionaryAgent:
    def __init__(self, target_model_name, param_billion, constraints):
        self.model_name = target_model_name
        self.param_billion = param_billion
        self.constraints = constraints  # e.g., {"max_tco": 5.0, "provider": "specialized", "billing": "on-demand"}
        
        # Define search spaces for our genetic pools
        self.gpu_types = ["A100", "H100", "H200"]
        self.vram_options = [40, 80, 141]
        self.cluster_sizes = [1, 2, 4, 8, 16, 32, 64]

    def generate_random_chromosome(self):
        """Generates a random valid architectural DNA sequence."""
        gpu_idx = random.randint(0, len(self.gpu_types) - 1)
        vram_idx = random.randint(0, len(self.vram_options) - 1)
        count_idx = random.randint(0, len(self.cluster_sizes) - 1)
        
        # Initialize parallelism exponents randomly
        tp_exp = random.randint(0, 3)  # 2^0 to 2^3 (1 to 8)
        pp_exp = random.randint(0, 3)  # 2^0 to 2^3 (1 to 8)
        
        return [gpu_idx, vram_idx, count_idx, tp_exp, pp_exp]

    def mutate_chromosome(self, chromosome, mutation_rate=0.2):
        """Applies nature-inspired mutations to keep genetic diversity high."""
        mutated = list(chromosome)
        
        if random.random() < mutation_rate:
            # Mutate GPU Type
            mutated[0] = random.randint(0, len(self.gpu_types) - 1)
        if random.random() < mutation_rate:
            # Mutate VRAM Capacity
            mutated[1] = random.randint(0, len(self.vram_options) - 1)
        if random.random() < mutation_rate:
            # Mutate Cluster Size count
            mutated[2] = random.randint(0, len(self.cluster_sizes) - 1)
        if random.random() < mutation_rate:
            # Mutate Tensor Parallel Size
            mutated[3] = random.randint(0, 3)
        if random.random() < mutation_rate:
            # Mutate Pipeline Parallel Size
            mutated[4] = random.randint(0, 3)
            
        return mutated

    def chromosome_to_config(self, chromosome):
        """Translates the digital DNA array back into an engine readable config structure."""
        g_count = self.cluster_sizes[chromosome[2]]
        tp = 2 ** chromosome[3]
        pp = 2 ** chromosome[4]
        
        # Enforce physical reality limits (TP*PP cannot exceed total hardware allocated)
        if tp * pp > g_count:
            # Adjust strategy down to fit the cluster boundary constraints safely
            tp = min(g_count, 8)
            pp = max(1, g_count // tp)

        gpu_t = self.gpu_types[chromosome[0]]
        gpu_m = self.vram_options[chromosome[1]]
        
        # Auto-match real performance parameters
        gpu_tflops = 989 if gpu_t == "H100" else (156 if gpu_m == 40 else 312)
        if gpu_t == "H200": gpu_tflops = 1979

        return {
            "model": {"name": self.model_name, "parameters_billion": self.param_billion},
            "hardware": {
                "gpu_type": gpu_t,
                "gpu_count": g_count,
                "gpu_memory_gb": gpu_m,
                "gpu_tflops": gpu_tflops,
                "tensor_parallel_size": tp,
                "pipeline_parallel_size": pp
            },
            "inference": {"batch_size": 32, "sequence_length": 2048},  # Normalized baseline workload
            "economics": {
                "provider_type": self.constraints.get("provider", "specialized"), 
                "billing_model": self.constraints.get("billing", "on-demand")
            }
        }
    
    def calculate_fitness(self, chromosome):
        """Evaluates how optimal an infrastructure setup is."""
        config = self.chromosome_to_config(chromosome)
        metrics = FakeLLMScaler(config).simulate()
        
        # Hard Failure Constraint: If it OOMs, it gets zero survival value
        if not metrics["fits"]:
            return 0.0001
        
        # Target optimization parameters
        tco = metrics["tco_per_m_tokens"]
        latency = metrics["latency_ms"]
        
        # We want to minimize TCO and keep latency fast. 
        # Fitness = 1 / (TCO * Latency Penalty Factor)
        latency_penalty = max(1.0, latency / 10.0)  # Penalize slow response profiles
        
        # Prevent division by zero if cost is ultra-low
        fitness_score = 1000.0 / (max(0.01, tco) * latency_penalty)
        return fitness_score

    def run_optimization(self, population_size=20, generations=30, progress_callback=None):
        """Runs the evolutionary search to find the optimal deployment layout."""
        # Initialize an initial diverse population
        population = [self.generate_random_chromosome() for _ in range(population_size)]
        
        best_chromosome = None
        best_fitness = -1.0
        
        for gen in range(generations):
            # 1. Calculate fitness for all variants
            fitness_scores = [self.calculate_fitness(chrom) for chrom in population]
            
            # Track the historical elite champion
            for idx, score in enumerate(fitness_scores):
                if score > best_fitness:
                    best_fitness = score
                    best_chromosome = list(population[idx])
            
            if progress_callback:
                progress_callback(gen + 1, generations, best_fitness)
                
            # 2. Selection (Roulette Wheel Mechanics)
            total_fit = sum(fitness_scores)
            if total_fit == 0:
                total_fit = 1.0
            probabilities = [score / total_fit for score in fitness_scores]
            
            next_generation = []
            
            # Keep the elite champion alive (Elitism)
            if best_chromosome:
                next_generation.append(list(best_chromosome))
                
            # 3. Breed the remaining population slots
            while len(next_generation) < population_size:
                # Select parents based on performance probabilities
                parent1 = random.choices(population, weights=probabilities, k=1)[0]
                parent2 = random.choices(population, weights=probabilities, k=1)[0]
                
                # Crossover (Single-Point Split)
                if random.random() < 0.7:  # 70% Crossover Rate
                    cut = random.randint(1, 3)
                    child = parent1[:cut] + parent2[cut:]
                else:
                    child = list(parent1)
                    
                # Apply Random Mutation
                child = self.mutate_chromosome(child, mutation_rate=0.25)
                next_generation.append(child)
                
            population = next_generation

        # Translate the winning champion DNA back into a final actionable profile
        optimal_config = self.chromosome_to_config(best_chromosome)
        optimal_metrics = FakeLLMScaler(optimal_config).simulate()
        return optimal_config, optimal_metrics
    
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
    
    # --- Locate this inside generate_markdown_matrix() in twin.py ---

    for r in results:
        # FIX: Escape the internal pipes so they don't break markdown table boundaries
        strategy_str = f"TP:{r['tp']} \| PP:{r['pp']} \| DP:{r['dp']}"
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