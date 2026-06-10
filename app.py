import streamlit as st  # type: ignore[import]
import os
# Import your verified engine classes directly from your production code
from twin import FakeLLMScaler, EvolutionaryAgent
from storage import init_db, save_scenario, list_scenarios, load_scenario_by_name

# --- Initialize Local Storage ---
init_db()

# --- Page Custom Configuration ---
st.set_page_config(
    page_title="llm-twin | Infrastructure & FinOps Studio",
    page_icon="🚦",
    layout="wide"
)

# --- Initialize Genetic State Engine Keys ---
if "agent_optimized" not in st.session_state:
    st.session_state["agent_optimized"] = False
if "gpu_type_opt" not in st.session_state:
    st.session_state["gpu_type_opt"] = "H100"
if "gpu_count_opt" not in st.session_state:
    st.session_state["gpu_count_opt"] = 8
if "gpu_mem_opt" not in st.session_state:
    st.session_state["gpu_mem_opt"] = 80
if "tp_opt" not in st.session_state:
    st.session_state["tp_opt"] = 8
if "pp_opt" not in st.session_state:
    st.session_state["pp_opt"] = 1

st.title("🚦 llm-twin: Distributed Architecture & FinOps Studio")
st.markdown("""
This interactive simulator evaluates LLM distributed inference strategies, communications efficiency bottlenecks, 
and multi-cloud compute unit economics across specialized and legacy hyperscaler networks.
""")

st.sidebar.header("🛠️ Model Configuration Hub")

# 1. Model Provider Dictionary Mapping
model_catalog = {
    "Meta (Open Source)": {
        "llama-3-8b": 8,
        "llama-3-70b": 70,
        "llama-3-405b": 405
    },
    "Mistral AI": {
        "mistral-7b": 7,
        "codestral-22b": 22,
        "mistral-large": 123
    },
    "DeepSeek": {
        "deepseek-coder-7b": 7,
        "deepseek-math-7b": 7,
        "deepseek-llm-67b": 67
    }
}

# Cascading dropdowns for model definitions
selected_provider = st.sidebar.selectbox("Model Framework Provider", list(model_catalog.keys()))
available_models = list(model_catalog[selected_provider].keys())
selected_model_name = st.sidebar.selectbox("Target Model Architecture", available_models)

# Automatically extract the correct parameter scale size from the map
param_size = model_catalog[selected_provider][selected_model_name]
st.sidebar.caption(f"Parameter scale auto-resolved to: **{param_size} Billion** parameters.")

st.sidebar.markdown("---")
st.sidebar.header("💻 Hardware Cluster Topology")

gpu_type = st.sidebar.selectbox("Accelerator Variant", ["H100", "A100", "H200"])
gpu_mem = st.sidebar.selectbox("VRAM Capacity Allocation", [40, 80, 141], index=1)

# Match real market baseline hardware performance specifications
gpu_tflops = 989 if gpu_type == "H100" else (156 if gpu_mem == 40 else 312)
if gpu_type == "H200": gpu_tflops = 1979

gpu_count = st.sidebar.slider("Total Cluster Accelerator Count", 1, 64, 8, step=1)

st.sidebar.markdown("---")
st.sidebar.header("📈 Inference Workload Profile")
batch_size = st.sidebar.slider("Active Concurrent Batch Size", 1, 256, 32, step=1)
seq_len = st.sidebar.slider("Sequence Tail Length Context", 512, 8192, 2048, step=512)

st.sidebar.markdown("---")
st.sidebar.header("💰 FinOps Brokerage Setup")

# 2. Cloud Provider Dropdown Layout Mappings
cloud_mapping = {
    "Amazon Web Services (AWS)": "hyperscaler",
    "Google Cloud Platform (GCP)": "hyperscaler",
    "Microsoft Azure": "hyperscaler",
    "CoreWeave Cloud": "specialized",
    "Lambda Labs": "specialized"
}

selected_cloud = st.sidebar.selectbox("Target Cloud Deployment Environment", list(cloud_mapping.keys()))
provider_type = cloud_mapping[selected_cloud]

billing_model = st.sidebar.radio("Contract Procurement Type", ["on-demand", "reserved"])

# --- Nature-Inspired Architecture Agent ---
st.sidebar.markdown("---")
st.sidebar.header("🤖 Nature-Inspired Architecture Agent")
st.sidebar.write("Let the Evolutionary Agent find the absolute mathematically optimal cluster configuration for this workload.")

# Choose optimization targets
optimization_metric = st.sidebar.selectbox("Optimization Priority", ["Minimize TCO (Balanced Latency)", "Max Speed (Under Cap)"])

if st.sidebar.button("Run Evolutionary Agent Optimizer", type="primary"):
    st.sidebar.info("🧬 Starting Genetic Algorithm Initialization...")
    
    # Establish optimization boundaries dynamically
    agent_constraints = {
        "provider": provider_type,
        "billing": billing_model,
        "priority": optimization_metric
    }
    
    # Instantiate the agent
    agent = EvolutionaryAgent(
        target_model_name=selected_model_name,
        param_billion=param_size,
        constraints=agent_constraints
    )
    
    # Create visual placeholders for live generation tracking
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    # Define a callback hook to feed loop metrics back into Streamlit real-time
    def update_evolution_progress(current_gen, total_gens, best_fit):
        pct = int((current_gen / total_gens) * 100)
        progress_bar.progress(pct)
        status_text.caption(f"Gen {current_gen}/{total_gens} | Top Fitness Score: {best_fit:.2f}")

    # Fire up the search loop across generations
    with st.spinner("Breeding optimal cluster strategies..."):
        opt_config, opt_metrics = agent.run_optimization(
            population_size=20, 
            generations=30, 
            progress_callback=update_evolution_progress
        )
    
    st.sidebar.success("✅ Optimal Configuration Discovered!")
    
    # Toggle optimized layout status to True to unlock multi-screen view
    st.session_state["agent_optimized"] = True
    
    # Store the agent's recommended parameters in session state
    st.session_state["gpu_type_opt"] = opt_config["hardware"]["gpu_type"]
    st.session_state["gpu_count_opt"] = opt_config["hardware"]["gpu_count"]
    st.session_state["gpu_mem_opt"] = opt_config["hardware"]["gpu_memory_gb"]
    st.session_state["tp_opt"] = opt_config["hardware"]["tensor_parallel_size"]
    st.session_state["pp_opt"] = opt_config["hardware"]["pipeline_parallel_size"]
    
    # Render an explicit alert card showcasing the winning chromosome's traits
    st.sidebar.markdown("### 🏆 Winner DNA Blueprint")
    st.sidebar.json({
        "Recommended Cluster": f"{opt_config['hardware']['gpu_count']}x {opt_config['hardware']['gpu_type']} ({opt_config['hardware']['gpu_memory_gb']}GB)",
        "Parallel Strategy": f"TP={opt_config['hardware']['tensor_parallel_size']} | PP={opt_config['hardware']['pipeline_parallel_size']}",
        "Projected Raw Cost/M": f"${opt_metrics['cost_per_m_tokens']:.4f}",
        "Risk-Adjusted TCO/M": f"${opt_metrics['tco_per_m_tokens']:.4f}",
        "Fabric Type": opt_metrics["fabric_type"]
    })

    # --- Scroll down to the very bottom of the sidebar definition code ---
# Locate right below the "Winner DNA Blueprint" card block, before main navigation logic

st.sidebar.markdown("---")
st.sidebar.header("💾 Workspace Snapshot Manager")
st.sidebar.write("Persist your active configurations and FinOps evaluations to the local team vault.")

# Gather enterprise metadata tags
with st.sidebar.form("snapshot_form", clear_on_submit=True):
    snapshot_name = st.text_input("Scenario Snapshot Name", placeholder="e.g., Llama-3 70B Production Scale")
    project_tag = st.text_input("Project / Team Tag", value="Core Infrastructure")
    
    # Determine which state payload to snapshot (Manual vs Agent Optimized)
    target_source = st.radio("Capture Strategy Target", ["Active Manual Workspace", "Agent Optimized Champion"])
    
    submit_snapshot = st.form_submit_button("Commit Scenario to Vault", type="secondary")

if submit_snapshot:
    if not snapshot_name.strip():
        st.sidebar.error("❌ Scenario Snapshot Name cannot be blank!")
    else:
        # Resolve the active payload configuration based on user selection
        if target_source == "Agent Optimized Champion" and st.session_state["agent_optimized"]:
            cfg_to_save = {
                "model": {"name": selected_model_name, "parameters_billion": param_size},
                "hardware": {
                    "gpu_type": st.session_state["gpu_type_opt"],
                    "gpu_count": st.session_state["gpu_count_opt"],
                    "gpu_memory_gb": st.session_state["gpu_mem_opt"],
                    "gpu_tflops": 1979 if st.session_state["gpu_type_opt"] == "H200" else (989 if st.session_state["gpu_type_opt"] == "H100" else 312)
                },
                "inference": {"batch_size": batch_size, "sequence_length": seq_len},
                "economics": {"provider_type": provider_type, "billing_model": billing_model}
            }
        else:
            # Fallback to current manual configuration metrics
            cfg_to_save = {
                "model": {"name": selected_model_name, "parameters_billion": param_size},
                "hardware": {
                    "gpu_type": gpu_type, "gpu_count": gpu_count, "gpu_memory_gb": gpu_mem, "gpu_tflops": gpu_tflops
                },
                "inference": {"batch_size": batch_size, "sequence_length": seq_len},
                "economics": {"provider_type": provider_type, "billing_model": billing_model}
            }
            
        # Run a quick high-fidelity simulation to ensure saved outputs are fresh
        metrics_to_save = FakeLLMScaler(cfg_to_save).simulate()
        
        # Dispatch to the SQLite core database engine
        success, message = save_scenario(
            name=snapshot_name.strip(),
            project_tag=project_tag.strip(),
            model_name=selected_model_name,
            param_billion=param_size,
            config_dict=cfg_to_save,
            metrics_dict=metrics_to_save
        )
        
        if success:
            st.sidebar.success(f"✅ {message}")
            # Force front-end redraw to ensure downstream tables pick up the new records instantly
            st.rerun()
        else:
            st.sidebar.error(message)

# --- Main Dashboard Core Navigation Hub ---
# --- Main Dashboard Core Navigation Hub ---
if st.session_state["agent_optimized"]:
    # Expose all three major corporate tabs if an agent run exists
    tab_manual, tab_agent, tab_vault = st.tabs([
        "📋 Manual Configuration Workspace", 
        "🧬 Agent Optimized Sandbox View", 
        "🏛️ Saved Blueprints Vault"
    ])
else:
    # Expose manual and vault tabs by default on clean boot
    tab_manual, tab_vault = st.tabs([
        "📋 Manual Configuration Workspace", 
        "🏛️ Saved Blueprints Vault"
    ])
    tab_agent = None

# ==================== SCREEN 1: MANUAL WORKSPACE ====================
with tab_manual:
    mock_config = {
        "model": {"name": selected_model_name, "parameters_billion": param_size},
        "hardware": {
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "gpu_memory_gb": gpu_mem,
            "gpu_tflops": gpu_tflops
        },
        "inference": {"batch_size": batch_size, "sequence_length": seq_len},
        "economics": {"provider_type": provider_type, "billing_model": billing_model}
    }

    metrics = FakeLLMScaler(mock_config).simulate()

    # --- Main Dashboard Metrics Panels Layout ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if metrics["fits"]:
            st.success("🚨 Memory Status: STABLE")
        else:
            st.error("🚨 Memory Status: CRITICAL OOM")
    with col2:
        st.metric("Total Throughput", f"{metrics['throughput_tok_sec']:,.2f} tok/s")
    with col3:
        st.metric("Hourly Run Rate", f"${metrics['hourly_cost']:.2f} / hr")
    with col4:
        st.metric("Cost / Million Tokens", f"${metrics['cost_per_m_tokens']:.4f}")

    st.markdown("---")

    # --- Deep Architectural Breakdown Sections ---
    left_panel, right_panel = st.columns(2)
    with left_panel:
        st.subheader("📊 Distributed Fabric Topologies")
        if metrics["fabric_type"] == "NVLink Mesh":
            st.success(f"🔗 Active Interconnect Fabric: {metrics['fabric_type']} (900 GB/s)")
        else:
            st.warning(f"🚨 Active Interconnect Fabric: {metrics['fabric_type']} (50 GB/s Network Bottleneck)")
        st.info(f"**Auto-Selected Parallel Split Strategy:** TP={metrics['tp']} | PP={metrics['pp']} | DP={metrics['dp']}")
        st.metric("All-Reduce Layer Payload Size", f"{metrics['payload_gb']:.4f} GB")
        st.write(f"**Calculated Fabric Scaling Efficiency:** {metrics['comm_efficiency_pct']:.2f}%")
        st.progress(int(metrics['comm_efficiency_pct']))

    with right_panel:
        st.subheader("💾 VRAM Allocation Breakdown")
        util_percentage = min(100.0, metrics['mem_util_pct'])
        st.write(f"**Aggregate Memory Threshold:** {metrics['required_mem_gb']:.2f} GB / {metrics['total_mem_gb']} GB")
        st.progress(int(util_percentage))
        st.write(f"**Active Workspace Utilization Ratio:** {metrics['mem_util_pct']:.2f}%")

    st.markdown("---")
    st.subheader("📁 Generated Infrastructure Blueprint Configuration")
    st.json(mock_config)

    st.markdown("---")
    st.header("👔 Executive Strategy & Capacity Planning Matrix")
    strat_col1, strat_col2, strat_col3 = st.columns(3)
    with strat_col1:
        st.subheader("📌 Corporate Alignment")
        st.info(f"**Active Operational Posture:** \n\n {metrics['strategy_label']}")
        st.write(f"**Vendor SLA Risk Premium:** {metrics['sla_risk_pct']:.1f}%")
    with strat_col2:
        st.subheader("💰 True Total Cost of Ownership (TCO)")
        st.metric("Risk-Adjusted Cost / M Tokens", f"${metrics['tco_per_m_tokens']:.4f}")
        st.write(f"**Baseline Raw Compute Cost:** ${metrics['cost_per_m_tokens']:.4f} / M tokens")
        st.write(f"**Network Data Egress Tax:** ${metrics['egress_fee']:.4f} / M tokens")
    with strat_col3:
        st.subheader("📢 Solution Architect Recommendation")
        if not metrics["fits"]:
            st.error("🛑 **HOLD PROVISIONING:** Current model configuration triggers a Critical OOM.")
        elif metrics["tco_per_m_tokens"] > 5.0:
            st.warning("⚠️ **OPEX ALERT:** True TCO is elevated due to egress taxes or fabric bottlenecking.")
        else:
            st.success("🚀 **PROCEED TO DEPLOYMENT:** This architecture configuration represents optimized unit economics.")

# ==================== SCREEN 2: AGENT OPTIMIZED SANDBOX ====================
if tab_agent is not None:
    with tab_agent:
        st.markdown("## 🧬 AI-Discovered Optimal Topology Sandbox")
        st.success(f"This specialized workspace is dynamically locked to the parameters bred by the Evolutionary Agent for **{selected_model_name}**.")
        
        # Override configuration payloads entirely using the agent session state outputs
        opt_config_payload = {
            "model": {"name": selected_model_name, "parameters_billion": param_size},
            "hardware": {
                "gpu_type": st.session_state["gpu_type_opt"],
                "gpu_count": st.session_state["gpu_count_opt"],
                "gpu_memory_gb": st.session_state["gpu_mem_opt"],
                "gpu_tflops": 1979 if st.session_state["gpu_type_opt"] == "H200" else (989 if st.session_state["gpu_type_opt"] == "H100" else 312)
            },
            "inference": {"batch_size": batch_size, "sequence_length": seq_len},
            "economics": {"provider_type": provider_type, "billing_model": billing_model}
        }
        
        opt_metrics = FakeLLMScaler(opt_config_payload).simulate()
        
        # Mirroring the entire architectural layouts with optimized agent parameters
        o_col1, o_col2, o_col3, o_col4 = st.columns(4)
        with o_col1:
            if opt_metrics["fits"]:
                st.success("🚨 Memory Status: STABLE")
            else:
                st.error("🚨 Memory Status: CRITICAL OOM")
        with o_col2:
            st.metric("Optimized Throughput", f"{opt_metrics['throughput_tok_sec']:,.2f} tok/s")
        with o_col3:
            st.metric("Optimized Run Rate", f"${opt_metrics['hourly_cost']:.2f} / hr")
        with o_col4:
            st.metric("Optimized Cost / M Tokens", f"${opt_metrics['cost_per_m_tokens']:.4f}")
            
        st.markdown("---")
        
        o_left, o_right = st.columns(2)
        with o_left:
            st.subheader("📊 Optimized Fabric Topology Breakdown")
            if opt_metrics["fabric_type"] == "NVLink Mesh":
                st.success(f"🔗 Active Interconnect Fabric: {opt_metrics['fabric_type']} (900 GB/s)")
            else:
                st.warning(f"🚨 Active Interconnect Fabric: {opt_metrics['fabric_type']} (50 GB/s Network Bottleneck)")
            st.info(f"**Agent-Selected Strategy:** TP={opt_metrics['tp']} | PP={opt_metrics['pp']} | DP={opt_metrics['dp']}")
            st.metric("All-Reduce Layer Payload Size", f"{opt_metrics['payload_gb']:.4f} GB")
            st.write(f"**Fabric Scaling Efficiency:** {opt_metrics['comm_efficiency_pct']:.2f}%")
            st.progress(int(opt_metrics['comm_efficiency_pct']))
            
        with o_right:
            st.subheader("💾 Optimized VRAM Allocation Breakdown")
            o_util_percentage = min(100.0, opt_metrics['mem_util_pct'])
            st.write(f"**Aggregate Memory Threshold:** {opt_metrics['required_mem_gb']:.2f} GB / {opt_metrics['total_mem_gb']} GB")
            st.progress(int(o_util_percentage))
            st.write(f"**Active Workspace Utilization Ratio:** {opt_metrics['mem_util_pct']:.2f}%")
            
        st.markdown("---")
        st.subheader("📁 Discovered Optimal Infrastructure Blueprint Configuration JSON")
        st.json(opt_config_payload)
        
        st.markdown("---")
        st.header("👔 Optimized Executive Strategy & Capacity Planning Matrix")
        os_col1, os_col2, os_col3 = st.columns(3)
        with os_col1:
            st.subheader("📌 Corporate Alignment")
            st.info(f"**Active Operational Posture:** \n\n {opt_metrics['strategy_label']}")
            st.write(f"**Vendor SLA Risk Premium:** {opt_metrics['sla_risk_pct']:.1f}%")
        with os_col2:
            st.subheader("💰 True Total Cost of Ownership (TCO)")
            st.metric("Risk-Adjusted Cost / M Tokens", f"${opt_metrics['tco_per_m_tokens']:.4f}")
            st.write(f"**Baseline Raw Compute Cost:** ${opt_metrics['cost_per_m_tokens']:.4f} / M tokens")
            st.write(f"**Network Data Egress Tax:** ${opt_metrics['egress_fee']:.4f} / M tokens")
        with os_col3:
            st.subheader("📢 Solution Architect Recommendation")
            if not opt_metrics["fits"]:
                st.error("🛑 **HOLD PROVISIONING:** Current model configuration triggers a Critical OOM.")
            elif opt_metrics["tco_per_m_tokens"] > 5.0:
                st.warning("⚠️ **OPEX ALERT:** True TCO is elevated due to egress taxes or fabric bottlenecking.")
            else:
                st.success("🚀 **PROCEED TO DEPLOYMENT:** This architecture configuration represents optimized unit economics.")

# ==================== SCREEN 3: SAVED BLUEPRINTS VAULT ====================
with tab_vault:
    st.markdown("## 🏛️ Enterprise Saved Blueprints Vault")
    st.markdown("Query, audit, and deep-dive into historical topology snapshots committed by engineering teams.")
    
    # Query fresh records directly from the SQLite database
    saved_records = list_scenarios()
    
    if not saved_records:
        st.info("The local blueprint vault is currently empty. Use the sidebar form to capture a workspace snapshot!")
    else:
        # Format the summary records into a scannable table overview
        st.markdown("### 📋 Historical Scenario Logs")
        st.table(saved_records)
        
        st.markdown("---")
        st.markdown("### 🔍 Deep-Dive Scenario Inspector")
        
        # Dropdown selection to inspect an explicit record
        record_names = [r["name"] for r in saved_records]
        selected_record = st.selectbox("Choose a Scenario Snapshot to Inspect", record_names)
        
        if selected_record:
            # Load json blobs from SQLite storage engine
            saved_cfg, saved_metrics = load_scenario_by_name(selected_record)
            
            if saved_cfg and saved_metrics:
                # Construct 4-column layout mirroring active workspace metrics
                v_col1, v_col2, v_col3, v_col4 = st.columns(4)
                
                with v_col1:
                    if saved_metrics["fits"]:
                        st.success("🚨 Memory Status: STABLE")
                    else:
                        st.error("🚨 Memory Status: CRITICAL OOM")
                        
                v_col2.metric("Archived Throughput", f"{saved_metrics['throughput_tok_sec']:,.2f} tok/s")
                v_col3.metric("Archived Hourly Bill", f"${saved_metrics['hourly_cost']:.2f} / hr")
                v_col4.metric("Risk-Adjusted TCO/M", f"${saved_metrics['tco_per_m_tokens']:.4f}")
                
                # Render technical layout details
                v_panel_l, v_panel_r = st.columns(2)
                with v_panel_l:
                    st.subheader("⚙️ Hardware Architecture")
                    st.json(saved_cfg["hardware"])
                with v_panel_r:
                    st.subheader("💰 FinOps Economics & Routing")
                    st.write(f"**Fabric Interconnect:** {saved_metrics['fabric_type']}")
                    st.write(f"**Procurement Contract:** {saved_metrics['billing'].upper()} ({saved_cfg['economics']['provider_type'].upper()})")
                    st.info(f"**Archived Posture:**\n\n{saved_metrics['strategy_label']}")

# Add a permanent risk governance advisory table for board reviews
st.markdown("---")
st.markdown("### 📊 Multi-Cloud Vendor Risk Concentration Matrix")
st.table([
    {"Dimension": "Tier-1 Hyperscalers (AWS/GCP/Azure)", "Cost Efficiency": "Low (Premium Markups)", "SLA & Uptime": "99.99% Guaranteed", "Egress Fees": "High ($0.12/M tokens tax)", "Strategic Fit": "Highly Conservative / Enterprise Scales"},
    {"Dimension": "Specialized GPU Clouds (CoreWeave/Lambda)", "Cost Efficiency": "High (Bare-Metal Rates)", "SLA & Uptime": "Variable / Volatile", "Egress Fees": "Zero Egress", "Strategic Fit": "Agile / High-Throughput Batch Inference"}
])