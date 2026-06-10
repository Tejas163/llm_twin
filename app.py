import streamlit as st
import os
# Import your verified engine classes directly from your production code
from twin import FakeLLMScaler

# --- Page Custom Configuration ---
st.set_page_config(
    page_title="llm-twin | Infrastructure & FinOps Studio",
    page_icon="🚦",
    layout="wide"
)

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

# --- Interface Core State Mapping ---
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

# Run the backend analytical scaler live on every input mutation
scaler = FakeLLMScaler(mock_config)
metrics = scaler.simulate()

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
# --- Deep Architectural Breakdown Sections ---
left_panel, right_panel = st.columns(2)

with left_panel:
    st.subheader("📊 Distributed Fabric Topologies")
    
    # Showcase active structural wire layers dynamically
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

# Create a clean 3-column layout for enterprise metrics
strat_col1, strat_col2, strat_col3 = st.columns(3)

with strat_col1:
    st.subheader("📌 Corporate Alignment")
    st.info(f"**Active Operational Posture:** \n\n {metrics['strategy_label']}")
    st.write(f"**Vendor SLA Risk Premium:** {metrics['sla_risk_pct']:.1f}%")
    st.caption("A risk premium is mathematically applied to asset-light specialized clouds to model potential capacity availability and volatility overhead.")

with strat_col2:
    st.subheader("💰 True Total Cost of Ownership (TCO)")
    st.metric("Risk-Adjusted Cost / M Tokens", f"${metrics['tco_per_m_tokens']:.4f}")
    st.write(f"**Baseline Raw Compute Cost:** ${metrics['cost_per_m_tokens']:.4f} / M tokens")
    st.write(f"**Network Data Egress Tax:** ${metrics['egress_fee']:.4f} / M tokens")

with strat_col3:
    st.subheader("📢 Solution Architect Recommendation")
    
    # Dynamically generate executive advice based on simulation boundaries
    if not metrics["fits"]:
        st.error("🛑 **HOLD PROVISIONING:** Current model configuration triggers a Critical OOM. Do not sign cloud vendor commitments under these cluster parameters.")
    elif metrics["tco_per_m_tokens"] > 5.0:
        st.warning("⚠️ **OPEX ALERT:** True TCO is elevated due to networking fabric bottlenecks or hyperscaler egress taxes. Recommend renegotiating reserved instance contracts or scaling batch configurations.")
    else:
        st.success("🚀 **PROCEED TO DEPLOYMENT:** This architecture configuration represents optimized unit economics. Financial margins fit cleanly within enterprise production targets.")

# Add a permanent risk governance advisory table for board reviews
st.markdown("### 📊 Multi-Cloud Vendor Risk Concentration Matrix")
st.table([
    {"Dimension": "Tier-1 Hyperscalers (AWS/GCP/Azure)", "Cost Efficiency": "Low (Premium Markups)", "SLA & Uptime": "99.99% Guaranteed", "Egress Fees": "High ($0.12/M tokens tax)", "Strategic Fit": "Highly Conservative / Enterprise Scales"},
    {"Dimension": "Specialized GPU Clouds (CoreWeave/Lambda)", "Cost Efficiency": "High (Bare-Metal Rates)", "SLA & Uptime": "Variable / Volatile", "Egress Fees": "Zero Egress", "Strategic Fit": "Agile / High-Throughput Batch Inference"}
])