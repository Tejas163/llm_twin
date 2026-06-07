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
left_panel, right_panel = st.columns(2)

with left_panel:
    st.subheader("📊 Distributed Cluster Strategies")
    st.info(f"**Auto-Selected Parallel Split Strategy:** TP={metrics['tp']} | PP={metrics['pp']} | DP={metrics['dp']}")
    
    st.write(f"**Interconnect Bus Scaling Efficiency:** {metrics['comm_efficiency_pct']:.2f}%")
    st.progress(int(metrics['comm_efficiency_pct']))
    st.caption(f"Evaluating {selected_cloud} network overhead metrics. Hyperscaler fabrics factor in distinct cross-node pipeline stalls.")

with right_panel:
    st.subheader("💾 VRAM Allocation Breakdown")
    util_percentage = min(100.0, metrics['mem_util_pct'])
    st.write(f"**Aggregate Memory Threshold:** {metrics['required_mem_gb']:.2f} GB / {metrics['total_mem_gb']} GB")
    st.progress(int(util_percentage))
    st.write(f"**Active Workspace Utilization Ratio:** {metrics['mem_util_pct']:.2f}%")

st.markdown("---")
st.subheader("📁 Generated Infrastructure Blueprint Configuration")
st.json(mock_config)