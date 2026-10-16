"""Local, interactive inspection of a real tiny language model."""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import torch

from context_lab.experiment import load_checkpoint
from context_lab.tokenizer import ByteTokenizer

ROOT = Path(__file__).parent
st.set_page_config(page_title="Context Lab", page_icon="◈", layout="wide")
st.markdown(
    """
<style>
  .stApp {background: #faf8f3; color: #223c3b;}
  .block-container {padding-top: 2.5rem; max-width: 1140px;}
  h1 {font-family: Georgia, serif !important; font-size: 4rem !important;
      font-weight: 400 !important; letter-spacing: -0.06em !important;}
  h2, h3 {font-weight: 500 !important;}
  div[data-testid="stMetric"] {background: #efeee6; padding: 20px; border-radius: 8px;}
  .eyebrow {color: #c65331; font: 600 12px monospace; letter-spacing: .17em;}
</style>
<div class="eyebrow">SMALL MODEL · VISIBLE MECHANICS · MEASURED RESULTS</div>
""",
    unsafe_allow_html=True,
)
st.title("Context Lab")
st.markdown(
    "A transformer you can inspect. Explore how a small model learns, "
    "what it attends to, and what changes when its context shrinks."
)


@st.cache_resource
def get_model(path, modified_ns):
    del modified_ns  # Cache invalidates whenever the checkpoint changes.
    torch.set_num_threads(4)
    return load_checkpoint(path)


default_checkpoint = ROOT / "checkpoints/context-128-seed-7.pt"
if not default_checkpoint.exists():
    st.info("Train a model and export its checkpoint to enable live generation. See README.md.")
    st.stop()
model, payload = get_model(str(default_checkpoint), default_checkpoint.stat().st_mtime_ns)
tokenizer = ByteTokenizer()
summary_path = ROOT / "results/reference/summary.json"
summary = json.loads(summary_path.read_text()) if summary_path.exists() else None

left, center, right = st.columns(3)
left.metric("Parameters", f"{model.parameter_count():,}")
center.metric("Training context", f"{payload['train_config']['context_length']} tokens")
right.metric("Vocabulary", "258 tokens")
st.caption(
    "Trained from random initialization on procedural fiction. "
    "Vocabulary: 256 byte values plus BOS/EOS. The toy corpus has recurring templates; "
    "this model has no factual grounding."
)
generate_tab, experiments_tab, attention_tab = st.tabs(
    ["01 / Generate", "02 / Experiments", "03 / Attention"]
)
with generate_tab:
    main, settings = st.columns([2, 1], gap="large")
    with main:
        st.subheader("Continue a story")
        prompt = st.text_area("Prompt", "Mira found a", max_chars=1000, height=120)
        if st.button("Generate continuation", type="primary", use_container_width=True):
            with st.spinner("Predicting one byte at a time…"):
                tokens = torch.tensor([[tokenizer.bos_id, *tokenizer.encode(prompt)]])
                result = model.generate(
                    tokens,
                    max_new_tokens=st.session_state.get("new_tokens", 200),
                    temperature=st.session_state.get("temperature", 0.8),
                    top_k=st.session_state.get("top_k", 40),
                    context_length=st.session_state.get("sample_context", 128),
                    generator=torch.Generator().manual_seed(st.session_state.get("sample_seed", 7)),
                )
                st.session_state["completion"] = tokenizer.decode(result[0].tolist())
        if "completion" in st.session_state:
            st.markdown("**Model output**")
            st.code(st.session_state["completion"], language=None, wrap_lines=True)
    with settings:
        st.subheader("Sampling controls")
        st.slider("Temperature", 0.1, 1.5, 0.8, 0.1, key="temperature")
        st.slider("Top-k", 1, 100, 40, key="top_k")
        st.slider("New tokens", 20, 400, 200, 20, key="new_tokens")
        st.select_slider("Available context", [16, 32, 64, 128], value=128, key="sample_context")
        st.number_input("Random seed", min_value=0, max_value=99999, value=7, key="sample_seed")
        st.caption(
            "Higher temperature spreads probability more evenly. Top-k keeps only "
            "the k most likely next tokens. Reducing context hides earlier prompt bytes."
        )

with experiments_tab:
    st.subheader("Compare the evidence")
    if summary:
        records = [
            {
                "Variant": r["variant"],
                "Seed": r["train_config"]["seed"],
                "Test perplexity": r["test"]["perplexity"],
                "Parameters": r["parameters"],
                "Best step": r["best_step"],
            }
            for r in summary["runs"]
        ]
        frame = pd.DataFrame(records)
        st.dataframe(frame, hide_index=True, use_container_width=True)
        curves = [
            {
                "Variant": r["variant"],
                "Seed": str(r["train_config"]["seed"]),
                "Step": h["step"],
                "Validation loss": h["validation_loss"],
            }
            for r in summary["runs"]
            for h in r["history"]
        ]
        chart = (
            alt.Chart(pd.DataFrame(curves))
            .mark_line()
            .encode(
                x="Step:Q",
                y=alt.Y("Validation loss:Q", title="Validation loss (nats/token)"),
                color="Variant:N",
                strokeDash="Seed:N",
                tooltip=list(curves[0]),
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            f"Count-bigram test perplexity: {summary['bigram']['test']['perplexity']:.3f}. "
            "Every run uses the same target budget. Checkpoints are selected on validation."
        )
        st.info(
            "Exact documents are disjoint, but templates are shared. Context changes also "
            "change batch size and evaluation resets. These results measure this toy task."
        )
    else:
        st.info("Export an experiment suite to display measured results here.")

with attention_tab:
    st.subheader("See the causal mask")
    text = st.text_input("Text to inspect (up to 64 UTF-8 bytes)", "Mira found a red key.")
    layer_col, head_col = st.columns(2)
    layer = layer_col.selectbox("Layer", range(model.config.layers))
    head = head_col.selectbox("Attention head", range(model.config.heads))
    ids = [tokenizer.bos_id, *tokenizer.encode(text)[:64]]
    labels = [
        "BOS" if t == 256 else repr(bytes([t]).decode("utf-8", errors="replace"))[1:-1] for t in ids
    ]
    with torch.inference_mode():
        weights = model.attention_weights(torch.tensor([ids]), layer=layer)[0, head]
    heat = [
        {
            "Query": i,
            "Key": j,
            "Query byte": labels[i],
            "Key byte": labels[j],
            "Weight": weights[i, j].item(),
        }
        for i in range(len(ids))
        for j in range(len(ids))
    ]
    chart = (
        alt.Chart(pd.DataFrame(heat))
        .mark_rect()
        .encode(
            x=alt.X("Key:O", title="Past / current byte position"),
            y=alt.Y("Query:O", title="Query byte position", sort="ascending"),
            color=alt.Color("Weight:Q", scale=alt.Scale(scheme="tealblues", domain=[0, 1])),
            tooltip=[
                "Query",
                "Key",
                "Query byte",
                "Key byte",
                alt.Tooltip("Weight:Q", format=".4f"),
            ],
        )
        .properties(height=450)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "Every row sums to one. Positions above the diagonal have zero weight: "
        "future bytes are masked before softmax. Attention weights show an internal "
        "computation, and do not by themselves explain a prediction."
    )
