# ==========================================
# Tool-Orchestrated Agentic AI
# Mutual Fund Analyzer with Visualizations
# ==========================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq

# ------------------------------------------
# Page Configuration
# ------------------------------------------
st.set_page_config(
    page_title="Agentic AI – Mutual Fund Analyzer",
    layout="wide"
)

st.title("🤖 Agentic AI – Mutual Fund Analyzer")
st.write(
    "A tool-orchestrated agent that analyzes mutual fund datasets, "
    "ranks funds using deterministic logic, visualizes insights, "
    "and explains recommendations using an LLM."
)

st.markdown("---")

# ------------------------------------------
# Load LLM Client
# ------------------------------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ------------------------------------------
# Sidebar Inputs
# ------------------------------------------
st.sidebar.header("User Preferences")

top_k = st.sidebar.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)

user_query = st.sidebar.text_input(
    "Ask the agent (low risk / high return / explanation)"
)

# ------------------------------------------
# Upload Dataset
# ------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Mutual Fund CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("👈 Upload a CSV file to activate the agent.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("📄 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# =====================================================
# 🔧 TOOLS (Deterministic)
# =====================================================

def analyze_dataset(df: pd.DataFrame):
    """Tool 1: Dataset Analysis"""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return {
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_cols
    }

def score_funds(df: pd.DataFrame, metadata: dict):
    """Tool 2: Fund Scoring"""
    df = df.copy()
    score = 0

    if "returns" in metadata["numeric_columns"]:
        score += 0.6 * df["returns"]

    if "expense_ratio" in metadata["numeric_columns"]:
        score -= 0.3 * df["expense_ratio"]

    if "risk" in metadata["numeric_columns"]:
        score -= 0.2 * df["risk"]

    if "rating" in metadata["numeric_columns"]:
        score += 0.1 * df["rating"]

    df["score"] = score
    return df.sort_values("score", ascending=False)

def visualize_funds(top_funds: pd.DataFrame):
    """Tool 3: Graph-Based Visualization"""
    st.subheader("📊 Visual Analysis of Top Funds")

    numeric_cols = top_funds.select_dtypes(include="number").columns

    # Score Comparison
    if "score" in numeric_cols:
        st.write("🔹 Score Comparison")
        fig, ax = plt.subplots()
        ax.bar(top_funds.index.astype(str), top_funds["score"])
        ax.set_xlabel("Fund Index")
        ax.set_ylabel("Score")
        st.pyplot(fig)

    # Return vs Risk
    if "returns" in numeric_cols and "risk" in numeric_cols:
        st.write("🔹 Return vs Risk")
        fig, ax = plt.subplots()
        ax.scatter(top_funds["risk"], top_funds["returns"])
        ax.set_xlabel("Risk")
        ax.set_ylabel("Returns")
        st.pyplot(fig)

    # Expense Ratio Comparison
    if "expense_ratio" in numeric_cols:
        st.write("🔹 Expense Ratio Comparison")
        fig, ax = plt.subplots()
        ax.bar(top_funds.index.astype(str), top_funds["expense_ratio"])
        ax.set_xlabel("Fund Index")
        ax.set_ylabel("Expense Ratio")
        st.pyplot(fig)

def explain_with_llm(top_funds: pd.DataFrame, user_query: str):
    """Tool 4: LLM Explanation"""
    snapshot = top_funds.to_string(index=False)

    system_prompt = """
You are a financial explanation agent.
Explain recommendations clearly.
Do not hallucinate missing metrics.
"""

    user_prompt = f"""
User request: {user_query if user_query else "Recommend the best mutual funds"}

Recommended funds:
{snapshot}

Explain:
- Why these funds were selected
- What metrics influenced the ranking
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

# =====================================================
# 🧠 ORCHESTRATOR (Agent Controller)
# =====================================================

def orchestrator(df, user_query, top_k):
    metadata = analyze_dataset(df)
    ranked_df = score_funds(df, metadata)
    top_funds = ranked_df.head(top_k)

    visualize_funds(top_funds)     # Visualization Tool
    explanation = explain_with_llm(top_funds, user_query)

    return top_funds, explanation

# ------------------------------------------
# Run Agent
# ------------------------------------------
if st.button("🚀 Run Agent"):

    with st.spinner("Agent orchestrating tools..."):
        recommendations, explanation = orchestrator(df, user_query, top_k)

    st.subheader("🏆 Top Recommended Funds")
    st.dataframe(recommendations, use_container_width=True)

    st.subheader("🧠 Agent Explanation")
    st.markdown(explanation)
