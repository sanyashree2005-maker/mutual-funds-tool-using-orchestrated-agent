# 🤖 Agentic AI – Mutual Fund Analyzer

This project implements a **tool-orchestrated agentic AI system** that analyzes mutual fund datasets, ranks funds using deterministic financial logic, and generates **clear, human-readable explanations** using a Large Language Model (LLM).

The system focuses on **explainability, transparency, and flexibility**, making it suitable for academic and demonstration purposes.

---

## 📌 Problem Statement

Investors often find it difficult to analyze mutual fund data due to:
- Large number of funds
- Multiple financial metrics
- Lack of explainability in automated systems

The goal of this project is to design an intelligent system that can:
- Analyze mutual fund datasets
- Rank funds objectively
- Explain recommendations in simple language

---

## 🎯 Objectives

- Build a **tool-orchestrated agentic AI system**
- Rank mutual funds using deterministic logic
- Use LLMs only for explanation (not decision-making)
- Reduce hallucination and improve trust
- Provide an interactive web interface

---

## 🧠 System Architecture

The system follows a **single-agent tool orchestration architecture**, where one controller manages multiple tools.

This design corresponds to a **Level-1 Agentic AI system**.

---

## 🔧 Tools Used in the System

### 1️⃣ Dataset Analysis Tool
- Inspects dataset structure
- Identifies numerical columns
- Enables flexibility across different CSV formats

### 2️⃣ Fund Scoring Tool
- Applies deterministic financial logic
- Uses metrics such as returns, risk, expense ratio, and ratings (if available)
- Produces a ranked list of mutual funds

### 3️⃣ Explanation Tool (LLM)
- Uses a Large Language Model to explain results
- Converts numerical insights into human-readable text
- Does not influence ranking decisions

---

## 🚀 Technologies Used

- **Python**
- **Streamlit** – Web application framework
- **Pandas** – Data handling and processing
- **Groq API** – LLM inference
- **LLaMA 3.1** – Natural language explanation

---


