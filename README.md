
FountryBot
Startup Idea Viability Analyzer (LLM + RAG)

FountryBot is an evidence-driven startup evaluation system that uses Large Language Models (LLMs) combined with Retrieval-Augmented Generation (RAG) to assess startup idea viability based on real historical startup outcome data.

Instead of generic startup advice, FountryBot retrieves comparable historical startup cases and generates grounded, structured viability analysis.

🚀 Problem Statement

Early-stage founders often validate ideas using intuition, anecdotal advice, or fragmented online content. However, structured startup datasets already reveal patterns in:

Industry-level failure rates

Funding-stage survival probabilities

Geographic funding disparities

Time-to-funding signals

Milestone progression trends

FountryBot converts historical startup data into a case-based reasoning engine for structured feasibility evaluation.

🧠 Core Methodology

FountryBot implements a Retrieval-Augmented Generation pipeline:

User submits startup idea attributes (industry, stage, funding status, location).

The idea summary is embedded.

The system retrieves similar historical startups from a vector database.

The LLM analyzes retrieved cases.

A structured viability assessment is generated.

This ensures outputs are grounded in comparable real-world startup trajectories rather than speculative advice.

📊 Dataset Features

The system is built on publicly available startup datasets containing:

Industry category (category_code)

Funding rounds and total funding

Investor participation (VC, Angel, Series A–D)

Milestone counts

Geographic indicators

Time-to-funding metrics

Final outcome status (operating, closed, acquired)

Each startup record is transformed into a structured RAG chunk representing a historical case.

🏗 System Architecture

User Input
→ Embedding Model
→ Vector Database (FAISS / Chroma)
→ Retrieval of Comparable Startups
→ LLM Reasoning Layer
→ Structured Viability Report

📈 Output Structure

Each evaluation includes:

Viability Score (0–100)

Comparable Historical Startups

Industry Failure Context

Funding Pattern Insights

Risk Indicators

Actionable Recommendations

🧩 RAG Design Strategy

Each startup row is converted into a structured case including:

Industry

Location

Funding profile

Lifecycle timing

Milestones

Final outcome

This enables case-based reasoning rather than simple statistical prediction.

🛠 Tech Stack

Python

Pandas

FAISS / Chroma

OpenAI API / Ollama

Streamlit (UI Layer)

🎯 What This Project Demonstrates

Applied LLM + RAG architecture

Structured data-to-text transformation

Case-based decision intelligence

Explainable AI outputs

Evidence-grounded reasoning

🔮 Future Enhancements

Survival probability modeling

Sector-level risk heatmaps

Confidence calibration scoring

Temporal funding trend analysis

Founder profile integration# FoundryBOT
