# TruthX: Alleviating Hallucinations by Editing Large Language Models
TruthX is a research-oriented experimental framework focused on reducing hallucinations in large language models (LLMs) through retrieval-augmented generation (RAG) and inference-time system design choices.


### Background and Motivation
This repository presents an independent, system-focused experimental implementation inspired by the **TruthX** framework originally proposed by the ICTNLP research group. The original TruthX work investigates hallucination mitigation in large language models through model editing techniques aimed at improving factual consistency.

Building on these core ideas, this repository shifts the focus toward **practical system integration and inference-time experimentation**, exploring how TruthX-style hallucination mitigation can be combined with **retrieval-augmented generation (RAG)** in an interactive setting.

### System-Oriented Contributions
Rather than reproducing the original TruthX results verbatim, this implementation emphasizes engineering and system-level experimentation. A key contribution of this repository is the adoption of an **interactive inference workflow**.

In typical experimental pipelines, language models are repeatedly loaded and initialized for each query, introducing significant computational overhead. This project addresses that inefficiency by loading the model once and reusing it across multiple user interactions. This design choice:
- Reduces per-query time complexity by avoiding repeated model initialization
- Lowers inference latency during experimentation
- Enables more interactive, conversational testing and rapid prompt iteration

These improvements make the framework better suited for hands-on analysis of hallucination behavior and system-level evaluation.

### Implementation Scope
The repository provides modular orchestration code, configuration-driven experimentation, and utilities for controlling inference-time behavior. Large model weights and offloaded parameter shards are intentionally excluded from version control in order to follow open-source best practices and ensure reproducibility.

Model artifacts are expected to be downloaded separately at runtime.

### Related Work and Resources
- **Original TruthX (ICTNLP):**  
  https://github.com/ictnlp/TruthX
- **Base Language Models (example):**  
  https://huggingface.co/meta-llama
- **External Frameworks:**  
  FastChat – https://github.com/lm-sys/FastChat

### Acknowledgements
This work is inspired by the original TruthX research conducted by the ICTNLP group and builds upon open-source contributions from the Hugging Face and FastChat communities.

### Project Status
This project is a **work in progress**. The codebase and experimental design are actively evolving as new ideas, system improvements, and evaluation strategies are explored.
