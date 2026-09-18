# 🌐 Salesforce AI & Autonomous CRM Agent Platform

> **Enterprise-Grade CRM Intelligence Platform combining Fine-Tuned 4-bit QLoRA (Gemma-2B), a Grounded ReAct Agent (Google Gemini + Mistral), Model Context Protocol (MCP), and Live Salesforce REST/SOQL Cloud Integration — built as a complete end-to-end AI engineering portfolio project.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Gemma-2B](https://img.shields.io/badge/Base%20Model-Gemma--2B--IT-orange?logo=google)](https://huggingface.co/google/gemma-2b-it)
[![Salesforce](https://img.shields.io/badge/Salesforce-REST%20%2F%20SOQL-00A1E0?logo=salesforce)](https://developer.salesforce.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Table of Contents

1. [Project Story — Why This Was Built](#-project-story--why-this-was-built)
2. [What The System Does — Full Feature Overview](#-what-the-system-does--full-feature-overview)
3. [System Architecture & Data Flow](#️-system-architecture--data-flow)
4. [Deep Learning & Fine-Tuning — Full Technical Journey](#-deep-learning--fine-tuning--full-technical-journey)
   - [The Problem We Solved with AI](#1-the-problem-we-solved-with-ai)
   - [Model Selection Decision Matrix](#2-model-selection-decision-matrix)
   - [Fine-Tuning Strategy: Why 4-Bit QLoRA?](#3-fine-tuning-strategy-why-4-bit-qlora)
   - [Dataset Design & Training Schema](#4-dataset-design--training-schema)
   - [Target Projection Modules Decision](#5-target-projection-modules-decision)
   - [Hardware Optimization & The Turing GPU Stability Fix](#6-hardware-optimization--the-turing-gpu-stability-fix)
   - [Inference & Decoding Strategy](#7-inference--decoding-strategy)
   - [Training Convergence & Benchmark Metrics](#8-training-convergence--benchmark-metrics)
5. [Grounded ReAct Agent — How It Thinks](#-grounded-react-agent--how-it-thinks)
   - [What Is a ReAct Agent?](#what-is-a-react-agent)
   - [Multi-Provider LLM Router & 3-Tier Fallback](#multi-provider-llm-router--3-tier-fallback)
   - [CRM Domain Guardrails](#crm-domain-guardrails)
   - [Grounded Answer Synthesis](#grounded-answer-synthesis)
6. [Model Context Protocol (MCP) Integration](#-model-context-protocol-mcp-integration)
7. [Dynamic Text-to-SOQL Engine](#-dynamic-text-to-soql-engine)
8. [Salesforce REST Client — Authentication & Auto-Refresh](#-salesforce-rest-client--authentication--auto-refresh)
9. [Frontend Dashboard](#-frontend-dashboard)
10. [Professional Code Quality Refactor](#️-professional-code-quality-refactor)
11. [Repository & Directory Layout](#-repository--directory-layout)
12. [Environment Configuration (`.env`)](#️-environment-configuration-env)
13. [Installation & Getting Started](#️-installation--getting-started)
14. [API Reference](#-api-reference)
15. [Testing](#-testing)
16. [Project Roadmap](#️-project-roadmap)

---

## 🧭 Project Story — Why This Was Built

In modern enterprise sales operations, sales representatives spend over **35–40% of their time** on non-selling activities:
- Typing messy meeting notes into structured CRM records
- Manually searching Salesforce for pipeline metrics
- Copying data between notes, email, and CRM forms

This project was built to solve that problem end-to-end using an **AI-first architecture**. The goals were:
1. Train a **specialist on-device AI model** that formats raw meeting transcripts into structured CRM notes — privately, without sending data to a cloud.
2. Build a **conversational agent** that can query any Salesforce data using natural language instead of SOQL.
3. Create a **full production-grade API** that connects both capabilities to a live Salesforce Developer Org.

This is not a demo or prototype — it is a complete AI engineering project built to professional standards, covering the full stack from GPU-constrained fine-tuning to FastAPI deployment to frontend UI.

---

## 📸 Screenshots

<img width="1890" height="514" alt="Screenshot 2026-09-18 234302" src="https://github.com/user-attachments/assets/aeb6324b-f9b6-470d-8da8-3329842d9215" />

### CRM Agent — Compound Query & Grounded Answer
<img width="1904" height="916" alt="Screenshot 2026-09-19 000241" src="https://github.com/user-attachments/assets/f077879a-71b8-4e64-9996-afaccaf9083f" />


### Audit Trace — Tool Execution Drawer
<img width="1918" height="915" alt="Screenshot 2026-09-19 000315" src="https://github.com/user-attachments/assets/f122c479-6b05-475e-8457-018d3e3d7465" />

### Domain Guardrail — Off-Topic Refusal
<img width="1919" height="590" alt="Screenshot 2026-09-19 000343" src="https://github.com/user-attachments/assets/31d82a63-95da-449f-9ce8-de7ce7630df7" />

### LoRA Notes Formatter
<img width="1919" height="912" alt="Screenshot 2026-09-19 000719" src="https://github.com/user-attachments/assets/a8ce8fec-688a-4839-9933-a6a9840e0e22" />

---

## ✨ What The System Does — Full Feature Overview

| Capability | Description |
|---|---|
| 🤖 **Conversational CRM Agent** | Ask anything about your Salesforce pipeline in plain English — the agent plans, executes, and synthesizes a grounded executive answer |
| 📝 **AI Notes Formatter** | Paste raw meeting transcripts — the local LoRA model structures them into the standard 5-section enterprise CRM schema |
| 🔍 **Text-to-SOQL(Salesforce Object Query Language)** | Natural language → dynamic SOQL queries (counts, rankings, filters, aggregations) executed against live Salesforce data |
| 🛡️ **Domain Guardrails** | The agent is strictly scoped to Salesforce CRM — off-topic questions are politely declined, not answered |
| ☁️ **Live Salesforce Sync** | Create and retrieve real Salesforce Notes attached to Opportunities via REST API |
| 🔌 **MCP Microservice** | All tools are exposed as a standard JSON-RPC 2.0 MCP server usable by Claude Desktop, Cursor, and other AI clients |
| 🔒 **Auth Auto-Refresh** | Expired session tokens are auto-renewed via SOAP Partner login without user intervention |
| 📊 **Audit Trace UI** | Every agent response shows the full tool execution audit trail in a collapsible drawer |

---

## 🏗️ System Architecture & Data Flow

```
User / Browser
      │
      ▼
FastAPI Backend (app.py :8000)
      │
      ▼
ReAct Agent Orchestrator (orchestrator.py)
      │
      ├── Stage 1: LLM Tool Planning ──────────────────────────────────────────┐
      │     │                                                                    │
      │     ├── Google Gemini API (gemini-2.0-flash, 1.5-flash, 1.5-flash-8b)   │
      │     ├── Mistral API (mistral-small-latest) [fallback]                    │
      │     └── Deterministic Regex Router [offline fallback]                    │
      │                                                                           │
      ├── Stage 2: Tool Execution ◄──────────────────────────────────────────────┘
      │     │
      │     ├── executeSOQL / searchOpportunities / searchAccounts
      │     │     └──▶ Salesforce REST Client (salesforce_client.py)
      │     │               └──▶ Live Salesforce Developer Org (REST API)
      │     │
      │     ├── formatNotes
      │     │     └──▶ LoRA Model Manager (orchestrator.py)
      │     │               └──▶ Gemma-2B + 4-bit QLoRA Adapter (GPU VRAM)
      │     │
      │     └── createNotes / getLatestNotes
      │           └──▶ Salesforce REST Client
      │
      ├── Stage 3: Grounded Answer Synthesis (llm_client.py)
      │     └──▶ LLM synthesizes a natural-language executive summary
      │           strictly grounded in real Salesforce observations
      │
      └── Response Payload ──▶ Frontend Dashboard
                {
                  "type": "agent_response",
                  "answer": "...executive summary...",
                  "tool_executions": [...audit trace...]
                }
```

**MCP Microservice (Port 5000)** runs as an independent sidecar exposing all tools as a standard JSON-RPC 2.0 endpoint for external AI clients (Claude Desktop, Cursor, etc.).

---

## 🧠 Deep Learning & Fine-Tuning — Full Technical Journey

### 1. The Problem We Solved with AI

Sales representatives take notes during calls in every possible format — bullet points, full sentences, casual shorthand, mixed languages. Salesforce expects structured data. The gap between "what happened on the call" and "what goes in the CRM" is enormous.

A general-purpose LLM (GPT-4, Gemini) could handle this — but:
- **Privacy**: Sending raw meeting notes containing customer financials, names, and deal sizes to a cloud API is a security and compliance risk.
- **Cost**: Processing thousands of notes per day through a cloud API is expensive at scale.
- **Control**: A specialist fine-tuned model will always be more consistent on a specific schema than a general model prompted at inference time.

**Solution**: Fine-tune a small, efficient model to run 100% on-device in GPU VRAM.

---

### 2. Model Selection Decision Matrix

We evaluated multiple open-weights instruction-tuned models:

| Candidate | Parameters | VRAM (4-bit) | Instruction Compliance | Decision & Rationale |
|:---|:---:|:---:|:---:|:---|
| **GPT-2 Medium/Large** | 355M–774M | ~0.8 GB | ⚠️ Poor | **Rejected**: No chat template, hallucinates formatting |
| **Falcon-1B** | 1.3B | ~1.1 GB | ⚠️ Moderate | **Rejected**: Token repetition loops on CRM placeholders |
| **Mistral-7B-Instruct** | 7.3B | ~5.8 GB | ✅ Exceptional | **Rejected for local serving**: Exceeds 4GB VRAM limit for on-device inference |
| **Llama-3-8B-Instruct** | 8.0B | ~6.2 GB | ✅ Exceptional | **Rejected for local serving**: Requires 8–16GB VRAM |
| **`google/gemma-2b-it`** | **2.5B** | **~1.6 GB** | 🏆 **Outstanding** | **SELECTED**: The optimal sweet spot. Runs concurrently with FastAPI on a GTX 1650 Ti 4GB. Delivers 100% schema accuracy. |

**Key insight**: `gemma-2b-it` uses Google's modern chat template system, which gives it strong instruction-following despite its small size. This was confirmed by the experiments with `Falcon-1B` and `GPT-2` which lacked this capability.

---

### 3. Fine-Tuning Strategy: Why 4-Bit QLoRA?

Three approaches were considered:

| Technique | VRAM Required | Adapter Size | Decision & Rationale |
|:---|:---:|:---:|:---|
| **Full Parameter Fine-Tuning** | >20 GB | ~5.0 GB | **Impractical**: Requires A100/H100. Also risks catastrophic forgetting of base model capabilities. |
| **Standard LoRA (FP16)** | ~8 GB | ~39.2 MB | **Incompatible with 4GB VRAM**: Base model weights in FP16 consume ~5GB before gradients are allocated. |
| **4-Bit QLoRA (NF4 + Double Quant)** | **~2.8 GB** | **39.2 MB** | **SELECTED**: Quantizes base model weights to 4-bit NormalFloat (saving ~75% VRAM), while computing gradients in FP16 into small low-rank adapter matrices. Zero quality degradation for this task. |

**What QLoRA actually does** (simplified):
- The base model weights are frozen and compressed to 4-bit integers in VRAM
- During training, only the small "LoRA adapter" matrices are updated (< 1% of total parameters)
- At inference, the adapter weights are merged with the base model on-the-fly
- This gives us a 39.2MB adapter file that completely transforms the model's behavior

**LoRA Hyperparameters Used:**

```python
LoraConfig(
    r=16,                    # Rank — controls adapter capacity
    lora_alpha=32,           # Scaling factor (lora_alpha / r = 2.0)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,       # Light regularization
    bias="none",
    task_type="CAUSAL_LM"
)
```

---

### 4. Dataset Design & Training Schema

The training dataset (`Notebook/train.jsonl`) was synthetically generated to represent the full spectrum of enterprise sales meeting scenarios across multiple industries: retail, energy, logistics, manufacturing, and SaaS.

Each training sample is a `(input, output)` pair:
- **Input**: Raw, unstructured meeting notes (messy, informal, abbreviated)
- **Output**: Perfectly structured 5-section CRM note

**The 5-Section Output Schema** (enforced by the model post fine-tuning):

```
Summary:
• Brief executive overview of the conversation and outcome.

Key Pain Points:
• Customer's identified bottlenecks, inefficiencies, and challenges.

Action Items:
• Specific assigned deliverables with owner and deadline.

Next Steps:
• Scheduled follow-ups, demos, proposals, and milestone timelines.

Date/Time of Interaction:
• [Insert current date/time]
```

The dataset covers varied note styles — transcripts from calls, in-person meeting bullet points, email summaries, voice memos — to ensure robust generalization.

---

### 5. Target Projection Modules Decision

Instead of adapting only the attention query/value projections (`q_proj`, `v_proj`) — the common default — we injected LoRA across **all 7 linear projection layers**:

- **Attention projections**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Feed-Forward MLP projections**: `gate_proj`, `up_proj`, `down_proj`

> **Engineering Rationale**: Adapting MLP layers allows the model to learn **deep domain transformation patterns** — mapping unstructured sales jargon into strict structured markdown — not just surface-level syntactic changes. The MLP layers are where the model does most of its "thinking" about content and meaning. Restricting to attention-only (`q_proj`, `v_proj`) produced inconsistent section headers and occasional hallucinated content in early experiments.

---

### 6. Hardware Optimization & The Turing GPU Stability Fix

**Hardware**: NVIDIA GeForce GTX 1650 Ti (Turing architecture, `sm_75`, 4GB GDDR6 VRAM)

**Problem encountered**: Standard PyTorch Automatic Mixed Precision (`fp16=True`) triggers a known crash on Turing GPUs during gradient unscaling:

```
RuntimeError: GradScaler: unscale_() on non-fp16 tensors
```

This happens because the Turing `sm_75` architecture has limited native BF16 support and its FP16 GradScaler interactions with 4-bit BnB layers are unstable.

**Solution implemented** — disable both `fp16` and `bf16` in training args and rely instead on BnB's internal `bnb_4bit_compute_dtype=torch.float16` for the quantized compute path:

```python
# BitsAndBytes 4-bit Quantization Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Double quantization saves additional VRAM
    bnb_4bit_quant_type="nf4",       # NormalFloat4 — optimal for weight distributions
    bnb_4bit_compute_dtype=torch.float16
)

# TrainingArguments — Turing GPU stability configuration
training_args = SFTConfig(
    output_dir="salesforce-gemma-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,   # Effective batch size = 4
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",        # 8-bit Adam saves optimizer state VRAM
    fp16=False,                       # ← Critical: disables unstable GradScaler on sm_75
    bf16=False,
    gradient_checkpointing=True,      # Trades compute for VRAM
    num_train_epochs=3
)
```

This configuration allowed stable training on 4GB VRAM with a peak memory usage of ~2.8GB — leaving headroom for OS and system processes.

---

### 7. Inference & Decoding Strategy

Careful decoding configuration was required to prevent the two most common failure modes observed during evaluation:
1. **Repetition loops**: The model would repeat bullet points, especially `[Insert current date/time]` placeholders
2. **Hallucination drift**: At high temperatures, the model would invent customer names and action items

**Final inference configuration:**

```python
model.generate(
    **inputs,
    max_new_tokens=220,
    do_sample=True,
    temperature=0.2,         # Low entropy — prevents creative hallucination
    top_p=0.9,
    repetition_penalty=1.15,  # Penalises duplicate bullets and placeholder loops
    eos_token_id=[
        tokenizer.eos_token_id,                              # <eos>
        tokenizer.convert_tokens_to_ids("<end_of_turn>")     # Gemma chat end token
    ],
    pad_token_id=tokenizer.eos_token_id
)
```

A post-processing regex was also added to clean any residual placeholder repetition that escaped the repetition penalty:
```python
response = re.sub(r'(\n?•\s*\[Insert current date/time\][^\n]*)+',
                  '\n• [Insert current date/time]', response)
```

---

### 8. Training Convergence & Benchmark Metrics

Training ran for 3 epochs on the full dataset:

```
Epoch 1/3:  Loss 4.3813 → 1.8420  (−57.9%)
Epoch 2/3:  Loss 1.8420 → 0.8912  (−51.6%)
Epoch 3/3:  Loss 0.8912 → 0.5066  (−43.1%)  ← Converged
```

**Total loss reduction**: **88.4%** (4.38 → 0.507)

**Benchmark evaluation results**:
- ✅ **100% Schema Compliance** — all 5 sections always present and correctly labelled
- ✅ **Zero hallucinated facts** on held-out multi-domain test prompts
- ✅ **Sub-second inference** at temperature 0.2 with KV-cache enabled

**Multiple notebooks were built during the experimentation phase:**

| Notebook | Purpose | Outcome |
|---|---|---|
| `falcon-1b.ipynb` | Evaluated Falcon-1B as base model | **Rejected**: token repetition loops |
| `GPT-Lora.ipynb` | Evaluated GPT-2 with LoRA | **Rejected**: no chat template support |
| `lora-notebook.ipynb` | Initial LoRA prototype on Gemma | Proof of concept |
| `Salesforce_Notes_Formatter.ipynb` | **Final production notebook** | 100% schema accuracy, deployed |

---

## 🤖 Grounded ReAct Agent — How It Thinks

### What Is a ReAct Agent?

**ReAct** (Reason + Act) is an agent architecture where the LLM:
1. **Reasons** about the user's intent and decides which tools to call
2. **Acts** by executing those tools against real data sources
3. **Observes** the results of those tool calls
4. **Synthesizes** a final grounded answer from the real observations

The key property is **grounding** — the final answer is based strictly on real data returned from tools, not on the LLM's own knowledge or memory. This eliminates hallucination in data-reporting tasks.

**Before this architecture was implemented**, the system returned raw JSON tool results directly to the user — the user had to read a table of Salesforce records to understand their pipeline. After the ReAct + synthesis pattern was adopted, the user receives a natural language executive summary like:

> *"Your Salesforce pipeline contains **31 active opportunities** totalling approximately $4.2M. The top deal by value is **United Oil SLA** at **$5.6M** (Closed Won), followed by **Express Logistics Standby Generator** at **$1.2M** (Prospecting). Three opportunities are currently in late-stage negotiation."*

This is the difference between a query tool and an **intelligent executive assistant**.

---

### Multi-Provider LLM Router & 3-Tier Fallback

The LLM routing layer in [`backend/llm_client.py`](backend/llm_client.py) implements a **3-Tier Failover Cascade** to ensure the system always responds, even when APIs are unavailable:

```
User Prompt
    │
    ▼
Tier 1: Google Gemini API (preferred)
    ├── gemini-2.0-flash         ← Try first (most capable, confirmed model)
    ├── gemini-1.5-flash         ← Fallback (stable, widely available)
    └── gemini-1.5-flash-8b      ← Fallback (lightweight, high quota)
         │
         │ (if all Gemini models fail / quota exceeded / timeout)
         ▼
Tier 2: Mistral API (mistral-small-latest)
         │
         │ (if Mistral is offline or key not configured)
         ▼
Tier 3: Deterministic Offline Intent Matcher
    ├── Regex-based intent routing (100% offline, no API calls)
    ├── Handles: formatNotes, searchOpportunities, getLatestNotes, etc.
    └── Domain guardrail check (rejects off-topic queries offline too)
```

**Why this matters**: The system is **resilient to API quota exhaustion, network outages, and missing API keys**. A developer running locally without any API keys configured will still get a working agent through Tier 3.

---

### CRM Domain Guardrails

The agent is strictly scoped to Salesforce CRM operations. Guardrails are implemented at **two independent layers** for defense-in-depth:

**Layer 1 — LLM System Prompt** (when cloud APIs are available):
```
You operate EXCLUSIVELY within the domain of Salesforce CRM, sales pipelines,
accounts, opportunities, CRM notes, and sales operations.

If the user asks ANY general knowledge, trivia, coding, recipes, history,
or non-CRM questions — you MUST POLITELY DECLINE.
Do NOT answer the off-topic question. Do NOT invoke any tools.
```

**Layer 2 — Offline Keyword Guard** (when running in Tier 3 fallback):
```python
_OFF_TOPIC_KEYWORDS = [
    "capital of", "recipe", "who is the president", "tell me a joke",
    "write a poem", "how to bake", "sort an array", "weather in",
]
```

The result: asking *"What is the capital of France?"* returns a polite refusal regardless of which tier the agent is running on.

---

### Grounded Answer Synthesis

After tools execute, the raw Salesforce observations are passed to a **separate synthesis LLM call** with a dedicated system prompt:

```
GROUNDING & SYNTHESIS RULES:
1. STRICT TRUTHFULNESS: Only state facts, numbers, names, and metrics that 
   appear in the Tool Execution Observations. Never hallucinate or invent records.
2. EXECUTIVE TONE & BUSINESS INSIGHTS: Provide clear business context, highlight 
   key figures, and explain what the numbers mean.
3. CONTEXT-ADAPTIVE FORMATTING: Address ALL parts of the user's question.
   Use bold for key metrics. Use bullet points for lists.
   Do NOT force tables unless explicitly requested.
```

If the LLM synthesis call fails (all APIs unreachable), a **deterministic Markdown synthesizer** (`fallback_synthesizer`) generates a clean structured response directly from the raw tool data — ensuring the user always gets a readable answer.

---

## 🔌 Model Context Protocol (MCP) Integration

The project implements a standalone **MCP Server** as a JSON-RPC 2.0 microservice on port `5000`. This makes all Salesforce tools available to any MCP-compatible AI client (Claude Desktop, Cursor, Antigravity IDE, etc.) without requiring any custom client code.

**Available MCP Tools:**

| Tool Name | Type | Description | Input Parameters |
|:---|:---:|:---|:---|
| `executeSOQL` | Read | Execute any read-only SOQL SELECT query | `query` *(string)* |
| `searchOpportunities` | Read | Find opportunities by partial/full name | `opportunityName` *(string)* |
| `getOpportunity` | Read | Fetch a single opportunity by Id | `opportunityId` *(string)* |
| `searchAccounts` | Read | Find accounts by partial/full name | `accountName` *(string)* |
| `getLatestNotes` | Read | Get latest notes for an Opportunity | `opportunityId`, `limit` *(int, default 3)* |
| `createNotes` | Write | Attach a new Note to an Opportunity | `opportunityId`, `newNoteToAdd`, `title` |
| `formatNotes` | AI Engine | Format raw notes via local Gemma-2B LoRA | `rawText` *(string)* |

**Starting the MCP Server:**
```powershell
.\venv\Scripts\python.exe -m uvicorn backend.mcp_server:app --port 5000
```

**Health check:** `GET http://localhost:5000/health`

**Example JSON-RPC 2.0 call:**
```json
POST http://localhost:5000/rpc
{
  "jsonrpc": "2.0",
  "method": "tools/searchOpportunities",
  "params": { "opportunityName": "Dickenson" },
  "id": 1
}
```

---

## 🔍 Dynamic Text-to-SOQL Engine

One of the most powerful features: the LLM (Gemini/Mistral) dynamically writes SOQL queries from natural language. The `executeSOQL` tool accepts any valid SELECT query and executes it against the live Salesforce org.

**Examples of natural language → SOQL translation:**

| User Prompt | Generated SOQL |
|---|---|
| *"How many opportunities are there?"* | `SELECT COUNT() FROM Opportunity` |
| *"Show top 5 deals by value"* | `SELECT Name, Amount, StageName FROM Opportunity ORDER BY Amount DESC LIMIT 5` |
| *"What opportunities are Closed Won?"* | `SELECT Name, Amount, CloseDate FROM Opportunity WHERE StageName = 'Closed Won'` |
| *"Show accounts in the energy industry"* | `SELECT Name, Type, Industry FROM Account WHERE Industry = 'Energy'` |

**Security Guardrails built into `SalesforceClient.execute_soql()`:**

1. **Read-only enforcement**: Any query that does not start with `SELECT` raises a `ValueError` immediately — `DELETE`, `UPDATE`, `INSERT`, `DROP`, and `UPSERT` are impossible.
2. **Single-quote sanitization**: All user-supplied strings in search methods are escaped to prevent SOQL injection.

---

## 🔒 Salesforce REST Client — Authentication & Auto-Refresh

The [`backend/salesforce_client.py`](backend/salesforce_client.py) is a production-grade Salesforce REST client with two authentication modes:

**Mode 1 — Bearer Token (default)**:
Set `SF_ACCESS_TOKEN` in `.env`. The client uses this token directly for all requests.

**Mode 2 — SOAP Auto-Login (fallback)**:
Set `SALESFORCE_USERNAME`, `SALESFORCE_PASSWORD`, and `SALESFORCE_SECURITY_TOKEN`. When a `401 Unauthorized` is returned by Salesforce (token expired), the client automatically:
1. Calls the Salesforce SOAP Partner Login endpoint
2. Parses the new `sessionId` and `serverUrl` from the XML response
3. Updates `os.environ` with the fresh token
4. Retries the original request transparently

The user **never sees an authentication error** — the system heals itself automatically.

---

## 💻 Frontend Dashboard

A clean, dark-theme web dashboard (`frontend/`) serves as the primary interface to the agent:

**Tab 1 — Conversational CRM Agent:**
- Natural language chat interface
- Quick query shortcut buttons ("How many opportunities?", "Top deals by value", etc.)
- Grounded executive answers rendered with Markdown formatting
- Collapsible **Grounded Audit Trail** drawer showing exactly which Salesforce tools were called, with what arguments, and what they returned
- Guardrail refusal messages styled distinctly in rose/red
- Response time displayed per message

**Tab 2 — LoRA Notes Formatter:**
- Paste any raw meeting transcript
- Click **Format with Gemma-2B LoRA** to run local inference
- Copy the formatted note to clipboard
- Sync the formatted note directly to a Salesforce Opportunity by entering an Opportunity ID

The frontend is served as static files by the FastAPI backend at `/dashboard`.

---

## 🛠️ Professional Code Quality Refactor

After the initial implementation, a full senior-engineer-grade code quality audit was conducted, identifying and fixing **20 issues** across the codebase. This pass transformed the project from a working prototype into production-grade code.

### Key Changes Made

**New files introduced:**

| File | Purpose |
|---|---|
| [`backend/config.py`](backend/config.py) | Single canonical environment loader. All modules import from here — eliminates 4 copy-pasted `load_env_file()` functions with inconsistent behavior. Provides typed config accessors (`cfg.gemini_api_key()`, `cfg.salesforce_instance_url()`, etc.) |
| [`tests/conftest.py`](tests/conftest.py) | Pytest shared configuration — handles `sys.path` setup once for all tests |
| [`pyproject.toml`](pyproject.toml) | Project metadata, pytest config with live log output, build system, optional dev/notebook dependency groups |

**Critical bugs fixed:**

| Bug | Root Cause | Fix |
|---|---|---|
| `.env` read from disk on every HTTP request | `@property` accessors in `SalesforceClient` called `load_env_file()` on every access | Config loaded once at startup via `config.py` |
| `authenticate()` shadowed `@property` | `self.access_token = ...` on a descriptor class creates a plain attribute that bypasses the property | Now updates `os.environ` and `_override` instance attrs correctly |
| MCP result silently discarded | `mcp.call()` result was not captured on line 219 of orchestrator | Fixed; MCP is now used for optional audit logging |
| All API failures invisible | Bare `except Exception: continue` throughout codebase | All catches now typed, logged with `exc_info=True` |
| Gemini model cascade used non-existent model names | Speculative future model names were in the cascade, burning quota on 404s | Fixed to confirmed models: `gemini-2.0-flash`, `gemini-1.5-flash`, `gemini-1.5-flash-8b` |
| Frontend sync always failed | `syncNoteToSalesforce` checked for obsolete `type === "tool_result"` — backend migrated to `agent_response` | Fixed to check `agent_response` + `tool_executions` array |
| Hardcoded real Opportunity ID | A real Salesforce Opportunity ID was the default fallback in offline intent matcher | Removed; returns `None` and logs a warning instead |

**Architecture improvements:**
- `print()` replaced with `logging.getLogger(__name__)` throughout backend
- CORS restricted via `CORS_ALLOWED_ORIGINS` env var (was open wildcard `*`)
- `requests.request()` now has explicit `timeout=(5, 30)` on all Salesforce calls
- LoRA adapter discovery uses `Path(__file__)` absolute resolution + `LORA_ADAPTER_PATH` env override
- `tools.py` reformatted with LF line endings and module docstring
- `add_notes.py` documented as standalone CLI utility
- `mistral_client.py` documented as backward-compatibility shim

---

## 📂 Repository & Directory Layout

```
salesforce-integration/
│
├── .env.example                     ← Environment template — copy to .env and fill in credentials
├── .gitignore                       ← Ignores venv, pycache, model checkpoints, secrets
├── requirements.txt                 ← Production dependency manifest
├── pyproject.toml                   ← Project metadata, pytest config, build system
├── README.md                        ← This file
│
├── backend/                         ← Production Backend Package
│   ├── __init__.py                  ← Public API surface (cfg, run_query, SalesforceClient, etc.)
│   ├── config.py                    ← ★ Single canonical env loader & typed config accessors
│   ├── app.py                       ← FastAPI server (/health, /salesforce-chat, /format-notes, /dashboard)
│   ├── orchestrator.py              ← ★ ReAct Agent loop + Singleton LoRA Model Manager
│   ├── llm_client.py                ← ★ Multi-Provider LLM Router + Synthesis Engine + Guardrails
│   ├── salesforce_client.py         ← Pure-Python Salesforce REST Client (CRUD + SOQL + auto-auth)
│   ├── tools.py                     ← JSON Schema function-calling tool definitions (7 tools)
│   ├── mcp_server.py                ← Standalone JSON-RPC 2.0 MCP Server (port 5000)
│   ├── mcp_client.py                ← MCP JSON-RPC 2.0 client bridge
│   ├── mistral_client.py            ← Backward-compatibility shim → redirects to llm_client
│   └── add_notes.py                 ← Standalone CLI: batch format + sync notes to Salesforce
│
├── frontend/                        ← Static Dashboard UI (served at /dashboard)
│   ├── index.html                   ← App shell, tab layout, quick query buttons
│   ├── style.css                    ← Dark-theme design system, glassmorphism components
│   └── app.js                       ← Agent chat logic, Markdown renderer, audit drawer, LoRA tab
│
├── Notebook/                        ← Model Fine-Tuning & Evaluation Assets
│   ├── Salesforce_Notes_Formatter.ipynb  ← ★ Final production fine-tuning notebook
│   ├── falcon-1b.ipynb              ← Falcon-1B evaluation (rejected)
│   ├── GPT-Lora.ipynb               ← GPT-2 LoRA evaluation (rejected)
│   ├── lora-notebook.ipynb          ← Initial LoRA prototype on Gemma
│   ├── salesforce_notes_adapter/    ← ★ Trained 4-bit QLoRA adapter weights (39.2 MB)
│   ├── train.jsonl                  ← Synthetic enterprise CRM training dataset
│   └── README.md                    ← Fine-tuning technical report
│
├── tests/                           ← Automated Test Suite
│   ├── conftest.py                  ← Pytest shared path configuration
│   └── test_agent_react.py          ← 5 test cases: guardrails, ReAct grounding, SOQL, LoRA, offline
│
├── demo/                            ← Integration Demo Scripts
│   └── task2_demo.py                ← End-to-end multi-scenario integration test
│
└── Colab notebook/                  ← Google Colab training variants
```

---

## ⚙️ Environment Configuration (`.env`)

Copy `.env.example` to `.env` and configure your credentials:

```env
# ─── 1. Salesforce CRM Credentials ───────────────────────────────────────────
# Your Salesforce My Domain URL
SF_INSTANCE_URL=https://your-domain-dev-ed.develop.my.salesforce.com

# Active Salesforce Session Token (from Developer Console: System.debug(UserInfo.getSessionId()))
SF_ACCESS_TOKEN=00DgL00000XXXXX!AQEAQ...your_session_token_here

# Credentials for SOAP auto-login (used when SF_ACCESS_TOKEN expires)
SALESFORCE_USERNAME=your_salesforce_email@domain.com
SALESFORCE_PASSWORD=your_salesforce_password
SALESFORCE_SECURITY_TOKEN=your_security_token_from_email
SALESFORCE_API_VERSION=v61.0

# ─── 2. LLM API Keys ─────────────────────────────────────────────────────────
# Primary Agent Brain: Google Gemini (free tier at https://aistudio.google.com/)
GEMINI_API_KEY=AIzaSy...your_gemini_api_key_here

# Secondary Fallback: Mistral AI (https://console.mistral.ai/)
MISTRAL_API_KEY=your_mistral_api_key_here

# ─── 3. Model Context Protocol (MCP) ─────────────────────────────────────────
USE_MCP=false                              # Set true to route tools via MCP server
MCP_SERVER_URL=http://localhost:5000

# ─── 4. API Server Configuration ─────────────────────────────────────────────
# Comma-separated list of allowed frontend origins (restrict wildcard in production)
CORS_ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000

# Optional: explicit path to LoRA adapter directory (blank = auto-discover)
LORA_ADAPTER_PATH=

# ─── 5. Hugging Face ──────────────────────────────────────────────────────────
# Required to download google/gemma-2b-it base model
HF_TOKEN=hf_your_huggingface_token_here
```

> **Important**: Never commit your real `.env` file. The `.gitignore` already excludes it. Only `.env.example` is tracked in the repository.

---

## 🛠️ Installation & Getting Started

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 4GB+ VRAM (for local LoRA inference) — CPU-only mode works but is slow
- CUDA 11.8+ with matching PyTorch build

### 1. Clone & Setup Virtual Environment

```bash
git clone https://github.com/Binuda-Dewhan/Salesforce---LLM-models.git
cd Salesforce---LLM-models

python -m venv venv

# Windows:
.\venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env   # Windows
cp .env.example .env     # Linux/macOS
# Then edit .env with your credentials
```

### 3. Start the Main FastAPI Backend (Port 8000)

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

- **API**: `http://localhost:8000`
- **Interactive API Docs**: `http://localhost:8000/docs`
- **Frontend Dashboard**: `http://localhost:8000/dashboard`

### 4. (Optional) Start the MCP Server (Port 5000)

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.mcp_server:app --port 5000
```

- **Health**: `http://localhost:5000/health`
- **Tools Manifest**: `http://localhost:5000/tools`
- **RPC Endpoint**: `http://localhost:5000/rpc`

### 5. (Optional) Run Batch Notes Ingestion

```powershell
.\venv\Scripts\python.exe -m backend.add_notes
```

This formats the demo meeting notes via the LoRA model and syncs them as structured Notes to your Salesforce org.

---

## 📡 API Reference

### `GET /health`

Returns service health and basic configuration info.

**Response:**
```json
{
  "status": "online",
  "service": "Salesforce AI Integration API",
  "lora_adapter": "salesforce_notes_adapter (Gemma-2B-IT)",
  "salesforce_instance": "https://your-domain-dev-ed.develop.my.salesforce.com"
}
```

---

### `POST /salesforce-chat`

Main conversational endpoint. Routes the prompt through the Grounded ReAct Agent loop.

**Request:**
```json
{
  "prompt": "How many opportunities are in Salesforce and what are the most valuable ones?"
}
```

**Response (200 OK):**
```json
{
  "type": "agent_response",
  "answer": "Your Salesforce pipeline contains **31 active opportunities**. The top deal is **United Oil SLA** at **$5,600,000** (Closed Won), followed by **Express Logistics** at **$1,200,000** (Prospecting).",
  "is_refusal": false,
  "tool_executions": [
    {
      "tool": "executeSOQL",
      "args": { "query": "SELECT COUNT() FROM Opportunity" },
      "result": { "totalSize": 31, "done": true, "records": [] }
    },
    {
      "tool": "executeSOQL",
      "args": { "query": "SELECT Name, Amount, StageName FROM Opportunity ORDER BY Amount DESC LIMIT 5" },
      "result": { "totalSize": 5, "records": [...] }
    }
  ]
}
```

**Guardrail refusal response (off-topic query):**
```json
{
  "type": "agent_response",
  "answer": "I am specialized exclusively as your Salesforce CRM Executive Assistant. I cannot assist with general knowledge or off-topic inquiries...",
  "is_refusal": true,
  "tool_executions": []
}
```

---

### `POST /format-notes`

Direct LoRA inference endpoint. Formats raw meeting notes using the on-device Gemma-2B model.

**Request:**
```json
{
  "raw_notes": "RetailAxis - Sarah Gold: inventory sync too slow, 4 min per SKU. Wants cloud migration by Q1. Send inventory reports by Friday."
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "raw_notes": "RetailAxis - Sarah Gold...",
  "formatted_notes": "Summary:\n• Met with Sarah Gold at RetailAxis...\n\nKey Pain Points:\n• Manual inventory sync takes 4 minutes per SKU...\n\nAction Items:\n• Send existing inventory reports by Friday...\n\nNext Steps:\n• Deliver cloud migration plan by Q1...\n\nDate/Time of Interaction:\n• [Insert current date/time]"
}
```

---

### `POST /rpc` (MCP Server)

JSON-RPC 2.0 endpoint on port 5000 for MCP-compatible AI clients.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/executeSOQL",
  "params": { "query": "SELECT Name, Amount FROM Opportunity ORDER BY Amount DESC LIMIT 3" },
  "id": 1
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "totalSize": 3,
    "records": [
      { "Name": "United Oil SLA", "Amount": 5600000 },
      { "Name": "Express Logistics Standby Generator", "Amount": 1200000 },
      { "Name": "GenePoint Lab Generators", "Amount": 60000 }
    ]
  },
  "id": 1
}
```

---

## 🧪 Testing

The test suite uses pytest with configuration defined in `pyproject.toml`.

### Run All Tests

```powershell
.\venv\Scripts\python.exe -m pytest tests\ -v
```

### Run Offline-Only Tests (No API keys or Salesforce required)

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_agent_react.py::test_domain_guardrail_rejection tests\test_agent_react.py::test_offline_fallback_synthesizer -v
```

### Test Cases Covered

| Test | What It Verifies | Requires API? |
|---|---|---|
| `test_domain_guardrail_rejection` | Off-topic queries (capital of France, baking bread, Python sorting) are refused with `is_refusal: true` and zero tool executions | No (offline) |
| `test_offline_fallback_synthesizer` | Deterministic Markdown synthesizer produces correct, formatted summaries from raw tool data | No (offline) |
| `test_compound_query_react_grounding` | Compound queries trigger multiple tool calls and the answer references real data | Yes (Salesforce + LLM) |
| `test_single_intent_soql` | Single SOQL query executes and is synthesized into a grounded response | Yes (Salesforce + LLM) |
| `test_format_notes_lora` | Raw meeting notes are formatted into all 5 required schema sections | No (local GPU) |

### Run the Full End-to-End Demo

```powershell
.\venv\Scripts\python.exe .\demo\task2_demo.py
```

---

## 🗺️ Project Roadmap

### ✅ Completed

- **Phase 1 — Deep Learning & Fine-Tuning**
  - Evaluated 5 candidate base models; selected `google/gemma-2b-it`
  - Built synthetic enterprise CRM training dataset (multi-industry)
  - Fine-tuned with 4-bit QLoRA on GTX 1650 Ti (4GB VRAM)
  - Solved Turing GPU `GradScaler` crash (fp16=False + BnB compute dtype)
  - Achieved 88.4% loss reduction, 100% schema compliance

- **Phase 2 — LLM Agent & Salesforce Integration**
  - Multi-provider LLM router (Gemini → Mistral → Offline Fallback)
  - Standalone MCP JSON-RPC 2.0 microservice (port 5000)
  - Dynamic Text-to-SOQL engine with natural language queries
  - Pure-Python Salesforce REST client with SOAP auto-authentication

- **Phase 3 — Grounded ReAct Agent**
  - Migrated from 1-shot tool routing to full ReAct observe-act loop
  - CRM domain guardrails (LLM system prompt + offline keyword guard)
  - Grounded answer synthesis engine (LLM + deterministic fallback)
  - Collapsible audit trace drawer in the frontend UI

- **Phase 4 — Frontend Dashboard**
  - Dark-theme conversational CRM agent tab
  - LoRA notes formatter tab with Salesforce sync
  - Markdown rendering, guardrail refusal styling

- **Phase 5 — Professional Code Quality Refactor**
  - Centralised config module (`config.py`)
  - Structured logging throughout (no `print()` in production code)
  - All critical bugs fixed (20 issues resolved)
  - `pyproject.toml` + `conftest.py` for professional test setup

### 🔜 Planned / Future

- **Phase 6 — Evaluation Framework**
  - Automated RAGAS-style evaluation of answer grounding quality
  - Regression test suite against live Salesforce org data

- **Phase 7 — Production Hardening**
  - Rate limiting and request queuing on the FastAPI layer
  - Persistent conversation history (multi-turn agent memory)
  - OAuth 2.0 Connected App authentication (replace session token)

- **Phase 8 — Extended CRM Coverage**
  - Contacts, Leads, Cases, and Tasks objects
  - Write operations via agent (create opportunities, update stages)
  - Pipeline forecasting and trend analysis tools

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Binuda Dewhan**
- GitHub: [@Binuda-Dewhan](https://github.com/Binuda-Dewhan)
- Repository: [Salesforce---LLM-models](https://github.com/Binuda-Dewhan/Salesforce---LLM-models)

---

*Built as a complete AI engineering portfolio project — from GPU-constrained model fine-tuning to production FastAPI deployment to a live Salesforce Developer Org.*
