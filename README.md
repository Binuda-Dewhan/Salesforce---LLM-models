# 🌐 Salesforce AI & Autonomous CRM Agent Platform
> **Enterprise-Grade CRM Intelligence Platform combining Fine-Tuned 4-bit QLoRA (Gemma-2B), Cloud Agent Tool Calling (Google Gemini & Mistral), Model Context Protocol (MCP), and Salesforce REST/SOQL Cloud Sync.**

---

## 📌 Table of Contents
1. [Executive Overview](#-executive-overview)
2. [System Architecture & Data Flow](#-system-architecture--data-flow)
3. [Deep Learning & Fine-Tuning Architectural Decisions](#-deep-learning--fine-tuning-architectural-decisions)
   - [Model Selection Decision Matrix](#1-model-selection-decision-matrix)
   - [Fine-Tuning Strategy: Why 4-Bit QLoRA?](#2-fine-tuning-strategy-why-4-bit-qlora)
   - [Target Projection Modules Decision](#3-target-projection-modules-decision)
   - [Hardware Optimization & The Turing GPU Stability Fix](#4-hardware-optimization--the-turing-gpu-stability-fix)
   - [Inference & Decoding Strategy](#5-inference--decoding-strategy)
   - [Training Convergence & Benchmark Metrics](#6-training-convergence--benchmark-metrics)
4. [Multi-Provider LLM Agent Router & 3-Tier Fallback](#-multi-provider-llm-agent-router--3-tier-fallback)
5. [Model Context Protocol (MCP) Integration](#-model-context-protocol-mcp-integration)
6. [Dynamic Text-to-SOQL Engine & Security Guardrails](#-dynamic-text-to-soql-engine--security-guardrails)
7. [Repository & Directory Layout](#-repository--directory-layout)
8. [Environment Configuration (`.env`)](#-environment-configuration-env)
9. [Installation & Getting Started](#-installation--getting-started)
10. [Comprehensive API Reference](#-comprehensive-api-reference)
11. [Verification & Automated Test Suite](#-verification--automated-test-suite)
12. [Project Roadmap](#-project-roadmap)

---

## 🚀 Executive Overview

In modern enterprise sales operations, sales representatives spend over **35–40% of their time** performing manual data entry, deciphering unstructured meeting notes, looking up CRM records, and typing structured summaries into Salesforce.

This repository delivers an **Autonomous Enterprise AI Agent Platform** that automates the entire sales workflow end-to-end:
- **Intelligent Conversational Agent Brain**: Uses **Google Gemini 2.5/3.5/3.6 Flash & Flash-Lite** with **Mistral AI** fallback for natural language intent routing and dynamic SOQL planning.
- **Specialist On-Device AI Formatter**: A custom fine-tuned **4-bit QLoRA adapter (`salesforce_notes_adapter`)** trained on `google/gemma-2b-it` that runs locally in GPU VRAM to structure messy meeting transcripts into a strict 5-section enterprise CRM schema.
- **Model Context Protocol (MCP)**: A standalone **JSON-RPC 2.0 microservice** on port `5000` that standardizes tool exposure for Claude Desktop, Cursor, Antigravity, and external AI clients.
- **Dynamic Text-to-SOQL Engine**: Converts natural language requests into read-only SOQL queries (counts, sums, aggregations, and rankings) with automated security guardrails.
- **Live Salesforce Cloud Sync**: Pure-Python REST client with auto-refreshing OAuth2 / SOAP login sessions.

---

## 🏗️ System Architecture & Data Flow

```mermaid
graph TD
    User([User / Browser / Demo Client]) --> API[FastAPI Backend: backend/app.py :8000]
    API --> Orch[Agent Orchestrator: backend/orchestrator.py]
    
    subgraph Agent Intelligence Layer [Multi-Provider LLM Brain]
        Orch --> Gemini["Google Gemini (3.6-Flash / 3.5-Flash-Lite / 3.1-Flash-Lite)"]
        Gemini -.->|429 Quota Fallback| Mistral["Mistral AI API (mistral-small-latest)"]
        Mistral -.->|Offline Fallback| RuleMatcher["Deterministic Intent Regex Matcher"]
    end
    
    subgraph Protocol & Tool Layer [Model Context Protocol (MCP)]
        Orch -->|USE_MCP=true| MCPClient[MCP JSON-RPC Client: backend/mcp_client.py]
        MCPClient -->|POST http://localhost:5000/rpc| MCPServer[MCP Server Microservice: backend/mcp_server.py]
    end
    
    subgraph Specialized Execution Engines [Local & Cloud Execution]
        Orch -->|Tool: formatNotes| LoRA["Local Gemma-2B LoRA Adapter (4-bit QLoRA in GPU VRAM)"]
        MCPServer -->|Tool: executeSOQL / CRM CRUD| SFClient[Salesforce REST Client: backend/salesforce_client.py]
    end
    
    LoRA --> FormattedNotes([5-Section Standard CRM Notes])
    SFClient --> Cloud[(Live Salesforce Developer Org)]
```

---

## 🧠 Deep Learning & Fine-Tuning Architectural Decisions

The custom deep learning component is a core pillar of this architecture. Below is the technical rationale behind every engineering decision made during dataset design, model selection, quantization, and training.

### 1. Model Selection Decision Matrix

We evaluated multiple open-weights instruction-tuned models for the specialized note-formatting task:

| Candidate Base Model | Parameter Count | VRAM Footprint (4-bit) | Instruction Compliance | Decision & Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **GPT-2 Medium / Large** | 355M / 774M | ~0.8 GB | ⚠️ Poor (Hallucinates formatting) | **Rejected**: Lacks modern instruction-tuning and conversational chat templates. |
| **Falcon-1B / 1.3B** | 1.3B | ~1.1 GB | ⚠️ Moderate (Repeats tokens) | **Rejected**: Context length limitations and tendency to loop on CRM placeholders. |
| **Mistral-7B-Instruct-v0.3** | 7.3B | ~5.8 GB | ✅ Exceptional | **Rejected for Local Serving**: Exceeds consumer 4GB VRAM capacity for on-device inference without heavy CPU offloading latency. |
| **Llama-3-8B-Instruct** | 8.0B | ~6.2 GB | ✅ Exceptional | **Rejected for Local Serving**: Requires 8GB–16GB VRAM for smooth inference. |
| **`google/gemma-2b-it`** | **2.5B** | **~1.6 GB** | 🏆 **Outstanding** | **SELECTED**: The optimal architectural sweet spot. Compact enough to run concurrently with FastAPI on consumer hardware (GTX 1650 Ti 4GB), while delivering 100% schema accuracy. |

---

### 2. Fine-Tuning Strategy: Why 4-Bit QLoRA?

| Technique | VRAM Required | Training Time | Adapter Size | Decision & Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Full Parameter Fine-Tuning** | > 20 GB | Long | ~5.0 GB | **Impractical**: Modifies all 2.5B weights; prone to catastrophic forgetting. |
| **Standard LoRA (16-bit FP16)** | ~8 GB | Medium | ~39.2 MB | **Incompatible with 4GB VRAM**: Base model weights in FP16 consume ~5GB before gradients. |
| **4-Bit QLoRA (NF4 + Double Quant)** | **~2.8 GB** | **Fast (~12 min)** | **39.2 MB** | **SELECTED**: Quantizes base model to 4-bit NormalFloat while computing gradients in FP16 into low-rank adapter matrices. Zero loss degradation. |

---

### 3. Target Projection Modules Decision

Instead of adapting only the attention query/value projections (`q_proj`, `v_proj`), we injected LoRA across **all 7 linear layers**:
- **Attention Projections**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Feed-Forward MLP Projections**: `gate_proj`, `up_proj`, `down_proj`

> **Engineering Rationale**: Adapting the MLP layers allows the model to learn deep domain transformation patterns (mapping unstructured sales jargon into strict 5-section enterprise markdown), rather than just surface-level syntactic adaptations.

---

### 4. Hardware Optimization & The Turing GPU Stability Fix

When training on NVIDIA GeForce GTX 1650 Ti (Turing architecture, `compute_capability=7.5`), standard PyTorch automatic mixed precision (`fp16=True`) triggers a known crash during gradient unscaling (`GradScaler: unscale_() on non-fp16 tensors`).

**Engineering Solution Implemented**:
```python
# BitsAndBytes 4-bit Quantization Configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# TrainingArguments Turing Stability Configuration
training_args = SFTConfig(
    output_dir="salesforce-gemma-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    fp16=False,   # Critical: Disables problematic GradScaler on Turing sm_75
    bf16=False,
    gradient_checkpointing=True,
    num_train_epochs=3
)
```

---

### 5. Inference & Decoding Strategy

To eliminate repetition loops while preserving factual accuracy:
- **Sampling Temperature**: `0.2` (Low entropy to prevent creative hallucination).
- **Repetition Penalty**: `1.15` (Penalizes duplicate bullets and placeholder loops).
- **EOS Stop Tokens**: Configured with both `<eos>` (`1`) and `<end_of_turn>` (`107`).
- **KV-Caching**: `model.config.use_cache = True` for sub-second streaming inference.

---

### 6. Training Convergence & Benchmark Metrics

```
Epoch 1/3:  [Loss: 4.3813 -> 1.8420]
Epoch 2/3:  [Loss: 1.8420 -> 0.8912]
Epoch 3/3:  [Loss: 0.8912 -> 0.5066] (Converged)
```

- **Loss Reduction**: **88.4%** descent ($4.38 \to 0.506$).
- **Benchmark Evaluation**: **100% Schema Compliance** across multi-domain enterprise meeting transcripts.

#### Standard 5-Section Output Schema:
```text
Summary:
• Brief executive overview of the conversation.

Key Pain Points:
• Identified customer bottlenecks, manual errors, and scalability limits.

Action Items:
• Assigned deliverables and immediate team responsibilities.

Next Steps:
• Scheduled demos, milestone timelines, and follow-ups.

Date/Time of Interaction:
• Interaction date and timestamp metadata.
```

---

## ⚡ Multi-Provider LLM Agent Router & 3-Tier Fallback

The conversational control plane in [`backend/llm_client.py`](file:///d:/Personal%20Project_certificate%20work%20Space(Projects%20)/salesforce%20project/salesforce-integration/backend/llm_client.py) implements a **3-Tier Failover Cascade**:

```mermaid
graph TD
    Query([User Prompt]) --> L1[Tier 1: Google Gemini Multi-Model Cascade]
    
    subgraph Gemini Cascade [Tier 1: Google AI Studio]
        L1 --> M1[1. gemini-3.6-flash]
        M1 -.->|429 Quota Exceeded| M2[2. gemini-3.5-flash-lite]
        M2 -.->|429 Quota Exceeded| M3[3. gemini-3.1-flash-lite]
        M3 -.->|429 Quota Exceeded| M4[4. gemini-flash-lite-latest]
        M4 -.->|429 Quota Exceeded| M5[5. gemini-2.5-flash]
    end
    
    Gemini Cascade -.->|If All Gemini Models Exhausted| L2[Tier 2: Mistral API Fallback<br/>mistral-small-latest]
    L2 -.->|If Mistral API Offline| L3[Tier 3: Local Deterministic Matcher<br/>100% Offline Regex Routing]
    
    M1 --> Success([Execute Tool])
    M2 --> Success
    M3 --> Success
    M4 --> Success
    M5 --> Success
    L2 --> Success
    L3 --> Success
```

---

## 🔌 Model Context Protocol (MCP) Integration

The project exposes all Salesforce CRM operations and LoRA formatting capabilities as standard **MCP Tools** over **JSON-RPC 2.0**:

- **MCP Server Microservice**: [`backend/mcp_server.py`](file:///d:/Personal%20Project_certificate%20work%20Space(Projects%20)/salesforce%20project/salesforce-integration/backend/mcp_server.py) on port `5000`.
- **MCP Client Bridge**: [`backend/mcp_client.py`](file:///d:/Personal%20Project_certificate%20work%20Space(Projects%20)/salesforce%20project/salesforce-integration/backend/mcp_client.py).

### Implemented MCP Tools Manifest:

| Tool Name | Type | Description | Input Parameters |
| :--- | :---: | :--- | :--- |
| **`executeSOQL`** | Read | Executes dynamic read-only SOQL queries (counts, sums, aggregations, filters). | `query` *(string)* |
| **`searchOpportunities`** | Read | Finds opportunities by partial/full name via SOQL. | `opportunityName` *(string)* |
| **`getOpportunity`** | Read | Fetches complete opportunity record details by 18-character Id. | `opportunityId` *(string)* |
| **`searchAccounts`** | Read | Finds account/company records in Salesforce by name. | `accountName` *(string)* |
| **`getLatestNotes`** | Read | Retrieves latest notes attached to an Opportunity. | `opportunityId` *(string)*, `limit` *(int)* |
| **`createNotes`** | Write | Creates and attaches a Note object in Salesforce Cloud. | `opportunityId`, `newNoteToAdd`, `title` |
| **`formatNotes`** | AI Engine | Formats raw messy notes via local Gemma-2B LoRA model. | `rawText` *(string)* |

---

## 🛡️ Dynamic Text-to-SOQL Engine & Security Guardrails

The `executeSOQL` engine allows Google Gemini to write ad-hoc SOQL queries dynamically:
- *"How many opportunities are in Salesforce?"* $\to$ `SELECT COUNT() FROM Opportunity`
- *"Show top 3 deals by amount"* $\to$ `SELECT Name, Amount, StageName FROM Opportunity ORDER BY Amount DESC LIMIT 3`

### Security Guardrails:
1. **Read-Only Enforcement**: Rejects any statement containing mutating commands (`DELETE`, `UPDATE`, `INSERT`, `DROP`, `UPSERT`).
2. **SOQL Injection Sanitization**: Escapes single quotes and special characters.

---

## 📂 Repository & Directory Layout

```
salesforce-integration/
├── .env.example                 <-- Sanitized environment template
├── .gitignore                   <-- Rules ignoring virtualenvs, caches, secrets
├── requirements.txt             <-- Full project dependency manifest
├── README.md                    <-- Root portfolio documentation & architecture report
│
├── backend/                     <-- Core Production Backend
│   ├── __init__.py              <-- Public API surface exports
│   ├── app.py                   <-- FastAPI server (/health, /salesforce-chat, /format-notes)
│   ├── orchestrator.py          <-- Agent Control Plane & Singleton LoRA Model Manager
│   ├── llm_client.py            <-- Multi-Provider LLM Router (Gemini + Mistral + Fallback)
│   ├── mistral_client.py        <-- Backward-compatibility shim
│   ├── salesforce_client.py     <-- Pure-Python Salesforce REST Client (CRUD + SOQL)
│   ├── mcp_server.py            <-- Standalone JSON-RPC 2.0 MCP Server Microservice
│   ├── mcp_client.py            <-- MCP JSON-RPC 2.0 Client Bridge
│   ├── tools.py                 <-- JSON Schema function calling tool definitions
│   └── add_notes.py             <-- Batch CRM note formatting and ingestion script
│
├── Notebook/                    <-- Model Fine-Tuning & Evaluation Assets
│   ├── Salesforce_Notes_Formatter.ipynb
│   ├── salesforce_notes_adapter/ <-- 4-Bit LoRA Adapter Weights (39.2 MB) & Tokenizer
│   ├── train.jsonl              <-- Standardized 5-section CRM training dataset
│   └── README.md                <-- Fine-tuning technical report (loss 4.38 -> 0.5066)
│
└── demo/                        <-- Verification Suite
    ├── __init__.py
    └── task2_demo.py            <-- End-to-end multi-scenario integration test
```

---

## ⚙️ Environment Configuration (`.env`)

Copy `.env.example` to `.env` and configure your credentials:

```env
# 1. Salesforce CRM Credentials
SF_INSTANCE_URL=https://your-domain-dev-ed.develop.my.salesforce.com
SF_ACCESS_TOKEN=00DgL00000XXXXX!AQEAQ...your_session_token_here
SALESFORCE_USERNAME=your_salesforce_email@domain.com
SALESFORCE_PASSWORD=your_salesforce_password
SALESFORCE_SECURITY_TOKEN=your_salesforce_security_token_from_email
SALESFORCE_INSTANCE_URL=https://your-domain-dev-ed.develop.my.salesforce.com
SALESFORCE_API_VERSION=v61.0

# 2. Large Language Model (LLM) API Keys
GEMINI_API_KEY=AIzaSy...your_gemini_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here

# 3. Model Context Protocol (MCP) Configuration
USE_MCP=true
MCP_SERVER_URL=http://localhost:5000

# 4. Hugging Face Credentials
HF_TOKEN=hf_your_huggingface_token_here
```

---

## 🛠️ Installation & Getting Started

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/Binuda-Dewhan/Salesforce---LLM-models.git
cd Salesforce---LLM-models

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Start the Standalone MCP Server (Port 5000)
```powershell
.\venv\Scripts\python.exe -m uvicorn backend.mcp_server:app --port 5000
```

### 3. Start the Main FastAPI Backend (Port 8000)
```powershell
.\venv\Scripts\python.exe -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 Comprehensive API Reference

### 1. `POST /salesforce-chat` (Main Conversational Endpoint)
- **URL**: `http://localhost:8000/salesforce-chat`
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`

**Sample Request**:
```json
{
  "prompt": "How many opportunities are in Salesforce?"
}
```

**Sample Response (200 OK)**:
```json
{
  "type": "tool_result",
  "tool": "executeSOQL",
  "args": {
    "query": "SELECT COUNT() FROM Opportunity"
  },
  "result": {
    "totalSize": 31,
    "done": true,
    "records": []
  }
}
```

---

### 2. `POST /format-notes` (Direct LoRA Note Formatter)
- **URL**: `http://localhost:8000/format-notes`
- **Method**: `POST`

**Sample Request**:
```json
{
  "raw_notes": "RetailAxis Sarah Gold: inventory sync too manual. Wants cloud migration by Q1."
}
```

**Sample Response (200 OK)**:
```json
{
  "status": "success",
  "raw_notes": "RetailAxis Sarah Gold...",
  "formatted_notes": "Summary:\n• Met with Sarah Gold at RetailAxis\n\nKey Pain Points:\n• Manual inventory sync...\n\nAction Items:\n• Schedule migration review...\n\nNext Steps:\n• Follow up by Q1...\n\nDate/Time of Interaction:\n• [Insert current date/time]"
}
```

---

### 3. `POST /rpc` (MCP Server JSON-RPC 2.0 Endpoint)
- **URL**: `http://localhost:5000/rpc`
- **Method**: `POST`

**Sample Request**:
```json
{
  "jsonrpc": "2.0",
  "method": "tools/searchOpportunities",
  "params": {
    "opportunityName": "Dickenson"
  },
  "id": 1
}
```

**Sample Response (200 OK)**:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "totalSize": 1,
    "records": [
      {
        "Id": "006gL00000A4GZtQAN",
        "Name": "Dickenson Mobile Generators",
        "StageName": "Qualification",
        "Amount": 15000.0,
        "CloseDate": "2025-06-14"
      }
    ]
  },
  "id": 1
}
```

---

## 🧪 Verification & Automated Test Suite

Run the full end-to-end integration demo:
```powershell
.\venv\Scripts\python.exe .\demo\task2_demo.py
```

Run the dynamic SOQL and MCP test suite:
```powershell
.\venv\Scripts\python.exe scratch/test_soql_mcp.py
```

---

## 🗺️ Project Roadmap
- [x] **Phase 1**: Fine-tuned 4-bit QLoRA model (`google/gemma-2b-it`) with 100% schema compliance.
- [x] **Phase 2**: Multi-provider LLM router (Gemini + Mistral), standalone MCP JSON-RPC 2.0 server, and dynamic Text-to-SOQL engine.
- [ ] **Phase 3**: Interactive Web Chat & CRM Analytics Dashboard UI.
