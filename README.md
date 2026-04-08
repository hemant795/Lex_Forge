---
title: LexForge — Comprehensive Legal Intelligence
emoji: ⚖️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: true
tags:
  - openenv
  - legal
  - compliance
  - nlp
  - rl-environment
---

# LexForge — Legal Document Review Environment

An OpenEnv environment for training AI agents on real-world legal document review across **10 legal domains**, covering both **Indian law** (BNS 2023, DPDPA 2023, Labour Codes 2020) and **global frameworks** (GDPR, HIPAA, CCPA, SOX, FATF).

## Motivation

Legal document review is a high-stakes, time-intensive task that costs organisations millions annually. LexForge provides a rigorous, law-verified benchmark for evaluating AI agents on clause identification, risk classification, adversarial detection, and multi-jurisdiction compliance — tasks that frontier LLMs genuinely struggle with.

## Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `flag_clause` | `clause_id`, `risks[]` | Mark a clause as legally risky |
| `clear_clause` | `clause_id` | Mark a clause as benign/safe |
| `classify_risk` | `clause_id`, `risks[]`, `citation` | Classify risk with regulation citation |
| `rewrite_clause` | `clause_id`, `rewritten_text` | Propose a compliant redline |
| `generate_report` | `report{}` | Produce structured audit report |
| `detect_adversarial_clauses` | `clause_ids[]` | Identify obfuscated violations |
| `submit_multi_party_sign_off` | `party_a_satisfied`, `party_b_satisfied`, `balance_justification` | Final sign-off |

## Observation Space

```json
{
  "task_id": "easy",
  "step": 1,
  "max_steps": 5,
  "cascade_multiplier": 1.1,
  "available_actions": ["flag_clause", "clear_clause"],
  "context": {
    "pending_clauses": {"C001": {"id": "C001", "text": "...", "severity_hint": "HIGH"}},
    "reviewed_count": 1,
    "total_clauses": 5,
    "flagged_so_far": ["C001"],
    "risk_taxonomy": {"unreasonable_duration": "..."}
  },
  "reward_breakdown": {"flag_accuracy": 0.5},
  "reward_explanation": "Reward=0.55 | cascade=1.1x"
}
```

## Tasks

| Task | Difficulty | Max Steps | Domains | Expected Baseline |
|------|-----------|-----------|---------|-------------------|
| `easy` | Easy | 5 | Contract Intelligence | ~0.40 |
| `medium` | Medium | 8 | GDPR + AML + Real Estate + Immigration | ~0.50 |
| `hard` | Hard | 12 | Contract + Employment + IP + AML + M&A + Criminal | ~0.55 |
| `expert` | Expert | 15 | All 10 domains, adversarial obfuscation | ~0.70 |

## Reward Function
## Setup

```bash
# Local development
cd ~/lex_forge
pip install openenv-core fastapi uvicorn pydantic openai
uvicorn server.app:app --port 8000

# Docker
docker build -t lex-forge ./server
docker run -p 7860:7860 lex-forge

# Validate
openenv validate
```

## Baseline Scores (gemma4:e4b local)

| Task | Mean Reward |
|------|------------|
| easy | 0.38 |
| medium | 0.50 |
| hard | 0.55 |
| expert | 0.70 |
| **Overall** | **0.53** |

## Inference

```bash
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen3-32B \
HF_TOKEN=hf_yourtoken \
python3 inference.py
```

## Legal Coverage

10 domains: Contract Intelligence, GDPR/DPDPA, Employment & Labour, IP, Corporate M&A, AML, Real Estate, Criminal Evidence, Immigration, Litigation Support. 20 law-verified clause fixtures (C001–C020) including adversarial obfuscated clauses (C019, C020).
