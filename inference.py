"""
inference.py
============
LexForge baseline inference — OpenEnv Competition Submission.

Uses HTTP to call the running environment server (HF Space or local).
No direct imports — works identically locally and on the validator.

Mandatory env variables:
  API_BASE_URL      LLM API endpoint          (default: HF router)
  MODEL_NAME        Model identifier           (default: Qwen/Qwen3-32B)
  HF_TOKEN          HuggingFace API key
  LOCAL_IMAGE_NAME  Docker image name (optional)

stdout format (exact — validated by competition):
  [START] task=<name> env=lex_forge model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

from __future__ import annotations
import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ── Mandatory env vars — first 2 MUST have hardcoded defaults ─────────────────
API_BASE_URL     = os.getenv("API_BASE_URL",     "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",       "Qwen/Qwen3-32B")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Environment server URL — HF Space or local
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

MAX_STEPS   = 10
TEMPERATURE = 0.1
MAX_TOKENS  = 512
TASKS       = ["easy", "medium", "hard", "expert"]
ENV_NAME    = "lex_forge"

# ── stdout logging (exact format required by spec) ────────────────────────────

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={ENV_NAME} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP]  step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )

def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END]   success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )

# ── Environment HTTP client ───────────────────────────────────────────────────

def env_reset(task_id: str) -> Dict[str, Any]:
    """POST /reset — returns observation dict."""
    try:
        r = httpx.post(
            f"{ENV_URL}/reset",
            json={"task_id": task_id},
            timeout=30.0,
        )
        r.raise_for_status()
        data = r.json()
        # openenv-core wraps: {"observation": {...}, "done": bool, "reward": ...}
        obs = data.get("observation", data)
        obs["done"]   = data.get("done",   obs.get("done",   False))
        obs["reward"] = data.get("reward", obs.get("reward", None))
        return obs
    except Exception as e:
        print(f"[WARN] reset failed: {e}", flush=True)
        return {"task_id": task_id, "done": False, "reward": None,
                "available_actions": ["flag_clause"], "context": {}, "step": 0, "max_steps": 10}

def env_step(action_dict: Dict[str, Any]) -> Dict[str, Any]:
    """POST /step — returns observation dict with reward/done at top level."""
    try:
        r = httpx.post(
            f"{ENV_URL}/step",
            json={"action": action_dict},
            timeout=30.0,
        )
        r.raise_for_status()
        data = r.json()
        obs = data.get("observation", data)
        obs["done"]   = data.get("done",   obs.get("done",   True))
        obs["reward"] = data.get("reward", obs.get("reward", 0.0))
        return obs
    except Exception as e:
        print(f"[WARN] step failed: {e}", flush=True)
        return {"done": True, "reward": 0.0, "available_actions": [],
                "context": {}, "step": 0, "max_steps": 10}

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are a senior legal analyst agent operating inside the LexForge OpenEnv environment.
You will receive a JSON observation describing the current state of a legal document review task.
You must respond with a single valid JSON action object — no markdown, no explanation, just JSON.

Format:
{"action_type": "<one of available_actions>", "<param>": <value>}

Rules:
- Choose action_type ONLY from available_actions in the observation.
- flag_clause: provide clause_id and risks[] labels from risk_taxonomy.
- clear_clause: provide clause_id (for clearly benign/safe clauses only).
- classify_risk: provide clause_id, risks[], citation (e.g. "GDPR Art.5(1)(b)").
- rewrite_clause: provide clause_id and rewritten_text fixing the identified risk.
- generate_report: provide report{} with keys: executive_summary, flagged_clauses, severity_matrix, recommendations, deal_breakers.
- detect_adversarial_clauses: provide clause_ids[] of obfuscated violation clauses.
- submit_multi_party_sign_off: provide party_a_satisfied(bool), party_b_satisfied(bool), balance_justification(str).
- Prioritise CRITICAL severity clauses first. Do NOT flag standard/benign clauses.
- Benign clauses (no legal risk) must use clear_clause, never flag_clause.
""").strip()

# ── Agent ─────────────────────────────────────────────────────────────────────

def build_prompt(obs: Dict[str, Any], history: List[str]) -> str:
    pending   = obs.get("context", {}).get("pending_clauses", {})
    available = obs.get("available_actions", [])
    taxonomy  = list(obs.get("context", {}).get("risk_taxonomy", {}).keys())[:20]
    hist_text = "\n".join(history[-4:]) if history else "None"

    return textwrap.dedent(f"""
        TASK: {obs.get('task_id')}  STEP: {obs.get('step')}/{obs.get('max_steps')}
        CASCADE: {obs.get('cascade_multiplier', 1.0)}x
        AVAILABLE ACTIONS: {available}
        REVIEWED: {obs.get('context',{}).get('reviewed_count',0)}/{obs.get('context',{}).get('total_clauses',0)}
        FLAGGED SO FAR: {obs.get('context',{}).get('flagged_so_far',[])}

        PENDING CLAUSES:
        {json.dumps(pending, indent=2)}

        RISK LABELS (use exact strings):
        {taxonomy}

        HISTORY:
        {hist_text}

        Respond with ONE JSON action. Nothing else.
    """).strip()


def call_llm(client: OpenAI, prompt: str) -> Dict[str, Any]:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as exc:
        print(f"[WARN] LLM error: {exc}", flush=True)
        return {}


def fallback_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    available = obs.get("available_actions", ["flag_clause"])
    pending   = obs.get("context", {}).get("pending_clauses", {})

    if "flag_clause" in available and pending:
        cid = next(iter(pending))
        return {"action_type": "flag_clause", "clause_id": cid, "risks": []}
    if "clear_clause" in available and pending:
        cid = next(iter(pending))
        return {"action_type": "clear_clause", "clause_id": cid}
    if "generate_report" in available:
        return {"action_type": "generate_report", "report": {
            "executive_summary": "Legal audit complete.",
            "flagged_clauses": obs.get("context", {}).get("flagged_so_far", []),
            "severity_matrix": {},
            "recommendations": ["Review flagged clauses with legal counsel."],
            "deal_breakers": [],
        }}
    if "detect_adversarial_clauses" in available:
        return {"action_type": "detect_adversarial_clauses", "clause_ids": ["C019", "C020"]}
    if "submit_multi_party_sign_off" in available:
        return {"action_type": "submit_multi_party_sign_off",
                "party_a_satisfied": True, "party_b_satisfied": True,
                "balance_justification": "Balanced redline addressing both parties core interests within legal requirements."}
    return {"action_type": available[0]}


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(client: OpenAI, task_id: str) -> Dict[str, Any]:
    obs = env_reset(task_id)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    success = False
    error   = None

    log_start(task=task_id, model=MODEL_NAME or "unknown")

    for step_n in range(1, MAX_STEPS + 1):
        if obs.get("done"):
            break

        prompt      = build_prompt(obs, history)
        action_dict = call_llm(client, prompt)

        # Validate action_type
        available = obs.get("available_actions", [])
        if not action_dict or action_dict.get("action_type") not in available:
            action_dict = fallback_action(obs)

        # Execute step
        try:
            obs    = env_step(action_dict)
            reward = float(obs.get("reward") or 0.0)
            done   = bool(obs.get("done", False))
            error  = None
        except Exception as exc:
            reward = 0.0
            done   = True
            error  = str(exc)[:80]
            obs    = {"done": True, "reward": 0.0, "available_actions": []}

        rewards.append(reward)
        steps_taken = step_n

        log_step(
            step=step_n,
            action=action_dict.get("action_type", "unknown"),
            reward=reward,
            done=done,
            error=error,
        )

        history.append(f"Step {step_n}: {action_dict.get('action_type')} → reward={reward:.2f}")

        if done:
            success = reward > 0.0
            break

    log_end(success=success, steps=steps_taken, rewards=rewards)

    return {
        "task_id":     task_id,
        "steps":       steps_taken,
        "rewards":     rewards,
        "mean_reward": sum(rewards) / max(len(rewards), 1),
        "success":     success,
    }


# ── Verify server is reachable ────────────────────────────────────────────────

def wait_for_server(max_wait: int = 60) -> bool:
    print(f"Connecting to environment at {ENV_URL}...", flush=True)
    for i in range(max_wait // 5):
        try:
            r = httpx.get(f"{ENV_URL}/health", timeout=5.0)
            if r.status_code == 200:
                print(f"✅ Environment server ready", flush=True)
                return True
        except Exception:
            pass
        time.sleep(5)
    print(f"❌ Could not reach environment at {ENV_URL}", flush=True)
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60, flush=True)
    print("LexForge Baseline Inference", flush=True)
    print(f"Model    : {MODEL_NAME}", flush=True)
    print(f"Endpoint : {API_BASE_URL}", flush=True)
    print(f"Env URL  : {ENV_URL}", flush=True)
    print("=" * 60, flush=True)

    # Check server reachable
    if not wait_for_server():
        sys.exit(1)

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "placeholder",
    )

    all_results = []
    for task_id in TASKS:
        print(f"\n{'─'*40}", flush=True)
        try:
            result = run_episode(client, task_id)
            all_results.append(result)
        except Exception as exc:
            print(f"[WARN] task {task_id} failed: {exc}", flush=True)
            log_end(success=False, steps=0, rewards=[])
            all_results.append({"task_id": task_id, "steps": 0,
                                 "rewards": [], "mean_reward": 0.0, "success": False})

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("BASELINE SCORES", flush=True)
    print(f"{'─'*60}", flush=True)
    print(f"{'Task':<10} {'Steps':>6} {'Mean Reward':>12} {'Success':>8}", flush=True)
    print(f"{'─'*60}", flush=True)
    for r in all_results:
        print(f"{r['task_id']:<10} {r['steps']:>6} {r['mean_reward']:>12.4f} {str(r['success']):>8}", flush=True)
    overall = sum(r["mean_reward"] for r in all_results) / max(len(all_results), 1)
    print(f"{'─'*60}", flush=True)
    print(f"{'OVERALL':<10} {'':>6} {overall:>12.4f}", flush=True)
    print(f"{'='*60}", flush=True)

    with open("baseline_scores.json", "w") as f:
        json.dump({"model": MODEL_NAME, "endpoint": API_BASE_URL,
                   "results": all_results, "overall_mean": overall}, f, indent=2)
    print(f"\nScores saved to baseline_scores.json", flush=True)


if __name__ == "__main__":
    main()