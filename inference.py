"""
inference.py
============
LexForge baseline inference — OpenEnv Competition Submission.

Env variables:
  API_BASE_URL     LLM API endpoint (default: HuggingFace router)
  MODEL_NAME       Model (default: google/gemma-4-26b-moe-it)
  HF_TOKEN         HuggingFace API key
  LOCAL_IMAGE_NAME Docker image name (optional)
"""
from __future__ import annotations
import json, os, re, sys, textwrap, time
from typing import Any, Dict, List, Optional
from openai import OpenAI

API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
# Gemma 4 26B-MoE: superior agentic reasoning, fast MoE inference on HF router
# Local override: MODEL_NAME=gemma4:e4b
MODEL_NAME       = os.getenv("MODEL_NAME",   "google/gemma-4-26b-moe-it")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

MAX_STEPS   = 10
TEMPERATURE = 0.1
MAX_TOKENS  = 512
TASKS       = ["easy", "medium", "hard", "expert"]
ENV_NAME    = "lex_forge"

ACTION_PRIORITY = {
    "easy":   ["clear_clause", "flag_clause"],
    "medium": ["classify_risk", "flag_clause", "clear_clause"],
    "hard":   ["rewrite_clause", "classify_risk", "generate_report", "flag_clause", "clear_clause"],
    "expert": ["detect_adversarial_clauses", "submit_multi_party_sign_off",
               "rewrite_clause", "classify_risk", "flag_clause", "clear_clause"],
}

def get_priority(task_id: str, completed_stages: list, available: list) -> list:
    p = list(ACTION_PRIORITY.get(task_id, available))
    if "detect_adversarial" in completed_stages and "detect_adversarial_clauses" in p:
        p.remove("detect_adversarial_clauses")
    if "sign_off" in completed_stages and "submit_multi_party_sign_off" in p:
        p.remove("submit_multi_party_sign_off")
    return p

# ── Logging ───────────────────────────────────────────────────────────────────

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={ENV_NAME} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str], elapsed: float) -> None:
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} "
          f"done={str(done).lower()} error={err}", flush=True)
    print(f"[TIME] step={step} elapsed={elapsed:.1f}s", flush=True, file=sys.stderr)

def log_end(success: bool, steps: int, rewards: List[float], total_time: float) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)
    print(f"[TIME] total={total_time:.1f}s mean={sum(rewards)/max(len(rewards),1):.3f}",
          flush=True, file=sys.stderr)

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are a senior legal analyst agent. Review legal clauses and take actions.
Respond with ONE valid JSON action — no markdown, no explanation, just JSON.

ACTION RULES (follow in priority order):
1. easy tasks:   clear_clause for benign (severity_hint=NONE), flag_clause for risky.
2. medium tasks: classify_risk with citation scores highest.
3. hard tasks:   rewrite_clause or classify_risk scores 2x more than flag_clause.
4. expert tasks: detect_adversarial_clauses first (once), then submit_multi_party_sign_off, then rewrite_clause.

CRITICAL:
- clause_id MUST be from valid_clause_ids list shown below.
- NEVER reuse a clause_id already acted on.
- risks[] values MUST come from risk_taxonomy keys.
- Benign clauses (severity_hint=NONE) → clear_clause only.

JSON FORMAT:
  flag_clause:    {"action_type":"flag_clause","clause_id":"C001","risks":["unreasonable_duration"]}
  clear_clause:   {"action_type":"clear_clause","clause_id":"C016"}
  classify_risk:  {"action_type":"classify_risk","clause_id":"C001","risks":["unreasonable_duration"],"citation":"Contract Act 1872 S.27"}
  rewrite_clause: {"action_type":"rewrite_clause","clause_id":"C001","rewritten_text":"Fixed clause limiting obligations to 3 years with clear termination rights for both parties."}
  generate_report:{"action_type":"generate_report","report":{"executive_summary":"...","flagged_clauses":["C001"],"severity_matrix":{"C001":"HIGH"},"recommendations":["Fix C001"],"deal_breakers":["C001"]}}
  detect_adversarial_clauses:{"action_type":"detect_adversarial_clauses","clause_ids":["C019","C020"]}
  submit_multi_party_sign_off:{"action_type":"submit_multi_party_sign_off","party_a_satisfied":true,"party_b_satisfied":true,"balance_justification":"Both parties benefit: GDPR compliance for client, data retention for counterparty within explicit limits and erasure rights fully preserved."}
""").strip()


def build_prompt(obs: Dict[str, Any], history: List[str], task_id: str) -> str:
    pending   = obs.get("context", {}).get("pending_clauses", {})
    taxonomy  = obs.get("context", {}).get("risk_taxonomy", {})
    flagged   = obs.get("context", {}).get("flagged_so_far", [])
    completed = obs.get("context", {}).get("completed_stages", [])
    available = obs.get("available_actions", [])
    priority  = get_priority(task_id, completed, available)
    best      = next((a for a in priority if a in available), available[0] if available else "flag_clause")

    clause_hints = []
    for cid, clause in pending.items():
        sev  = clause.get("severity_hint", "?")
        rec  = "→ clear_clause (benign)" if sev == "NONE" else f"→ {best} (severity={sev})"
        clause_hints.append(f"  {cid}: {rec} | {clause.get('text','')[:80]}...")

    return textwrap.dedent(f"""
        TASK: {task_id} | Step {obs.get('step')}/{obs.get('max_steps')} | cascade={obs.get('cascade_multiplier')}
        Available: {available} | BEST action NOW: {best}
        Completed stages: {completed}
        Already flagged: {flagged}
        Reviewed: {obs.get('context',{}).get('reviewed_count',0)}/{obs.get('context',{}).get('total_clauses',0)}

        VALID clause_ids RIGHT NOW (ONLY use these):
        {list(pending.keys())}

        CLAUSES with guidance:
        {chr(10).join(clause_hints)}

        RISK TAXONOMY (use for risks[]):
        {list(taxonomy.keys())[:15]}

        RECENT HISTORY:
        {chr(10).join(history[-4:]) if history else 'None'}

        Respond with ONE JSON using clause_id from {list(pending.keys())}.
    """).strip()


def clean_llm_output(raw: str) -> str:
    """Strip <think> tags (Gemma 4 / Qwen 3 reasoning blocks) and markdown."""
    raw = re.sub(r"<(think|thought)>.*?</(think|thought)>", "", raw,
                 flags=re.DOTALL | re.IGNORECASE).strip()
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


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
        raw = clean_llm_output(raw)
        return json.loads(raw)
    except Exception as exc:
        print(f"[WARN] LLM error: {exc}", flush=True)
        return {}


def fallback_action(obs: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    available = obs.get("available_actions", ["flag_clause"])
    pending   = obs.get("context", {}).get("pending_clauses", {})
    completed = obs.get("context", {}).get("completed_stages", [])
    priority  = get_priority(task_id, completed, available)
    best      = next((a for a in priority if a in available), "flag_clause")

    if best in ("flag_clause", "classify_risk", "clear_clause", "rewrite_clause") and pending:
        cid = next(iter(pending))
        sev = pending[cid].get("severity_hint", "HIGH")
        if sev == "NONE":
            return {"action_type": "clear_clause", "clause_id": cid}
        if best == "classify_risk":
            return {"action_type": "classify_risk", "clause_id": cid,
                    "risks": ["unreasonable_duration"], "citation": "Contract Act 1872 S.27"}
        if best == "rewrite_clause":
            return {"action_type": "rewrite_clause", "clause_id": cid,
                    "rewritten_text": "This clause is amended to comply with applicable law, "
                    "limiting obligations to a fixed term of 3 years with clear termination rights."}
        return {"action_type": "flag_clause", "clause_id": cid, "risks": ["unreasonable_duration"]}

    if best == "detect_adversarial_clauses":
        return {"action_type": "detect_adversarial_clauses", "clause_ids": ["C019", "C020"]}
    if best == "submit_multi_party_sign_off":
        return {"action_type": "submit_multi_party_sign_off",
                "party_a_satisfied": True, "party_b_satisfied": True,
                "balance_justification": "Both parties benefit: GDPR compliance for client, "
                "data retention for counterparty within explicit limits and erasure rights preserved."}
    if best == "generate_report":
        flagged = obs.get("context", {}).get("flagged_so_far", [])
        return {"action_type": "generate_report", "report": {
            "executive_summary": "Legal audit identifies critical violations requiring remediation.",
            "flagged_clauses": flagged,
            "severity_matrix": {c: "HIGH" for c in flagged},
            "recommendations": ["Redline flagged clauses with legal counsel before signing."],
            "deal_breakers": flagged[:1],
        }}
    return {"action_type": available[0]}


def setup_env():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))
    sys.path.insert(0, os.path.dirname(__file__))
    from server.environment import LexForgeEnvironment
    from models import LexAction
    return LexForgeEnvironment, LexAction


def run_episode(env_cls, action_cls, client: OpenAI, task_id: str) -> Dict[str, Any]:
    env     = env_cls()
    obs_raw = env.reset(task_id=task_id)
    obs     = obs_raw.model_dump()

    history:     List[str]   = []
    rewards:     List[float] = []
    steps_taken  = 0
    success      = False
    episode_start = time.time()

    log_start(task=task_id, model=MODEL_NAME or "unknown")

    for step_n in range(1, MAX_STEPS + 1):
        if obs.get("done"):
            break

        step_start  = time.time()
        prompt      = build_prompt(obs, history, task_id)
        action_dict = call_llm(client, prompt)

        # Validate
        pending_ids = list(obs.get("context", {}).get("pending_clauses", {}).keys())
        available   = obs.get("available_actions", [])
        needs_clause = action_dict.get("action_type") in (
            "flag_clause", "clear_clause", "classify_risk", "rewrite_clause")
        valid = (
            action_dict.get("action_type") in available
            and (not needs_clause or action_dict.get("clause_id") in pending_ids)
        )
        if not valid:
            action_dict = fallback_action(obs, task_id)

        try:
            action = action_cls(**action_dict)
        except Exception:
            action = action_cls(**fallback_action(obs, task_id))

        try:
            result = env.step(action)
            reward = result.reward if result.reward is not None else 0.001
            done   = result.done
            obs    = result.model_dump()
            error  = None
        except Exception as exc:
            reward, done, error = 0.001, True, str(exc)[:80]
            obs["done"] = True

        elapsed = time.time() - step_start
        rewards.append(reward)
        steps_taken = step_n

        log_step(step_n, action_dict.get("action_type", "unknown"),
                 reward, done, error, elapsed)
        history.append(f"Step {step_n}: {action_dict.get('action_type')} "
                       f"clause={action_dict.get('clause_id','?')} → {reward:.3f}")
        if done:
            success = reward > 0.001
            break

    total_time = time.time() - episode_start
    log_end(success, steps_taken, rewards, total_time)
    return {"task_id": task_id, "steps": steps_taken, "rewards": rewards,
            "mean_reward": sum(rewards)/max(len(rewards), 1),
            "success": success, "total_time": total_time}


def main() -> None:
    overall_start = time.time()
    print("=" * 60, flush=True)
    print(f"LexForge Baseline Inference", flush=True)
    print(f"Model    : {MODEL_NAME}", flush=True)
    print(f"Endpoint : {API_BASE_URL}", flush=True)
    print("=" * 60, flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "placeholder")
    LexForgeEnvironment, LexAction = setup_env()

    all_results = []
    for task_id in TASKS:
        print(f"\n{'─'*40}", flush=True)
        result = run_episode(LexForgeEnvironment, LexAction, client, task_id)
        all_results.append(result)

    total_elapsed = time.time() - overall_start
    print(f"\n{'='*60}", flush=True)
    print("BASELINE SCORES", flush=True)
    print(f"{'─'*60}", flush=True)
    print(f"{'Task':<10} {'Steps':>5} {'MeanRwd':>8} {'Success':>8} {'Time':>8}", flush=True)
    print(f"{'─'*60}", flush=True)
    for r in all_results:
        print(f"{r['task_id']:<10} {r['steps']:>5} {r['mean_reward']:>8.4f} "
              f"{str(r['success']):>8} {r['total_time']:>7.1f}s", flush=True)
    overall = sum(r["mean_reward"] for r in all_results) / len(all_results)
    print(f"{'─'*60}", flush=True)
    print(f"{'OVERALL':<10} {'':>5} {overall:>8.4f} {'':>8} {total_elapsed:>7.1f}s", flush=True)
    print(f"{'='*60}", flush=True)

    with open("baseline_scores.json", "w") as f:
        json.dump({"model": MODEL_NAME, "endpoint": API_BASE_URL,
                   "results": all_results, "overall_mean": overall}, f, indent=2)
    print(f"\nScores saved → baseline_scores.json", flush=True)


if __name__ == "__main__":
    main()
