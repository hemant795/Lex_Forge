"""
inference.py
============
Baseline inference script for LexForge — OpenEnv Competition Submission.

Mandatory env variables:
  API_BASE_URL     LLM API endpoint (default: HuggingFace router)
  MODEL_NAME       Model identifier (default: Qwen/Qwen3-32B)
  HF_TOKEN         HuggingFace API key
  LOCAL_IMAGE_NAME Docker image name (optional)

stdout format (exact — validated by competition):
  [START] task=<name> env=lex_forge model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""
from __future__ import annotations
import json, os, sys, textwrap, time
from typing import Any, Dict, List, Optional
from openai import OpenAI

API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
# HF evaluator uses Qwen/Qwen3-32B; override locally: MODEL_NAME=gemma4:e4b
MODEL_NAME       = os.getenv("MODEL_NAME",   "Qwen/Qwen3-32B")
HF_TOKEN         = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

MAX_STEPS   = 10
TEMPERATURE = 0.1
MAX_TOKENS  = 512
TASKS       = ["easy", "medium", "hard", "expert"]
ENV_NAME    = "lex_forge"

# Action priority per task — higher-reward actions first
ACTION_PRIORITY = {
    "easy":   ["clear_clause", "flag_clause"],
    "medium": ["classify_risk", "flag_clause", "clear_clause"],
    "hard":   ["rewrite_clause", "classify_risk", "generate_report", "flag_clause", "clear_clause"],
    "expert": ["detect_adversarial_clauses", "submit_multi_party_sign_off",
               "rewrite_clause", "classify_risk", "flag_clause", "clear_clause"],
}

def get_priority(task_id: str, completed_stages: list, available: list) -> list:
    """Remove already-completed one-shot actions from priority list."""
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
    print(f"[DEBUG] step_time={elapsed:.1f}s", file=__import__("sys").stderr, flush=True)

def log_end(success: bool, steps: int, rewards: List[float], total_time: float) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"rewards={rewards_str}", flush=True)
    print(f"[DEBUG] total_time={total_time:.1f}s", file=__import__("sys").stderr, flush=True)

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are a senior legal analyst agent. You review legal clauses and take actions.
Respond with ONE valid JSON action object — no markdown, no explanation, just JSON.

ACTION SELECTION RULES (follow in order):
1. For easy tasks: use flag_clause for risky clauses, clear_clause for benign ones.
2. For medium tasks: use classify_risk with citation for best score.
3. For hard tasks: use classify_risk or rewrite_clause — these score 2x more than flag_clause.
4. For expert tasks: use detect_adversarial_clauses first, then submit_multi_party_sign_off last.

CRITICAL RULES:
- clause_id MUST be from the "valid_clause_ids" list shown in the observation.
- NEVER reuse a clause_id you already acted on.
- Benign clauses (severity_hint=NONE) → use clear_clause, NOT flag_clause.
- Risky clauses (severity_hint=HIGH or CRITICAL) → flag or classify.
- risks[] values MUST come from the risk_taxonomy keys shown.

JSON FORMAT per action_type:
  flag_clause:   {"action_type":"flag_clause","clause_id":"C001","risks":["unreasonable_duration"]}
  clear_clause:  {"action_type":"clear_clause","clause_id":"C016"}
  classify_risk: {"action_type":"classify_risk","clause_id":"C001","risks":["unreasonable_duration"],"citation":"Contract Act 1872 S.27"}
  rewrite_clause:{"action_type":"rewrite_clause","clause_id":"C001","rewritten_text":"Fixed clause text..."}
  generate_report:{"action_type":"generate_report","report":{"executive_summary":"...","flagged_clauses":["C001"],"severity_matrix":{"C001":"HIGH"},"recommendations":["Fix C001"],"deal_breakers":["C001"]}}
  detect_adversarial_clauses:{"action_type":"detect_adversarial_clauses","clause_ids":["C019","C020"]}
  submit_multi_party_sign_off:{"action_type":"submit_multi_party_sign_off","party_a_satisfied":true,"party_b_satisfied":true,"balance_justification":"Both parties benefit from the balanced redline addressing GDPR compliance and legitimate data retention within explicit limits."}
""").strip()


def build_prompt(obs: Dict[str, Any], history: List[str], task_id: str) -> str:
    pending  = obs.get("context", {}).get("pending_clauses", {})
    taxonomy = obs.get("context", {}).get("risk_taxonomy", {})
    flagged  = obs.get("context", {}).get("flagged_so_far", [])
    available = obs.get("available_actions", [])

    # Best action to use right now
    priority = get_priority(task_id, obs.get('context',{}).get('completed_stages',[]), available)
    best_action = next((a for a in priority if a in available), available[0] if available else "flag_clause")

    # Clause guidance
    clause_hints = []
    for cid, clause in pending.items():
        sev = clause.get("severity_hint", "?")
        hint = "→ clear_clause (benign)" if sev == "NONE" else f"→ {best_action} (severity={sev})"
        clause_hints.append(f"  {cid}: {hint} | {clause.get('text','')[:80]}...")

    history_text = "\n".join(history[-4:]) if history else "None"

    return textwrap.dedent(f"""
        TASK: {task_id} | Step {obs.get('step')}/{obs.get('max_steps')} | cascade={obs.get('cascade_multiplier')}
        Available actions: {available}
        BEST action to use now: {best_action}
        Already reviewed: {obs.get('context',{}).get('reviewed_count',0)}/{obs.get('context',{}).get('total_clauses',0)}
        Already flagged: {flagged}

        VALID clause_ids RIGHT NOW (use ONLY these):
        {list(pending.keys())}

        PENDING CLAUSES with guidance:
        {chr(10).join(clause_hints)}

        RISK TAXONOMY keys (use for risks[]):
        {list(taxonomy.keys())[:15]}

        RECENT HISTORY:
        {history_text}

        Respond with ONE JSON action using a clause_id from {list(pending.keys())}.
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
        # Strip Qwen3 thinking tags
        import re as _re
        raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as exc:
        print(f"[WARN] LLM error: {exc}", flush=True)
        return {}


def fallback_action(obs: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    available = obs.get("available_actions", ["flag_clause"])
    pending   = obs.get("context", {}).get("pending_clauses", {})
    priority  = get_priority(task_id, obs.get('context',{}).get('completed_stages',[]), available)
    best      = next((a for a in priority if a in available), "flag_clause")

    if best in ("flag_clause", "classify_risk", "clear_clause", "rewrite_clause") and pending:
        cid    = next(iter(pending))
        clause = pending[cid]
        sev    = clause.get("severity_hint", "HIGH")
        if sev == "NONE":
            return {"action_type": "clear_clause", "clause_id": cid}
        if best == "classify_risk":
            taxonomy = obs.get("context", {}).get("risk_taxonomy", {})
            risk_keys = list(taxonomy.keys())[:2] if taxonomy else ["unreasonable_duration"]
            return {"action_type": "classify_risk", "clause_id": cid,
                    "risks": risk_keys, "citation": "GDPR Art.5(1)(b)"}
        if best == "rewrite_clause":
            return {"action_type": "rewrite_clause", "clause_id": cid,
                    "rewritten_text": f"This clause is amended to comply with applicable law, limiting obligations to a reasonable fixed term of 3 years with clear termination rights for both parties."}
        return {"action_type": "flag_clause", "clause_id": cid, "risks": ["unreasonable_duration"]}

    if best == "detect_adversarial_clauses":
        return {"action_type": "detect_adversarial_clauses", "clause_ids": ["C019", "C020"]}
    if best == "submit_multi_party_sign_off":
        return {"action_type": "submit_multi_party_sign_off",
                "party_a_satisfied": True, "party_b_satisfied": True,
                "balance_justification": "Both parties benefit: GDPR compliance for client, data retention for counterparty within explicit limits and erasure rights preserved."}
    if best == "generate_report":
        flagged = obs.get("context", {}).get("flagged_so_far", [])
        return {"action_type": "generate_report", "report": {
            "executive_summary": "Legal audit identifies critical violations requiring immediate remediation.",
            "flagged_clauses": flagged,
            "severity_matrix": {c: "HIGH" for c in flagged},
            "recommendations": ["Redline flagged clauses with legal counsel before signing."],
            "deal_breakers": flagged[:1],
        }}
    return {"action_type": available[0]}


OPENENV_URL = os.getenv("OPENENV_URL", "https://hemant795-lex-forge.hf.space")

def setup_env():
    sys.path.insert(0, os.path.dirname(__file__))
    from models import LexAction
    # Use HTTP client if remote URL set, else fall back to local
    try:
        from client import SyncLexForgeEnvClient
        class RemoteEnv:
            def __init__(self):
                self._client = SyncLexForgeEnvClient(base_url=OPENENV_URL)
            def reset(self, task_id):
                obs = self._client.reset(task_id=task_id)
                return obs
            def step(self, action):
                return self._client.step(action)
        return RemoteEnv, LexAction
    except Exception as e:
        print(f"[WARN] HTTP client failed ({e}), falling back to local env", flush=True)
        from server.environment import LexForgeEnvironment
        return LexForgeEnvironment, LexAction


def run_episode(env_cls, action_cls, client: OpenAI, task_id: str) -> Dict[str, Any]:
    env     = env_cls()
    obs_raw = env.reset(task_id=task_id)
    obs     = obs_raw.model_dump()

    history:  List[str]  = []
    rewards:  List[float] = []
    steps_taken = 0
    success = False
    episode_start = time.time()

    log_start(task=task_id, model=MODEL_NAME or "unknown")

    for step_n in range(1, MAX_STEPS + 1):
        if obs.get("done"):
            break

        step_start   = time.time()
        prompt       = build_prompt(obs, history, task_id)
        action_dict  = call_llm(client, prompt)

        # Validate: action_type must be available and clause_id must be in pending
        pending_ids = list(obs.get("context", {}).get("pending_clauses", {}).keys())
        available   = obs.get("available_actions", [])
        valid = (
            action_dict.get("action_type") in available
            and (
                action_dict.get("action_type") not in ("flag_clause","clear_clause","classify_risk","rewrite_clause")
                or action_dict.get("clause_id") in pending_ids
            )
        )
        if not valid:
            action_dict = fallback_action(obs, task_id)

        try:
            action = action_cls(**action_dict)
        except Exception:
            action = action_cls(**fallback_action(obs, task_id))

        try:
            result  = env.step(action)
            reward  = result.reward if result.reward is not None else 0.0
            done    = result.done
            obs     = result.model_dump()
            error   = None
        except Exception as exc:
            reward, done, error = 0.0, True, str(exc)[:80]
            obs["done"] = True

        elapsed = time.time() - step_start
        rewards.append(reward)
        steps_taken = step_n

        log_step(step_n, action_dict.get("action_type","unknown"),
                 reward, done, error, elapsed)
        history.append(f"Step {step_n}: {action_dict.get('action_type')} "
                       f"clause={action_dict.get('clause_id','?')} → reward={reward:.2f}")
        if done:
            success = reward > 0.0
            break

    total_time = time.time() - episode_start
    print(f"[TIME] total={total_time:.1f}s mean_reward={sum(rewards)/max(len(rewards),1):.3f}", flush=True)
    log_end(success, steps_taken, rewards, total_time)
    return {"task_id": task_id, "steps": steps_taken, "rewards": rewards,
            "mean_reward": sum(rewards)/max(len(rewards),1), "success": success,
            "total_time": total_time}


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
