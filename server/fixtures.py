"""
server/fixtures.py
==================
All clause fixture data for LexForge.
Pure data — no logic, no imports except typing.

Contents
--------
CLAUSES          — dict of 20 clause specs C001-C020 (law-verified)
TASK_CLAUSE_SETS — maps task_id → list of clause IDs
RISK_TAXONOMY    — all valid risk label strings with descriptions
IMPROVEMENT_KEYWORDS — keywords grader looks for in rewrites per risk label
BENIGN_CLAUSE_IDS    — clauses that are safe (false-positive traps)
ADVERSARIAL_CLAUSE_IDS — obfuscated violation clauses (expert task)
"""

from typing import Any, Dict, List

# ─── Clause Registry ──────────────────────────────────────────────────────────

CLAUSES: Dict[str, Dict[str, Any]] = {

    # ── RISKY: Contract Intelligence ─────────────────────────────────────────

    "C001": {
        "id": "C001",
        "domain": "contract",
        "contract_type": "NDA",
        "severity": "HIGH",
        "text": (
            "The Receiving Party shall keep all Confidential Information strictly "
            "secret for a period of UNLIMITED years following the date of this Agreement."
        ),
        "risks": ["unreasonable_duration", "unenforceable_term"],
        "indian_law": "Void under Contract Act 1872 S.27 — restraint of trade",
        "global_law": "Unenforceable under common law — must be reasonable duration",
        "rewrite_hint": "Replace 'UNLIMITED years' with a fixed term of 3-5 years from disclosure date.",
        "is_benign": False,
        "is_adversarial": False,
    },

    "C002": {
        "id": "C002",
        "domain": "contract",
        "contract_type": "NDA",
        "severity": "CRITICAL",
        "text": (
            "The Receiving Party may share Confidential Information with any third party "
            "provided the Receiving Party deems it appropriate at its sole discretion."
        ),
        "risks": ["unilateral_disclosure_authority", "no_third_party_controls"],
        "indian_law": "Contract Act S.16 — undue influence; S.41 Specific Relief Act — injunction available",
        "global_law": "Violates fundamental NDA purpose — no consent requirement",
        "rewrite_hint": "Require prior written consent from Disclosing Party before any third-party disclosure.",
        "is_benign": False,
        "is_adversarial": False,
    },

    "C003": {
        "id": "C003",
        "domain": "ip",
        "contract_type": "NDA",
        "severity": "HIGH",
        "text": (
            "All intellectual property created during this engagement, including all "
            "inventions, works, and discoveries, shall vest solely in the Client "
            "without any further compensation to the Receiving Party."
        ),
        "risks": ["ip_ownership_overreach", "work_for_hire_ambiguity"],
        "indian_law": "Patents Act S.6+S.70 — inventor must assign in writing. Copyright Act S.17 — employer owns works IN COURSE of employment only.",
        "global_law": "Work-for-hire doctrine — must carve out pre-existing IP",
        "rewrite_hint": "Limit IP assignment to work created specifically under this engagement. Exclude pre-existing IP.",
        "is_benign": False,
        "is_adversarial": False,
    },

    # ── RISKY: Regulatory / Data Protection ──────────────────────────────────

    "C004": {
        "id": "C004",
        "domain": "gdpr",
        "contract_type": "SaaS",
        "severity": "CRITICAL",
        "text": (
            "The Provider shall process Customer Personal Data for any purpose "
            "the Provider considers commercially reasonable, including product "
            "improvement, analytics, and marketing."
        ),
        "risks": ["gdpr_purpose_limitation_violation", "lawful_basis_missing"],
        "indian_law": "DPDPA 2023 S.5 — purpose limitation. S.4 — consent must be specific. Penalty up to ₹200 crore.",
        "global_law": "GDPR Art.5(1)(b) — purpose limitation. Art.6 — lawful basis required.",
        "rewrite_hint": "Limit processing to stated, explicit purposes only. Document lawful basis for each purpose.",
        "is_benign": False,
        "is_adversarial": False,
    },

    "C005": {
        "id": "C005",
        "domain": "gdpr",
        "contract_type": "SaaS",
        "severity": "HIGH",
        "text": (
            "Personal data collected under this agreement may be retained indefinitely "
            "to improve service quality and for future product development purposes."
        ),
        "risks": ["gdpr_storage_limitation_violation", "no_retention_schedule"],
        "indian_law": "DPDPA 2023 S.8(7) — data must be erased after purpose fulfilled. Penalty up to ₹200 crore.",
        "global_law": "GDPR Art.5(1)(e) — storage limitation. Must define maximum retention period.",
        "rewrite_hint": "Define maximum retention period (e.g. 2 years post-contract). Establish automated deletion schedule.",
        "is_benign": False,
        "is_adversarial": False,
    },

    "C006": {
        "id": "C006",
        "domain": "gdpr",
        "contract_type": "SaaS",
        "severity": "CRITICAL",
        "text": (
            "Customer grants Provider an irrevocable, perpetual, worldwide license "
            "to use, copy, modify, and distribute all Customer data for any purpose."
        ),
        "risks": ["gdpr_data_ownership_conflict", "excessive_data_rights", "irrevocable_license"],
        "indian_law": "DPDPA 2023 S.4+S.8 — consent can always be withdrawn. S.8 — right to erasure.",
        "global_law": "GDPR Art.17 — right to erasure. Art.20 — data portability. Irrevocable license conflicts with both.",
        "rewrite_hint": "Limit license to service delivery purposes only. Remove 'irrevocable'. Preserve Customer right to deletion and portability.",
        "is_benign": False,
        "is_adversarial": False,
    },

    # ── RISKY: AML & Financial Compliance ────────────────────────────────────

    "C007": {
        "id": "C007",
        "domain": "aml",
        "contract_type": "Financial",
        "severity": "CRITICAL",
        "text": (
            "Payment may be made via bearer instruments, cash equivalents, "
            "cryptocurrency, or any other mutually agreed method without "
            "further identity verification."
        ),
        "risks": ["aml_risk", "no_kyc", "bearer_instrument_exposure"],
        "indian_law": "PMLA 2002 S.3 — money laundering offence. FEMA S.3 — unauthorised forex. Bearer cheques above ₹20,000 banned.",
        "global_law": "FATF Recommendation 10 — customer due diligence mandatory. OFAC — sanctions screening required.",
        "rewrite_hint": "Restrict to traceable payment methods only (bank transfer, verified digital payment). Require KYC documentation before payment.",
        "is_benign": False,
        "is_adversarial": False,
    },

    "C008": {
        "id": "C008",
        "domain": "aml",
        "contract_type": "Financial",
        "severity": "HIGH",
        "text": (
            "All fees shall be invoiced and processed through a third-party "
            "intermediary of the Provider's choosing, without obligation to "
            "disclose the identity of such intermediary."
        ),
        "risks": ["aml_layering_risk", "transparency_gap", "beneficial_ownership_concealment"],
        "indian_law": "PMLA S.12 — reporting entities must identify all parties. Benami Act 2016 — concealing beneficial ownership.",
        "global_law": "FATF Recommendation 24 — beneficial ownership transparency. EU AMLD — intermediary disclosure required.",
        "rewrite_hint": "Name the intermediary specifically. Require 30-day advance notice if intermediary changes. Disclose beneficial ownership.",
        "is_benign": False,
        "is_adversarial": False,
    },

    # ── RISKY: Employment & Labour ────────────────────────────────────────────

    "C009": {
        "id": "C009",
        "domain": "employment",
        "contract_type": "Employment",
        "severity": "HIGH",
        "text": (
            "All compensation under this agreement shall be structured as consulting "
            "fees to minimise applicable employment tax obligations for both parties, "
            "and the Contractor shall be solely responsible for all tax filings."
        ),
        "risks": ["tax_misclassification_risk", "ir35_exposure", "employment_law_evasion"],
        "indian_law": "Labour Codes 2020 — economic reality test. PF/ESI mandatory if employee. Courts look at control, exclusivity, integration.",
        "global_law": "IR35 (UK) — off-payroll working rules. IRS 20-factor test (US). Misclassification = arrears + penalties.",
        "rewrite_hint": "Classify correctly based on working relationship substance. If employee, use employment contract with statutory deductions.",
        "is_benign": False,
        "is_adversarial": False,
    },

    # ── RISKY: Intellectual Property ─────────────────────────────────────────

    "C010": {
        "id": "C010",
        "domain": "ip",
        "contract_type": "Technology",
        "severity": "HIGH",
        "text": (
            "Employee hereby assigns to Employer all right, title, and interest "
            "in any and all inventions, whether or not related to Employee's work, "
            "created during the term of employment, including inventions made "
            "on personal time using personal equipment."
        ),
        "risks": ["ip_ownership_overreach", "personal_time_capture", "pre_existing_ip_risk"],
        "indian_law": "Copyright Act S.17 — employer owns only works 'in course of employment'. Patents Act S.6 — inventor must assign in writing.",
        "global_law": "California Labor Code S.2870 — prohibits assignment of inventions unrelated to employer's business.",
        "rewrite_hint": "Limit assignment to inventions directly related to the scope of employment. Carve out personal time and unrelated inventions.",
        "is_benign": False,
        "is_adversarial": False,
    },

    # ── RISKY: Corporate Governance / M&A ────────────────────────────────────

    "C011": {
        "id": "C011",
        "domain": "corporate",
        "contract_type": "M&A",
        "severity": "HIGH",
        "text": (
            "Either party may terminate this Agreement upon the occurrence of "
            "a Material Adverse Change, as determined in the sole and absolute "
            "discretion of the terminating party."
        ),
        "risks": ["mac_clause_ambiguity", "unilateral_mac_determination", "deal_certainty_risk"],
        "indian_law": "No statutory MAC definition. Specific Relief Act S.14 — seller can seek specific performance if buyer walks away citing MAC.",
        "global_law": "MAC clauses must have objective standard (ABN AMRO v. Hexion). Subjective MAC allows abuse.",
        "rewrite_hint": "Define MAC with objective measurable thresholds (e.g. >20% revenue decline). List specific exclusions: general economic conditions, pandemic.",
        "is_benign": False,
        "is_adversarial": False,
    },

    # ── RISKY: Real Estate ────────────────────────────────────────────────────

    "C012": {
        "id": "C012",
        "domain": "real_estate",
        "contract_type": "Lease",
        "severity": "MEDIUM",
        "text": (
            "Annual rent shall be increased each year by the higher of (i) 10% "
            "or (ii) the Retail Price Index, with no upper limit on the total "
            "cumulative increase over the lease term."
        ),
        "risks": ["uncapped_rent_escalation", "tenant_financial_exposure"],
        "indian_law": "State Rent Control Acts — rent escalation clauses may be void for controlled premises. RERA 2016 — escalation must be agreed upfront.",
        "global_law": "RICS Code — rent reviews must have cap. UK Landlord and Tenant Act — open market rent review standard.",
        "rewrite_hint": "Cap total cumulative increase at 20% over lease term. Use fixed percentage only — not RPI-linked without cap.",
        "is_benign": False,
        "is_adversarial": False,
    },

    # ── RISKY: Criminal Law / Evidence ───────────────────────────────────────

    "C013": {
        "id": "C013",
        "domain": "criminal",
        "contract_type": "Evidence",
        "severity": "HIGH",
        "text": (
            "Evidence Item EX-004 (biological sample) was collected at 09:00 on "
            "14 March and transferred to the forensic laboratory at 15:30 on "
            "15 March. No record of storage, handling, or custody exists for the "
            "intervening 30-hour period."
        ),
        "risks": ["chain_of_custody_gap", "evidence_admissibility_risk"],
        "indian_law": "BSA 2023 S.63 — electronic/physical evidence admissibility requires documented chain of custody. Panchnama required at seizure.",
        "global_law": "4th Amendment (US) — evidence handling gaps challenge admissibility. PACE (UK) — Code B requires continuous custody record.",
        "rewrite_hint": "Document complete custody log for all 30 hours. Identify each custodian, storage location, and transfer time. Obtain FSL certification.",
        "is_benign": False,
        "is_adversarial": False,
    },

    # ── RISKY: Immigration ────────────────────────────────────────────────────

    "C014": {
        "id": "C014",
        "domain": "immigration",
        "contract_type": "Employment",
        "severity": "MEDIUM",
        "text": (
            "The Company confirms that right-to-work verification was completed "
            "for all new hires within 30 days of their employment commencement date."
        ),
        "risks": ["i9_violation", "right_to_work_timing_breach"],
        "indian_law": "Foreigners Act 1946 — FRRO registration must be filed within 14 days of arrival. Employer liable for employing overstayed visa holder.",
        "global_law": "I-9 (US) — must be completed on day 1 or before, not within 30 days. Tier 2 (UK) — right-to-work check before employment starts.",
        "rewrite_hint": "Change to: right-to-work verification completed BEFORE or ON the first day of employment commencement.",
        "is_benign": False,
        "is_adversarial": False,
    },

    # ── RISKY: Litigation ─────────────────────────────────────────────────────

    "C015": {
        "id": "C015",
        "domain": "litigation",
        "contract_type": "Settlement",
        "severity": "HIGH",
        "text": (
            "The parties agree that the terms of this Settlement Agreement are "
            "strictly confidential and neither party shall disclose any information "
            "relating to this matter to any person, authority, or court, without "
            "the prior written consent of the other party."
        ),
        "risks": ["confidentiality_overreach", "mandatory_reporting_blocked", "court_disclosure_prevented"],
        "indian_law": "BSA 2023 S.125 — court can compel disclosure. BNS — confidentiality cannot prevent FIR filing for cognisable offences.",
        "global_law": "SEC Rule 21F-17 (US) — settlement cannot prevent reporting to regulators. UK FCA — confidentiality cannot prevent regulatory disclosure.",
        "rewrite_hint": "Add carve-out: 'except as required by law, court order, or mandatory regulatory reporting obligations.'",
        "is_benign": False,
        "is_adversarial": False,
    },

    # ── BENIGN (false-positive traps) ─────────────────────────────────────────

    "C016": {
        "id": "C016",
        "domain": "contract",
        "contract_type": "NDA",
        "severity": "NONE",
        "text": (
            "This Agreement shall be governed by and construed in accordance "
            "with the laws of England and Wales, and the parties submit to "
            "the exclusive jurisdiction of the courts of England and Wales."
        ),
        "risks": [],
        "indian_law": "Standard governing law clause — enforceable in Indian courts for commercial disputes.",
        "global_law": "Standard choice of law clause — fully enforceable.",
        "rewrite_hint": "",
        "is_benign": True,
        "is_adversarial": False,
    },

    "C017": {
        "id": "C017",
        "domain": "contract",
        "contract_type": "SaaS",
        "severity": "NONE",
        "text": (
            "Either party may terminate this Agreement for convenience upon "
            "thirty (30) days' written notice to the other party."
        ),
        "risks": [],
        "indian_law": "Standard termination clause — S.55 Contract Act — reasonable notice satisfies the requirement.",
        "global_law": "Standard mutual termination — industry norm for SaaS agreements.",
        "rewrite_hint": "",
        "is_benign": True,
        "is_adversarial": False,
    },

    "C018": {
        "id": "C018",
        "domain": "contract",
        "contract_type": "NDA",
        "severity": "NONE",
        "text": (
            "Confidential Information does not include information that: "
            "(a) is or becomes publicly available through no fault of the "
            "Receiving Party; (b) was known to the Receiving Party prior to "
            "disclosure; or (c) is independently developed by the Receiving Party."
        ),
        "risks": [],
        "indian_law": "Standard NDA carve-out — consistent with Contract Act principles of free consent and public interest.",
        "global_law": "Standard confidentiality exclusions — present in all well-drafted NDAs worldwide.",
        "rewrite_hint": "",
        "is_benign": True,
        "is_adversarial": False,
    },

    # ── ADVERSARIAL (expert task — obfuscated violations) ────────────────────

    "C019": {
        "id": "C019",
        "domain": "gdpr",
        "contract_type": "Adversarial",
        "severity": "CRITICAL",
        "text": (
            "For operational efficiency, the Processing Entity reserves the right "
            "to engage sub-processing partners on a rolling administrative basis, "
            "notification of which shall be communicated through the Provider's "
            "standard administrative newsletter distributed to registered account holders."
        ),
        "risks": ["gdpr_subprocessor_consent_violation", "no_prior_notice"],
        "indian_law": "DPDPA 2023 S.4+S.8 — consent required before sub-processor engagement. Rules still being notified.",
        "global_law": "GDPR Art.28(2) — prior written consent required before engaging sub-processors. Post-facto newsletter notification is insufficient.",
        "rewrite_hint": "Require explicit prior written consent before any new sub-processor engagement. Provide minimum 30-day advance notice.",
        "is_benign": False,
        "is_adversarial": True,
        "obfuscation_note": "Violation hidden in neutral operational language. 'Administrative newsletter' disguises the lack of prior consent.",
    },

    "C020": {
        "id": "C020",
        "domain": "litigation",
        "contract_type": "Adversarial",
        "severity": "HIGH",
        "text": (
            "All disputes arising from this Agreement shall be finally resolved "
            "through binding resolution proceedings administered by a qualified "
            "neutral appointed through the Provider's standard dispute resolution "
            "framework, whose decision shall be conclusive and not subject to appeal."
        ),
        "risks": ["unfair_arbitration", "no_neutral_arbitrator", "appeal_rights_waived"],
        "indian_law": "Arbitration Act 1996 S.12(5) — arbitrator appointed by one party is disqualified. Perkins Eastman v. HSCC (SC 2019).",
        "global_law": "ICC/UNCITRAL rules — arbitrator must be independent of both parties. No-appeal clauses limited by public policy.",
        "rewrite_hint": "Use mutually agreed neutral arbitration institution (ICC, LCIA, MCIA). Ensure right of appeal on points of law.",
        "is_benign": False,
        "is_adversarial": True,
        "obfuscation_note": "'Provider's standard dispute resolution framework' hides that the provider controls arbitrator selection.",
    },
}

# ─── Task → Clause Sets ───────────────────────────────────────────────────────

TASK_CLAUSE_SETS: Dict[str, List[str]] = {
    "easy":   ["C001", "C002", "C003", "C016", "C017"],
    "medium": ["C004", "C005", "C006", "C007", "C008", "C012", "C014", "C017", "C018"],
    "hard":   ["C001", "C002", "C003", "C007", "C009", "C010", "C011", "C013", "C016", "C017", "C018"],
    "expert": list(CLAUSES.keys()),  # All 20
}

# ─── Task metadata ────────────────────────────────────────────────────────────

TASK_MAX_STEPS: Dict[str, int] = {
    "easy":   5,
    "medium": 8,
    "hard":   12,
    "expert": 15,
}

TASK_DESCRIPTIONS: Dict[str, str] = {
    "easy":   "Identify risky vs benign clauses in a 5-clause startup NDA. Contract Intelligence domain only.",
    "medium": "Multi-regulation compliance audit: GDPR + AML + Real Estate + Immigration across 9 clauses.",
    "hard":   "Full M&A due diligence redline across 5 legal domains. Flag, classify, rewrite, and produce audit report.",
    "expert": "Adversarial multi-jurisdiction contract with obfuscated violations and multi-party sign-off required.",
}

TASK_AVAILABLE_ACTIONS: Dict[str, List[str]] = {
    "easy":   ["flag_clause", "clear_clause"],
    "medium": ["flag_clause", "clear_clause", "classify_risk", "cite_regulation"],
    "hard":   ["flag_clause", "clear_clause", "classify_risk", "cite_regulation",
               "rewrite_clause", "generate_report"],
    "expert": ["flag_clause", "clear_clause", "classify_risk", "cite_regulation",
               "rewrite_clause", "generate_report",
               "detect_adversarial_clauses", "submit_multi_party_sign_off"],
}

# ─── Risk Taxonomy ────────────────────────────────────────────────────────────

RISK_TAXONOMY: Dict[str, str] = {
    "unreasonable_duration":                "Duration term is unreasonably long or unlimited — likely unenforceable.",
    "unenforceable_term":                   "Term cannot be enforced as written under applicable law.",
    "unilateral_disclosure_authority":      "One party can share confidential information without the other's consent.",
    "no_third_party_controls":              "No restrictions on what third parties can do with disclosed information.",
    "ip_ownership_overreach":               "IP assignment is broader than the scope of the engagement.",
    "work_for_hire_ambiguity":              "Unclear whether work-for-hire doctrine applies — creates ownership dispute risk.",
    "personal_time_capture":               "Employer attempts to claim IP created on personal time with personal resources.",
    "pre_existing_ip_risk":                "Pre-existing IP of one party is captured by over-broad assignment clause.",
    "gdpr_purpose_limitation_violation":   "Data processed beyond its stated, explicit, and specified purpose (GDPR Art.5(1)(b)).",
    "lawful_basis_missing":                "No documented lawful basis for processing personal data (GDPR Art.6).",
    "gdpr_storage_limitation_violation":   "Data retained beyond the period necessary for its stated purpose (GDPR Art.5(1)(e)).",
    "no_retention_schedule":               "No defined data retention schedule or deletion trigger.",
    "gdpr_data_ownership_conflict":        "Excessive data rights conflict with data subject's right to erasure (GDPR Art.17).",
    "excessive_data_rights":               "License to data is broader than necessary for service delivery.",
    "irrevocable_license":                 "License cannot be revoked, conflicting with right to withdraw consent.",
    "aml_risk":                            "Payment method or structure creates money laundering exposure.",
    "no_kyc":                              "No identity verification (Know Your Customer) requirement specified.",
    "bearer_instrument_exposure":          "Bearer instruments allow anonymous transfer — prohibited in most jurisdictions.",
    "aml_layering_risk":                   "Intermediary structure obscures the origin or destination of funds.",
    "transparency_gap":                    "Insufficient disclosure of parties, amounts, or purposes in financial flows.",
    "beneficial_ownership_concealment":    "Beneficial owner of funds or assets is not identified or disclosed.",
    "tax_misclassification_risk":          "Worker classified as contractor to avoid employment tax — may be misclassification.",
    "ir35_exposure":                       "Arrangement may fall inside IR35 (UK) or IRS worker classification tests.",
    "employment_law_evasion":              "Structure designed to avoid statutory employment protections.",
    "mac_clause_ambiguity":               "Material Adverse Change clause lacks objective standard for determination.",
    "unilateral_mac_determination":        "One party alone determines whether a MAC has occurred — no objective test.",
    "deal_certainty_risk":                 "Clause creates uncertainty about whether the transaction will close.",
    "uncapped_rent_escalation":            "Rent can increase without limit, creating unpredictable tenant financial exposure.",
    "tenant_financial_exposure":           "Clause creates disproportionate financial risk for the tenant.",
    "chain_of_custody_gap":               "Evidence handling gap compromises admissibility in court.",
    "evidence_admissibility_risk":         "Procedural defect may result in evidence being excluded at trial.",
    "i9_violation":                        "Right-to-work verification not completed before or on first day of employment.",
    "right_to_work_timing_breach":         "Verification timing does not meet statutory requirement.",
    "confidentiality_overreach":           "Confidentiality clause prevents legally required disclosures.",
    "mandatory_reporting_blocked":         "Settlement or NDA prevents party from making mandatory regulatory reports.",
    "court_disclosure_prevented":          "Clause attempts to block disclosure ordered by a court of competent jurisdiction.",
    "gdpr_subprocessor_consent_violation": "Sub-processors engaged without prior written consent of data controller (GDPR Art.28(2)).",
    "no_prior_notice":                     "Changes are communicated after the fact rather than in advance.",
    "unfair_arbitration":                  "Arbitration administered by or controlled by one party — lacks independence.",
    "no_neutral_arbitrator":              "Arbitrator selected by one party — conflicts with natural justice.",
    "appeal_rights_waived":               "Right to appeal arbitral award is waived — disproportionate finality.",
}

# ─── Improvement Keywords (for rewrite grader) ───────────────────────────────

IMPROVEMENT_KEYWORDS: Dict[str, List[str]] = {
    "unreasonable_duration":                ["years", "fixed", "term", "specific", "period", "defined"],
    "unenforceable_term":                   ["enforceable", "reasonable", "lawful", "valid"],
    "unilateral_disclosure_authority":      ["written consent", "prior approval", "mutual", "agree"],
    "no_third_party_controls":              ["nda", "confidentiality", "restrict", "bound", "obligation"],
    "ip_ownership_overreach":               ["scope", "carve", "exclude", "pre-existing", "limit", "related"],
    "work_for_hire_ambiguity":              ["clearly", "defined", "scope", "employment", "specific"],
    "personal_time_capture":               ["personal time", "exclude", "own equipment", "unrelated"],
    "pre_existing_ip_risk":                ["pre-existing", "carve-out", "prior", "exclude", "retain"],
    "gdpr_purpose_limitation_violation":   ["specified", "explicit", "lawful", "purpose", "limited", "only"],
    "lawful_basis_missing":                ["lawful basis", "consent", "legitimate", "legal basis", "article 6"],
    "gdpr_storage_limitation_violation":   ["retention", "delete", "erase", "period", "schedule", "months"],
    "no_retention_schedule":               ["schedule", "period", "delete", "after", "months", "years"],
    "gdpr_data_ownership_conflict":        ["erasure", "portability", "right", "delete", "withdraw", "revoke"],
    "excessive_data_rights":               ["service delivery", "limited", "specific", "necessary", "only"],
    "irrevocable_license":                 ["revocable", "withdraw", "terminate", "consent", "right to"],
    "aml_risk":                            ["kyc", "traceable", "verified", "bank transfer", "identification"],
    "no_kyc":                              ["identity", "verify", "kyc", "documentation", "proof"],
    "bearer_instrument_exposure":          ["traceable", "bank", "verified", "electronic", "identified"],
    "aml_layering_risk":                   ["disclose", "identity", "beneficial", "named", "transparent"],
    "transparency_gap":                    ["disclose", "name", "identity", "beneficial owner", "notify"],
    "beneficial_ownership_concealment":    ["beneficial owner", "disclose", "identify", "name", "reveal"],
    "tax_misclassification_risk":          ["employment", "employee", "statutory", "classify", "correct"],
    "ir35_exposure":                       ["ir35", "employment status", "worker", "classify", "assessment"],
    "employment_law_evasion":              ["employment contract", "statutory", "rights", "entitled", "proper"],
    "mac_clause_ambiguity":               ["objective", "measurable", "threshold", "percentage", "criteria"],
    "unilateral_mac_determination":        ["mutual", "agreed", "objective", "both parties", "independent"],
    "deal_certainty_risk":                 ["certain", "defined", "objective", "clear", "specified"],
    "uncapped_rent_escalation":            ["cap", "maximum", "limit", "ceiling", "no more than"],
    "tenant_financial_exposure":           ["cap", "protect", "maximum", "ceiling", "limit"],
    "chain_of_custody_gap":               ["continuous", "documented", "record", "custody", "log", "seal"],
    "evidence_admissibility_risk":         ["procedure", "documented", "certified", "compliant", "chain"],
    "i9_violation":                        ["before", "prior to", "first day", "commencement", "day one"],
    "right_to_work_timing_breach":         ["before", "prior", "first day", "start date", "commence"],
    "confidentiality_overreach":           ["except", "required by law", "court order", "regulatory", "mandatory"],
    "mandatory_reporting_blocked":         ["regulatory", "required", "law", "authority", "except"],
    "court_disclosure_prevented":          ["court order", "required by law", "legal obligation", "except"],
    "gdpr_subprocessor_consent_violation": ["prior", "written consent", "advance notice", "approve", "before"],
    "no_prior_notice":                     ["advance", "prior", "notice", "before", "days notice"],
    "unfair_arbitration":                  ["neutral", "mutual", "independent", "agreed", "institution"],
    "no_neutral_arbitrator":              ["neutral", "independent", "both parties", "mutually", "agreed"],
    "appeal_rights_waived":               ["appeal", "review", "court", "challenge", "right"],
}

# ─── Convenience sets ─────────────────────────────────────────────────────────

BENIGN_CLAUSE_IDS: List[str] = [
    cid for cid, c in CLAUSES.items() if c["is_benign"]
]

ADVERSARIAL_CLAUSE_IDS: List[str] = [
    cid for cid, c in CLAUSES.items() if c["is_adversarial"]
]

RISKY_CLAUSE_IDS: List[str] = [
    cid for cid, c in CLAUSES.items()
    if not c["is_benign"] and not c["is_adversarial"]
]