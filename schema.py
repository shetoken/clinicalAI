"""Pydantic schema for visit-note feature extraction.

Designed to feed the gold-tier feature tables described in the CenterWell
Home Health Feature Store spec (visit_note_embeddings_gold, hosp_risk_indicators_gold,
oasis_e_features_gold). Every extracted signal carries verbatim evidence so the
output is auditable and PII/PHI provenance is traceable.

Conservative-by-default: if something isn't documented, the field is null,
False, or "not_documented" — never inferred.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# --------- enums (kept small; Gemini structured output handles these well) ----------

class DyspneaLevel(str, Enum):
    NONE = "none"
    ON_EXERTION = "on_exertion"
    AT_REST = "at_rest"
    UNSPECIFIED_PRESENT = "unspecified_present"
    NOT_DOCUMENTED = "not_documented"


class OverallStatus(str, Enum):
    STABLE = "stable"
    IMPROVING = "improving"
    DECLINING = "declining"
    ACUTE_CONCERN = "acute_concern"


class DiagnosisCluster(str, Enum):
    HF = "HF"
    COPD = "COPD"
    DM = "DM"
    POST_SURGICAL = "post_surgical"
    WOUND = "wound"
    MSK = "MSK"
    NEURO = "neuro"
    OTHER = "other"
    UNCLEAR = "unclear"


class VisitType(str, Enum):
    SOC = "SOC"
    RESUMPTION = "resumption"
    RECERT = "recert"
    FOLLOWUP = "followup"
    DISCHARGE = "discharge"
    UNKNOWN = "unknown"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# --------- sub-blocks ----------

class Vitals(BaseModel):
    """Vital signs documented in the note. All fields nullable — only populate if explicitly stated."""
    bp_systolic_mmhg: int | None = None
    bp_diastolic_mmhg: int | None = None
    heart_rate_bpm: int | None = None
    respiratory_rate: int | None = None
    spo2_percent: int | None = None
    spo2_on_room_air: bool | None = None
    temperature_f: float | None = None
    weight_lbs: float | None = None
    weight_change_lbs: float | None = Field(
        default=None,
        description="Positive = gain, negative = loss. Only populate if note states a delta.",
    )
    weight_change_window_days: int | None = None
    pain_score_0_10: int | None = None


class Cardiopulmonary(BaseModel):
    """High-leverage signals for HF/COPD decompensation and 30-day ACH risk."""
    dyspnea: DyspneaLevel = DyspneaLevel.NOT_DOCUMENTED
    dyspnea_evidence: str | None = Field(
        default=None,
        description="Verbatim quote supporting the dyspnea level.",
    )
    orthopnea: bool = False
    lung_sounds_description: str | None = None
    crackles: bool = False
    wheezes: bool = False
    diminished_breath_sounds: bool = False
    jvd: bool = False
    peripheral_edema: bool = False
    edema_location_grade: str | None = None
    cough: bool = False
    sputum_change: bool = False


class Neurocognitive(BaseModel):
    altered_mental_status: bool = False
    confusion_new_onset: bool = False
    confusion_evidence: str | None = None
    lethargy: bool = False


class Medications(BaseModel):
    prn_diuretic_used: bool = False
    prn_diuretic_evidence: str | None = None
    rescue_inhaler_used: bool = False
    rescue_inhaler_frequency: str | None = None
    inhaler_technique_concern: bool = False
    adherence_concern: bool = False
    adherence_evidence: str | None = None
    new_medications: list[str] = Field(default_factory=list)
    medication_changes: list[str] = Field(default_factory=list)
    high_risk_meds_mentioned: list[str] = Field(
        default_factory=list,
        description="Beers-criteria or otherwise high-risk meds named in the note "
                    "(opioids, benzos, anticoagulants, insulin, etc.).",
    )


class FunctionSafety(BaseModel):
    falls_since_last_visit: bool = False
    fall_count: int | None = None
    fall_risk_concerns: list[str] = Field(default_factory=list)
    ambulation_status: str | None = None
    assistive_device_used: str | None = None
    adl_concerns: list[str] = Field(default_factory=list)
    home_safety_hazards: list[str] = Field(default_factory=list)


class Wound(BaseModel):
    wound_present: bool = False
    wound_location: str | None = None
    wound_stage_npuap: str | None = Field(
        default=None,
        description="NPUAP stage (1, 2, 3, 4, unstageable, DTI) if a pressure injury, else null.",
    )
    wound_size_cm: str | None = None
    tissue_description: str | None = None
    exudate_amount: str | None = None
    exudate_type: str | None = None
    signs_of_infection: bool = False
    infection_signs_evidence: str | None = None


class CareContext(BaseModel):
    caregiver_present_at_visit: bool = False
    caregiver_role: str | None = Field(
        default=None, description="e.g. 'daughter', 'spouse', 'paid aide'."
    )
    caregiver_concerns_reported: list[str] = Field(default_factory=list)
    education_topics_addressed: list[str] = Field(default_factory=list)
    sdoh_concerns: list[str] = Field(
        default_factory=list,
        description="Food insecurity, transportation, housing, financial, social isolation, etc.",
    )


class ClinicalImpression(BaseModel):
    primary_diagnosis_cluster: DiagnosisCluster
    overall_status: OverallStatus
    escalation_indicators: list[str] = Field(
        default_factory=list,
        description="Findings that warrant clinical follow-up or case-manager attention. "
                    "These feed the risk-trigger pipeline.",
    )
    patient_reported_symptoms: list[str] = Field(default_factory=list)
    rationale: str = Field(
        description="One-to-three-sentence justification of status and impression, grounded in the note."
    )


class DocCompleteness(BaseModel):
    """Companion to the visit-note QA use case (Walkthrough 4)."""
    oasis_items_likely_addressed: list[str] = Field(
        default_factory=list,
        description="OASIS-E item codes/names the note appears to address (e.g. M1242 pain, "
                    "M1400 dyspnea, M1860 ambulation).",
    )
    potential_documentation_gaps: list[str] = Field(
        default_factory=list,
        description="Items expected for the primary diagnosis cluster that appear missing or unaddressed.",
    )


class Signal(BaseModel):
    """Flat (name, value, evidence) record — convenient for downstream feature store ingestion."""
    name: str
    value: str
    evidence_quote: str | None = None
    confidence: Confidence = Confidence.MEDIUM


# --------- top-level ----------

class VisitNoteFeatures(BaseModel):
    """Structured feature extraction from one home health visit note."""
    visit_type: VisitType
    vitals: Vitals
    cardiopulmonary: Cardiopulmonary
    neurocognitive: Neurocognitive
    medications: Medications
    function_safety: FunctionSafety
    wound: Wound
    care_context: CareContext
    clinical_impression: ClinicalImpression
    doc_completeness: DocCompleteness
    flat_signals: list[Signal] = Field(
        default_factory=list,
        description="Flat list of the most important extracted signals for direct feature-store landing. "
                    "Each entry must carry an evidence quote.",
    )
    extraction_notes: str | None = Field(
        default=None,
        description="Any caveats about ambiguous wording or items the extractor was unsure about.",
    )
