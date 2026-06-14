from typing import Literal, Optional

from pydantic import BaseModel, Field


class ExtractedRequest(BaseModel):
    """Structured fields parsed from a raw prior-auth referral note.

    Returned by the LLM via `with_structured_output`. Procedure/diagnosis
    descriptions and clinical justification are required -- without them
    there isn't really a request to evaluate. Codes and demographics are
    optional since referral notes often omit them.
    """

    patient_age: Optional[int] = None
    patient_sex: Optional[Literal["M", "F", "other", "unknown"]] = None

    procedure_description: str = Field(
        description="Free-text description of the requested procedure/service."
    )
    procedure_code: Optional[str] = Field(
        default=None, description="CPT/HCPCS code, if mentioned or confidently inferable."
    )

    diagnosis_description: str = Field(
        description="Free-text description of the patient's diagnosis."
    )
    diagnosis_code: Optional[str] = Field(
        default=None, description="ICD-10 code, if mentioned or confidently inferable."
    )

    clinical_justification: str = Field(
        description=(
            "Summary of the clinical rationale for the request -- symptoms, "
            "duration, failed conservative treatments, relevant exam findings, "
            "and anything else a reviewer would weigh against policy criteria."
        )
    )
    requested_setting: Optional[str] = Field(
        default=None,
        description="Care setting for the request, e.g. 'outpatient', 'inpatient', 'office'.",
    )

    extraction_confidence: float = Field(
        description=(
            "Self-assessed confidence (0-1) that this extraction is complete "
            "and correct. Use a low value if required clinical details were "
            "vague, missing, or had to be guessed."
        )
    )
