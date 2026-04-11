"""
real_civic_sample.py

This module holds a hand-curated CIViC sample derived from official CIViC API and
documentation examples.

The goal is not to mirror the full CIViC schema. The goal is to map a small real slice
into the existing memory_lab research schema so we can rerun the same routing experiments
without redesigning the lab.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


# This dataclass stores one raw real-data record before normalization.
# What this does:
# - Keeps the CIViC-like fields we care about from the official example evidence items.
# Why this exists:
# - We want to preserve the source-facing structure long enough to normalize it explicitly.
# What assumption it is making:
# - These fields are enough to map a small real CIViC slice into the current sandbox.
@dataclass(frozen=True)
class RealCivicSampleRecord:
    record_id: str
    gene: str
    variant: str
    disease: str
    evidence_type: str
    therapy: str
    evidence_level: str
    direction: str
    source: str
    statement: str


# These records are drawn from official CIViC API and CIViC documentation examples.
# What this does:
# - Provides a larger real-data slice we can ship locally with the project.
# Why this exists:
# - The user asked us to scale the real CIViC validation without adding package or data
#   loading complexity.
# What assumption it is making:
# - A documentation-derived slice is enough for a first larger-scale real-data stability
#   check, even though it is still much smaller than the full CIViC knowledgebase.
#
# Source references used when creating this sample:
# - CIViC API Reference evidence index examples (EID1, EID2)
# - CIViC API Reference evidence detail example (EID512)
# - CIViC API Reference variant detail example for ALK F1174L (EID1271)
# - CIViC API Reference assertion detail example for ERBB2 amplification (EID528, EID529, EID1122)
# - CIViC documentation evidence level examples for BRAF, FLT3, and NPM1
REAL_CIVIC_SAMPLE = [
    RealCivicSampleRecord(
        record_id="EID1",
        gene="JAK2",
        variant="V617F",
        disease="Lymphoid Leukemia",
        evidence_type="Diagnostic",
        therapy="NO_THERAPY",
        evidence_level="B",
        direction="supports",
        source="PMID:16081687",
        statement="JAK2 V617F is not associated with lymphoid leukemia.",
    ),
    RealCivicSampleRecord(
        record_id="EID2",
        gene="PDGFRA",
        variant="D842V",
        disease="Gastrointestinal Stromal Tumor",
        evidence_type="Diagnostic",
        therapy="NO_THERAPY",
        evidence_level="B",
        direction="supports",
        source="PMID:15146165",
        statement="PDGFRA D842V GIST tumors are more likely to be benign than malignant.",
    ),
    RealCivicSampleRecord(
        record_id="EID512",
        gene="SMARCA4",
        variant="Nonsense Mutation",
        disease="Small Cell Carcinoma Of The Ovary Hypercalcemic Type",
        evidence_type="Diagnostic",
        therapy="NO_THERAPY",
        evidence_level="B",
        direction="supports",
        source="PMID:24658004",
        statement="Nonsense mutations in SMARCA4 were associated with small cell carcinoma of the ovary, hypercalcemic type.",
    ),
    RealCivicSampleRecord(
        record_id="EID1271",
        gene="ALK",
        variant="F1174L",
        disease="Neuroblastoma",
        evidence_type="Predictive",
        therapy="Crizotinib",
        evidence_level="B",
        direction="does_not_support",
        source="PMID:23598171",
        statement="ALK F1174L neuroblastoma did not respond well to crizotinib in a pediatric phase I trial subset.",
    ),
    RealCivicSampleRecord(
        record_id="EID528",
        gene="ERBB2",
        variant="AMPLIFICATION",
        disease="Her2-receptor Positive Breast Cancer",
        evidence_type="Predictive",
        therapy="Trastuzumab",
        evidence_level="B",
        direction="supports",
        source="PMID:11248153",
        statement="ERBB2 amplification predicts sensitivity to trastuzumab in metastatic breast cancer.",
    ),
    RealCivicSampleRecord(
        record_id="EID529",
        gene="ERBB2",
        variant="AMPLIFICATION",
        disease="Her2-receptor Positive Breast Cancer",
        evidence_type="Predictive",
        therapy="Trastuzumab",
        evidence_level="B",
        direction="supports",
        source="PMID:15911866",
        statement="ERBB2 amplification predicts sensitivity to trastuzumab plus docetaxel in metastatic breast cancer.",
    ),
    RealCivicSampleRecord(
        record_id="EID1122",
        gene="ERBB2",
        variant="AMPLIFICATION",
        disease="Her2-receptor Positive Breast Cancer",
        evidence_type="Predictive",
        therapy="Trastuzumab",
        evidence_level="A",
        direction="supports",
        source="PMID:16236737",
        statement="ERBB2 amplification predicts sensitivity to adjuvant trastuzumab in breast cancer.",
    ),
    # These additional rows are documentation-derived examples from the official CIViC
    # evidence model pages. They are still real CIViC examples, but some are described in
    # prose rather than exposed as one exact EID in the API reference page.
    RealCivicSampleRecord(
        record_id="DOC-BRAF-A",
        gene="BRAF",
        variant="V600E",
        disease="Melanoma",
        evidence_type="Predictive",
        therapy="Vemurafenib",
        evidence_level="A",
        direction="supports",
        source="CIViC_DOCS:evidence_level_A_example",
        statement="BRAF V600E predicts sensitivity to vemurafenib in untreated metastatic melanoma in a phase 3 randomized clinical trial example.",
    ),
    RealCivicSampleRecord(
        record_id="DOC-BRAF-B",
        gene="BRAF",
        variant="V600E",
        disease="Melanoma",
        evidence_type="Predictive",
        therapy="Vemurafenib",
        evidence_level="B",
        direction="supports",
        source="CIViC_DOCS:evidence_level_B_example",
        statement="BRAF V600E predicts sensitivity to vemurafenib in previously treated melanoma in a phase 2 randomized clinical trial example.",
    ),
    RealCivicSampleRecord(
        record_id="DOC-BRAF-C",
        gene="BRAF",
        variant="V600E",
        disease="Melanoma",
        evidence_type="Predictive",
        therapy="Pictilisib",
        evidence_level="C",
        direction="supports",
        source="CIViC_DOCS:evidence_level_C_example",
        statement="A single patient with BRAF V600E melanoma demonstrated sensitivity to pictilisib in a case-study example.",
    ),
    RealCivicSampleRecord(
        record_id="DOC-BRAF-D",
        gene="BRAF",
        variant="V600E",
        disease="Melanoma",
        evidence_type="Predictive",
        therapy="Dactolisib + Selumetinib",
        evidence_level="D",
        direction="supports",
        source="CIViC_DOCS:evidence_level_D_example",
        statement="BRAF-mutant melanoma cell lines exhibited resistance to dactolisib plus selumetinib in a preclinical evidence example.",
    ),
    RealCivicSampleRecord(
        record_id="DOC-BRAF-E",
        gene="BRAF",
        variant="V600 Amplification",
        disease="Colorectal Cancer",
        evidence_type="Predictive",
        therapy="Selumetinib",
        evidence_level="E",
        direction="supports",
        source="CIViC_DOCS:evidence_level_E_example",
        statement="BRAF V600 amplification was described as a possible mechanism of selumetinib resistance in colorectal cancer in an inferential example.",
    ),
    RealCivicSampleRecord(
        record_id="DOC-FLT3-C",
        gene="FLT3",
        variant="Over-expression",
        disease="Acute Myeloid Leukemia",
        evidence_type="Predictive",
        therapy="Sunitinib",
        evidence_level="C",
        direction="supports",
        source="CIViC_DOCS:evidence_level_case_study_definition",
        statement="A single patient with FLT3 over-expression responded to the FLT3 inhibitor sunitinib in a CIViC case-study example.",
    ),
    RealCivicSampleRecord(
        record_id="DOC-FLT3-D",
        gene="FLT3",
        variant="Internal Tandem Duplication",
        disease="Acute Myeloid Leukemia",
        evidence_type="Predictive",
        therapy="AG1296",
        evidence_level="D",
        direction="supports",
        source="CIViC_DOCS:evidence_level_preclinical_definition",
        statement="AG1296 was effective in triggering apoptosis in cells with FLT3 internal tandem duplication in a preclinical CIViC example.",
    ),
    RealCivicSampleRecord(
        record_id="DOC-NPM1-A",
        gene="NPM1",
        variant="Mutation",
        disease="Acute Myeloid Leukemia",
        evidence_type="Diagnostic",
        therapy="NO_THERAPY",
        evidence_level="A",
        direction="supports",
        source="CIViC_DOCS:evidence_level_validated_definition",
        statement="AML with mutated NPM1 is described as a provisional entity in WHO classification and is used as a validated association example.",
    ),
    RealCivicSampleRecord(
        record_id="DOC-BRAF-PROG-B",
        gene="BRAF",
        variant="V600E",
        disease="Papillary Thyroid Cancer",
        evidence_type="Prognostic",
        therapy="NO_THERAPY",
        evidence_level="B",
        direction="supports",
        source="CIViC_DOCS:evidence_level_clinical_definition",
        statement="BRAF V600E is correlated with poor prognosis in papillary thyroid cancer in a CIViC clinical evidence example.",
    ),
    RealCivicSampleRecord(
        record_id="DOC-ERBB2-LOWPOWER",
        gene="ERBB2",
        variant="AMPLIFICATION",
        disease="Esophageal Adenocarcinoma",
        evidence_type="Predictive",
        therapy="Capecitabine + Oxaliplatin + Cetuximab",
        evidence_level="B",
        direction="does_not_support",
        source="CIViC_DOCS:evidence_rating_one_star_example",
        statement="A low-powered clinical study found no outcome difference for ERBB2 amplification under capecitabine, oxaliplatin, and cetuximab-based therapy.",
    ),
    RealCivicSampleRecord(
        record_id="DOC-EGFR-PRED",
        gene="EGFR",
        variant="Mutation",
        disease="Non-Small Cell Lung Cancer",
        evidence_type="Predictive",
        therapy="Erlotinib",
        evidence_level="B",
        direction="supports",
        source="CIViC_DOCS:therapy_example",
        statement="EGFR mutation is used as a predictive therapy example for response to erlotinib in CIViC documentation.",
    ),
    RealCivicSampleRecord(
        record_id="DOC-ESR1-PRED",
        gene="ESR1",
        variant="Y537S",
        disease="Breast Cancer",
        evidence_type="Predictive",
        therapy="Fulvestrant",
        evidence_level="C",
        direction="supports",
        source="CIViC_DOCS:evidence_overview_example",
        statement="ESR1 Y537S is used in CIViC documentation as an example molecular profile with therapeutic relevance in breast cancer.",
    ),
]


# This helper returns the built-in real sample as a plain list.
# What this does:
# - Gives the rest of the project a tiny stable real-data slice to work with.
# Why this exists:
# - Keeping the sample in one place makes the real-data path easy to inspect and revise.
# What assumption it is making:
# - A static local slice is enough for the first real-data validation pass.
def load_real_civic_sample() -> List[RealCivicSampleRecord]:
    return list(REAL_CIVIC_SAMPLE)
