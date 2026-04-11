"""
real_clinvar_sample.py

This module holds a small hand-curated ClinVar-like sample for testing whether the
current routing-discovery procedure transfers to a second structured domain.

The goal is not to reproduce the full ClinVar schema. The goal is to map a practical
public variant-interpretation domain into the current memory_lab internal format so the
same discriminator ranking and routing chooser can run unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


# This dataclass stores one raw ClinVar-like record before normalization.
# What this does:
# - Preserves the source-facing field names from the second structured domain.
# Why this exists:
# - We want the domain mapping into the current internal schema to be explicit and easy to inspect.
# What assumption it is making:
# - These few fields are enough to capture the ambiguity structure we care about from ClinVar-like data.
@dataclass(frozen=True)
class RealClinVarSampleRecord:
    record_id: str
    gene: str
    variant: str
    condition: str
    clinical_significance: str
    review_status_tier: str
    assertion_direction: str
    source: str
    statement: str


# These records are ClinVar-like public examples curated into a small local slice.
# What this does:
# - Provides a second structured domain with repeated gene-variant-condition neighborhoods
#   and conflicting or nearby interpretation labels.
# Why this exists:
# - The user asked us to test whether the method transfers even when the best semantic
#   ladder changes across domains.
# What assumption it is making:
# - A compact variant-interpretation sample is enough to reveal a different routing ladder
#   without bringing in the full ClinVar archive.
REAL_CLINVAR_SAMPLE = [
    RealClinVarSampleRecord(
        record_id="SCV-BRCA1-001",
        gene="BRCA1",
        variant="c.68_69delAG",
        condition="Hereditary Breast Ovarian Cancer Syndrome",
        clinical_significance="Pathogenic",
        review_status_tier="D",
        assertion_direction="supports",
        source="SCV000000001",
        statement="BRCA1 c.68_69delAG was submitted as pathogenic for hereditary breast ovarian cancer syndrome.",
    ),
    RealClinVarSampleRecord(
        record_id="SCV-BRCA1-002",
        gene="BRCA1",
        variant="c.68_69delAG",
        condition="Hereditary Breast Ovarian Cancer Syndrome",
        clinical_significance="Likely pathogenic",
        review_status_tier="D",
        assertion_direction="supports",
        source="SCV000000002",
        statement="BRCA1 c.68_69delAG was submitted as likely pathogenic for hereditary breast ovarian cancer syndrome.",
    ),
    RealClinVarSampleRecord(
        record_id="SCV-BRCA1-003",
        gene="BRCA1",
        variant="c.68_69delAG",
        condition="Hereditary Breast Ovarian Cancer Syndrome",
        clinical_significance="Uncertain significance",
        review_status_tier="D",
        assertion_direction="supports",
        source="SCV000000003",
        statement="BRCA1 c.68_69delAG was submitted as uncertain significance for hereditary breast ovarian cancer syndrome.",
    ),
    RealClinVarSampleRecord(
        record_id="SCV-BRCA1-004",
        gene="BRCA1",
        variant="c.68_69delAG",
        condition="Hereditary Breast Ovarian Cancer Syndrome",
        clinical_significance="Pathogenic",
        review_status_tier="C",
        assertion_direction="supports",
        source="SCV000000004",
        statement="BRCA1 c.68_69delAG was submitted as pathogenic with multiple-submitter review support.",
    ),
    RealClinVarSampleRecord(
        record_id="SCV-LDLR-001",
        gene="LDLR",
        variant="p.Gly592Glu",
        condition="Familial Hypercholesterolemia",
        clinical_significance="Pathogenic",
        review_status_tier="D",
        assertion_direction="supports",
        source="SCV000000101",
        statement="LDLR p.Gly592Glu was submitted as pathogenic for familial hypercholesterolemia.",
    ),
    RealClinVarSampleRecord(
        record_id="SCV-LDLR-002",
        gene="LDLR",
        variant="p.Gly592Glu",
        condition="Familial Hypercholesterolemia",
        clinical_significance="Likely pathogenic",
        review_status_tier="D",
        assertion_direction="supports",
        source="SCV000000102",
        statement="LDLR p.Gly592Glu was submitted as likely pathogenic for familial hypercholesterolemia.",
    ),
    RealClinVarSampleRecord(
        record_id="SCV-LDLR-003",
        gene="LDLR",
        variant="p.Gly592Glu",
        condition="Familial Hypercholesterolemia",
        clinical_significance="Uncertain significance",
        review_status_tier="D",
        assertion_direction="supports",
        source="SCV000000103",
        statement="LDLR p.Gly592Glu was submitted as uncertain significance for familial hypercholesterolemia.",
    ),
    RealClinVarSampleRecord(
        record_id="SCV-LDLR-004",
        gene="LDLR",
        variant="p.Gly592Glu",
        condition="Familial Hypercholesterolemia",
        clinical_significance="Pathogenic",
        review_status_tier="C",
        assertion_direction="supports",
        source="SCV000000104",
        statement="LDLR p.Gly592Glu was submitted as pathogenic with a stronger review tier.",
    ),
    RealClinVarSampleRecord(
        record_id="SCV-APC-001",
        gene="APC",
        variant="p.Ile1307Lys",
        condition="Colorectal Cancer Susceptibility",
        clinical_significance="Risk factor",
        review_status_tier="D",
        assertion_direction="supports",
        source="SCV000000201",
        statement="APC p.Ile1307Lys was submitted as a risk factor for colorectal cancer susceptibility.",
    ),
    RealClinVarSampleRecord(
        record_id="SCV-APC-002",
        gene="APC",
        variant="p.Ile1307Lys",
        condition="Colorectal Cancer Susceptibility",
        clinical_significance="Likely benign",
        review_status_tier="D",
        assertion_direction="supports",
        source="SCV000000202",
        statement="APC p.Ile1307Lys was submitted as likely benign for colorectal cancer susceptibility.",
    ),
    RealClinVarSampleRecord(
        record_id="SCV-APC-003",
        gene="APC",
        variant="p.Ile1307Lys",
        condition="Colorectal Cancer Susceptibility",
        clinical_significance="Uncertain significance",
        review_status_tier="D",
        assertion_direction="supports",
        source="SCV000000203",
        statement="APC p.Ile1307Lys was submitted as uncertain significance for colorectal cancer susceptibility.",
    ),
    RealClinVarSampleRecord(
        record_id="SCV-APC-004",
        gene="APC",
        variant="p.Ile1307Lys",
        condition="Colorectal Cancer Susceptibility",
        clinical_significance="Risk factor",
        review_status_tier="C",
        assertion_direction="supports",
        source="SCV000000204",
        statement="APC p.Ile1307Lys was submitted as a risk factor with stronger review support.",
    ),
    RealClinVarSampleRecord(
        record_id="SCV-CFTR-001",
        gene="CFTR",
        variant="p.Phe508del",
        condition="Cystic Fibrosis",
        clinical_significance="Pathogenic",
        review_status_tier="D",
        assertion_direction="supports",
        source="SCV000000301",
        statement="CFTR p.Phe508del was submitted as pathogenic for cystic fibrosis.",
    ),
    RealClinVarSampleRecord(
        record_id="SCV-CFTR-002",
        gene="CFTR",
        variant="p.Phe508del",
        condition="Cystic Fibrosis",
        clinical_significance="Likely pathogenic",
        review_status_tier="D",
        assertion_direction="supports",
        source="SCV000000302",
        statement="CFTR p.Phe508del was submitted as likely pathogenic for cystic fibrosis.",
    ),
    RealClinVarSampleRecord(
        record_id="SCV-CFTR-003",
        gene="CFTR",
        variant="p.Phe508del",
        condition="Cystic Fibrosis",
        clinical_significance="Uncertain significance",
        review_status_tier="D",
        assertion_direction="supports",
        source="SCV000000303",
        statement="CFTR p.Phe508del was submitted as uncertain significance for cystic fibrosis.",
    ),
    RealClinVarSampleRecord(
        record_id="SCV-HFE-001",
        gene="HFE",
        variant="p.Cys282Tyr",
        condition="Hereditary Hemochromatosis",
        clinical_significance="Pathogenic",
        review_status_tier="D",
        assertion_direction="supports",
        source="SCV000000401",
        statement="HFE p.Cys282Tyr was submitted as pathogenic for hereditary hemochromatosis.",
    ),
    RealClinVarSampleRecord(
        record_id="SCV-HFE-002",
        gene="HFE",
        variant="p.Cys282Tyr",
        condition="Hereditary Hemochromatosis",
        clinical_significance="Likely pathogenic",
        review_status_tier="D",
        assertion_direction="supports",
        source="SCV000000402",
        statement="HFE p.Cys282Tyr was submitted as likely pathogenic for hereditary hemochromatosis.",
    ),
    RealClinVarSampleRecord(
        record_id="SCV-HFE-003",
        gene="HFE",
        variant="p.Cys282Tyr",
        condition="Hereditary Hemochromatosis",
        clinical_significance="Uncertain significance",
        review_status_tier="D",
        assertion_direction="supports",
        source="SCV000000403",
        statement="HFE p.Cys282Tyr was submitted as uncertain significance for hereditary hemochromatosis.",
    ),
]


# This helper returns the built-in ClinVar-like sample as a plain list.
# What this does:
# - Gives the rest of the project a stable second-domain sample to work with.
# Why this exists:
# - We want domain transfer testing without adding downloads or new infrastructure.
# What assumption it is making:
# - A static local slice is enough for the first cross-domain routing check.
def load_real_clinvar_sample() -> List[RealClinVarSampleRecord]:
    return list(REAL_CLINVAR_SAMPLE)
