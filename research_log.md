# Research Log

## Purpose

This file is a simple running notebook for the `memory_lab` project.

Use it to capture:
- what experiment was run
- what memory policy was tested
- what failed or looked surprising
- what the next refinement idea should be

## Initial hypotheses

1. `SaveAll` should have very high recall because it throws nothing away.
2. `SaveAll` may also create more retrieval confusion as distractors become more similar.
3. `RuleBasedSalience` may reduce confusion and storage size, but can lose useful facts when its rules are too aggressive.
4. Hard distractors should be the most informative early stress test because they expose whether retrieval is using the correct structured identity fields.

## Key Results (as of 2026-04-11)

### Result 1: Memory policy comparison
- SaveAll: perfect recall, highest storage
- RuleBasedSalience: misses useful facts, catastrophic on late-relevance (0% accuracy)
- TieredMemory: matches SaveAll recall with less storage
- StubMemory: prevents total forgetting but cannot return exact answers

### Result 2: Discriminator Ladder Discovery
- Database Whisper infers field rankings from ambiguity structure
- Coarse splitter: evidence_type (best for neighborhood narrowing)
- Final tie-breaker: therapy (best for disambiguation under ambiguity)
- Source field acts as ID leak, not a semantic discriminator
- Semantic ladder: identity → evidence_type → therapy

### Result 3: Two-Stage Routing
- Staged route (identity → evidence_type → therapy) drives confusion to 0%
- Identity-only: 30-41% confusion under ambiguity
- Identity+therapy: 18-26% confusion
- Two-stage: 0% confusion on both CIViC and ClinVar
- Chooser rule: low ambiguity → identity only; high ambiguity → staged route

### Result 4: Cross-Domain Transfer
- Method transferred from CIViC to ClinVar-like data
- Same ladder shape discovered: identity → evidence_type → therapy
- Same chooser rule emerged
- Two-stage routing hit 0% confusion on both domains
- Stub policies failed badly on ClinVar (100% confusion) — routing works, stubs do not

### Current Diagnosis
- Routing layer is strong and generalizing
- Stub memory format is broken: stubs do not retain the answer-bearing field
- Next step: fix stub answer-retention, then re-test

## First experiment to run

Command:

```powershell
python baseline_runner.py --episodes 20 --distractor-level medium
```

Questions to inspect:
- Which policy gets the right answer most often?
- Which policy stores fewer records on average?
- When the answer is wrong, is it a miss or a confusion error?
