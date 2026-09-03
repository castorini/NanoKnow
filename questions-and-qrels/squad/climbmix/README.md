# SQuAD projected onto ClimbMix

`topics.nanoknow-squad-climbmix.supported.tsv` holds the SQuAD questions that the ClimbMix corpus can
actually answer, `...unsupported.tsv` the rest, and `qrels.nanoknow-squad-climbmix.supported.txt` the
answering documents. The two topic files partition all 10570 questions; qids
match `../answers.nanoknow-squad.jsonl`.

## What "supported" means here

A question is supported only if a ClimbMix document genuinely **answers it** with the gold answer.

1. **Contamination filter.** 142 questions removed up front: the answer
   is guessable from the question text, or the question turns on nationality.
2. **Retrieval.** Question, question+answer, and rare-term BM25 queries; a document is a candidate
   only if it contains a gold answer string and shares real question vocabulary.
   6448 questions got at least one candidate.
3. **Verification.** An agent read the candidates (at most 3 per question) and judged whether the
   document answers THIS question - same entity, same place, same time frame. The answer string
   merely occurring somewhere is not support. 6448 questions judged.
4. **Adversarial re-check.** A second, independent agent tried to refute every survivor.
   4331 re-checked, 80 refuted and dropped.
5. **Snippet audit.** Every cited snippet was machine-checked to occur verbatim in its document.

## Benchmark leakage

ClimbMix contains quiz pages, model-evaluation dumps (`Ground Truth Answers: ... Prediction: ...`)
and NLP papers that reproduce SQuAD verbatim. Such a document "supports" a question perfectly while
making it useless for retrieval evaluation, since a system would be graded against a copy of the
answer key. 78 questions whose evidence document poses the question and gives
its answer were excluded. The detector is conservative in the direction of dropping: a Wikipedia
article that merely shares wording with the question is NOT flagged, because SQuAD questions were
written from those articles and that overlap is the grounding working correctly.

## qrels

Relevance is granted only on confirmed evidence: the document an agent verified, plus any candidate
containing that verified snippet verbatim. Documents are never marked relevant for containing the
answer string with loose word overlap. Judgments are then closed over the CMASS duplicate map (exact
copies and near duplicates at Jaccard >= 0.7) in both directions, since the
map holds one row per document and a judged document's partner does not always carry a reciprocal
row. This added 3 exact and 568 near-duplicate judgments.

| | |
|---|---:|
| SQuAD questions | 10570 |
| supported | 4193 |
| unsupported | 6377 |
| qrel pairs | 6581 |
| mean documents per supported question | 1.57 |

## Caveat

This is a precision-first release: a question ships only if it was checked and held. Questions whose
retrieval found no candidate were never verified and ship as `unsupported` rather than as unverified
claims, so the supported file understates how much of SQuAD ClimbMix could support.
