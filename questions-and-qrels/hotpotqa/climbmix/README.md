# HotpotQA projected onto ClimbMix

`topics.nanoknow-hotpotqa-climbmix.supported.tsv` holds the HotpotQA questions that the ClimbMix
corpus can actually answer, `...unsupported.tsv` the rest, and
`qrels.nanoknow-hotpotqa-climbmix.supported.txt` the supporting documents for the supported ones.
The two topic files partition all 7405 HotpotQA questions.

## What "supported" means here

A question is supported only if **every** one of its official HotpotQA supporting facts is stated by
a ClimbMix document, and the gold answer follows from those same documents.

1. **Contamination filter.** 864 questions were removed up front: those
   whose answer is guessable from the question text, and nationality questions. They appear in the
   unsupported file.
2. **Retrieval.** Each supporting fact was searched several ways, because a ClimbMix document states
   a fact in its own words rather than Wikipedia's - the fact sentence, the article title plus the
   fact, the title plus the fact's rarest terms, and the title alone.
3. **Verification.** Word overlap alone passes documents that are merely on topic, so every candidate
   question was judged by an agent reading the documents: a fact counts as supported only when one
   document states the whole fact about the same entity, including every date, number and proper name,
   quoted verbatim. Answerability was then judged separately - the gold answer had to follow from the
   confirmed documents, not from the model's own knowledge. **1235 questions were
   verified this way and 553 survived**, so roughly half to three quarters of the
   questions retrieval called supported did not hold up.
4. **Snippet audit.** Every cited snippet was machine-checked to occur verbatim in the document named.

## qrels

Relevance is granted only on confirmed evidence: the document an agent verified for each fact, any
candidate document containing that verified snippet verbatim, and any candidate carrying every hard
element of an already-confirmed fact. Documents are never marked relevant for containing the answer
string with loose word overlap - that is the signal verification exists to reject.

Following the CMASS construction, the judgments are then closed over the ClimbMix duplicate map:
exact copies and near duplicates at Jaccard >= 0.7, so a system that
retrieves a copy is not counted wrong. The closure runs in both directions, since the map holds one
row per document and a judged document's partner does not always carry a reciprocal row.
This added 3 exact and 264 near-duplicate judgments.

| | |
|---|---:|
| HotpotQA questions | 7405 |
| supported | 553 |
| unsupported | 6852 |
| qrel pairs | 2879 |
| mean documents per supported question | 5.21 |

Answers are in `../answers.nanoknow-hotpotqa.jsonl`; qids match that file.

## Benchmark leakage

ClimbMix contains quiz pages and NLP papers that reproduce these benchmarks verbatim - question,
answer key and all. A document like that will "support" a question perfectly while making it
worthless for retrieval evaluation, since the system would be graded against a copy of the answer
key. 4 questions whose evidence document poses the question and gives its
answer were excluded on that basis. The detector is deliberately conservative in the direction of
dropping: an article that merely shares wording with the question is NOT treated as leakage, because
these questions were written from such articles and that overlap is the grounding working correctly.

## Caveat on coverage

Verification is the expensive step, so it was spent highest-yield-first rather than uniformly.
Questions that were never verified ship as unsupported rather than as unverified claims: the
supported file is meant to contain only questions that were checked and held, so it understates how
many HotpotQA questions ClimbMix could support.
