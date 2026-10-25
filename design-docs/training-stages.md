# Two-stage training (2026-10-25)

The tagger trains in two stages: **pretraining** on many rule-based corruptions of mostly
clean text, then **fine-tuning** on a small set of LLM-written errors whose labels a second
LLM pass has verified. Real human errors stay for evaluation only (implementation plan,
Phase 4). This replaces two earlier assumptions: that LLM data is mixed into one training
set (about 70% rule, 30% LLM; implementation plan 1c, feature plan F22), and that the whole
clean corpus is corrected by an LLM before corruption.

## Why

The real-word run (PoC plan, Real-word error run) scored val F0.5 0.713 with precision
0.690 and recall 0.827, so precision limits it. Of about 5,000 false positives on the v3
val set, Determiner had 1,577 (881 on "the", "a" and "an"), Preposition 854 and Conjunction
415. A sample of them shows two causes:

- **Errors already in the source text.** Some flags are right ("in both cancerous
  proliferation or vascular repair"), but the corruptor only labels the errors it adds, so
  they count as false positives in evaluation and as `O` in training.
- **Doubtful synthetic labels.** Masked-LM word-choice edits are 46% of RealWord gold
  spans and the tagger finds 16% of them, but many are acceptable English ("who chose to
  prevent").

Correcting the whole corpus with an LLM fixes the first cause, but costs about 28 hours of
annotation per 100,000 paragraphs at 60 a minute (feature plan, corpus annotation), plus a
verification pass. Most of that cost buys little: a large pretraining set tolerates some
label noise, as long as the evaluation data and the final training stage are clean.

## Why not reinforcement learning

Scoring the tagger's output with an LLM judge and training on that reward was considered
and rejected:

- The tagger labels every word, and exact word-level labels can be built. A reward per
  paragraph gives far less signal per example.
- RL needs a fresh judge call on the model's own output at every step. Supervised labels
  are judged once and reused every epoch. The LLM, not the tagger, is the slow part.
- A judge that only checks flags rewards flagging less. To also score misses it must
  annotate the whole paragraph, which is a set of labels anyway.
- One LLM pass is noisy; verification needs a second pass.

The useful part of the idea, spending judge calls where the model is wrong, is kept as
supervised hard examples (stage 2c).

## Stage 1: rule-based pretraining

- **Data:** rule corruptions (`corrupt`) of the clean corpus. Lean on the cleanest
  sources: the generated stories (`tinystories`, `simplestories`, `cosmopedia-stories`)
  drew 4.2 to 5.2 flags per 1,000 words from the real-word tagger, against 6.2 to 10.4 for
  the human sources (lowest FineWeb-Edu, highest Wikinews). Keep the human sources for
  register; their leftover errors are tolerated here.
- **No corpus-wide correction.** The training split is corrupted from raw text as it is.
- **Training:** as today (`contcheck train`, implementation plan 3c), early stop on the
  clean val split below.

### Clean evaluation splits

Measured precision is only meaningful on clean text, so the val and test documents are
corrected before corruption, and only they. `assign_split` places a document by a hash of
its `doc_id`, so these documents are known from the raw corpus before `corrupt` runs.
About 10% of the corpus is corrected (around 10,000 paragraphs for a 100,000-paragraph
corpus), with the corpus-correction pipeline: annotate, verify, correct. Paragraphs whose
annotation fails, or whose correction a check rejects, are dropped from val and test.

## Stage 2: fine-tuning on verified LLM errors

### 2a. The fine-tuning set

- **Clean source:** a few thousand paragraphs not used in stage 1, corrected with the same
  pipeline as the evaluation splits, so the only errors are the ones added next.
- **Errors:** an LLM rewrites each paragraph as a given writer would (implementation plan
  1c and 1f: learner profiles by first language, a fast phone typist, a 12-year-old),
  adding 1 to 4 mistakes. ERRANT labels the edits from `(noisy, clean)`, ignoring the
  LLM's own description, and F21's filters drop rewrites with too many edits or too
  little overlap.
- **Verification:** a second LLM pass checks each edit, as `corpus verify` checks
  corrections: the noisy form is a real mistake, the clean form is correct, and the
  meaning is kept. Rewrites with a rejected edit are dropped, not partly labeled, so no
  unlabeled error is left in a kept paragraph.
- **Size:** start with 5,000 to 10,000 paragraphs, about two LLM calls each (rewrite and
  verify), or 3 to 6 hours at the annotation rate. Split by document like stage 1, with its
  own val split.

### 2b. Fine-tuning

- Start from the stage-1 checkpoint with lower learning rates (encoder about 1e-5, head
  about 1e-4), for 1 to 3 epochs.
- Replay 20 to 30% stage-1 records in each batch, so kinds that the LLM rarely writes are
  not forgotten.
- Early stop on the stage-2 val split. Report the stage-1 val split as well; a drop there
  means forgetting.

### 2c. Judge-checked hard examples (optional)

After a stage-2 run:

1. Tag new clean-source paragraphs and new LLM-error paragraphs with the tagger.
2. Pick paragraphs where it is unsure: a flag or a word with error probability between
   about 0.3 and 0.8, or disagreement with the ERRANT labels.
3. Have the judge check each flag (a real mistake or not) and annotate the paragraph for
   missed errors, with verification as in 2a.
4. Add the result to the fine-tuning set as labels: confirmed flags and found misses
   become error spans, rejected flags become `O`. Paragraphs the judge cannot settle are
   dropped.

Each round costs judge calls only on the examples the model gets wrong, and nothing is
judged again during training.

## Evaluation

Every run reports:

- per-kind F0.5, detection F0.5 and false alarms per 1,000 words on the clean stage-1
  test split and the stage-2 test split;
- the probe checks (real misspellings, seen and held-out confusions);
- the real eval sets once F12 lands (BEA-2019 dev, FCE test and the others), which decide
  between stage 1 alone, stage 1 plus 2b, and each round of 2c.

## Done when

- Val and test are built from corrected documents, and the real-word checkpoint is
  re-scored on them, so later runs compare against a clean baseline.
- A stage-2 checkpoint beats its stage-1 checkpoint on the stage-2 test split without
  losing more than 1 point of F0.5 on the stage-1 test split.
- The comparison is repeated on real human errors once F12 is available.

## Open questions

- How large the fine-tuning set must be before more examples stop helping; start at 5,000
  paragraphs and double while the stage-2 test score rises.
- Whether masked-LM word-choice RealWord edits stay in stage 1 at their current share,
  with tighter filter thresholds, or move to stage 2 with verification.
- Which model verifies the LLM-written errors; the corpus-correction pipeline's choice is
  the starting point.
