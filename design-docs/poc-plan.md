# Context-aware tagger — Proof-of-concept plan

The proof of concept trains the baseline tagger on synthetic errors and tags real text
from the command line:

```bash
just setup rocm
just cli corpus fetch --source wikipedia --limit 60000 --out data/raw/wiki.jsonl
just cli corrupt --config configs/gen.yaml --in data/raw --out data/synthetic/rule.jsonl
just cli merge data/synthetic/rule.jsonl --out-dir data/synthetic
just cli report data/synthetic/train.jsonl --config configs/gen.yaml
just cli train --config configs/train.yaml --device cuda:0
just cli tag "Yesterday I go to the store and buyed there apples."
```

It is a thin slice through M1 to M3 of [feature-plan.md](feature-plan.md), built first so
the end-to-end path is proven before the full milestones fill in. Like any design doc, it
does not authorize implementation; it starts when the user asks.

## Not throwaway

Every PoC feature is a full feature from the feature plan with a trimmed scope, built to
the same `AGENTS.md` rules (protocols, typed results, mirrored tests, `just check`). The
full feature later adds the trimmed parts to the same modules instead of rewriting them.
The one PoC-only piece is `contcheck tag`, which runs the PyTorch checkpoint directly; it
stays as a debugging command after M6's ONNX `check()` exists.

## Scope

| ID | PoC scope | Left for the full feature |
| --- | --- | --- |
| F1 | All of F1: `ErrorKind`, `Operation`, labels, BIO helpers | — |
| C2 | `Edit`, `Record`, `CorruptionContext`, `GenConfig`; `Corruptor`, `TextAnalyzer`, `Inflector`, `Dictionary` protocols; corruptor registry; shared fakes; CLI split. Dependencies: `pydantic`, `pyyaml`, `cyclopts`, `spacy`, `en-core-web-sm`, `lemminflect`, `symspellpy`, `datasets` | `Annotator` protocol and `errant` (with F5) |
| F3 | All of F3: spaCy analyzer, tense histogram, doc stats | — |
| F4 | Dictionary membership and suggestions | User words, token skipping rules |
| F6 | Plain-text and JSONL readers, `corpus fetch` streaming an English Wikipedia subset through `datasets`, document grouping, document-level split | SymSpell-based filtering, more sources |
| F7 | All of F7: the applier | — |
| F8 | Spelling | Capitalization, Apostrophe, WordBoundary, Punctuation |
| F9 | RealWord, Determiner | Preposition, Conjunction, ExtraWord, MissingWord, WordOrder |
| F10 | VerbTense, SubjectVerbAgreement, `Inflector` adapter | VerbForm, NounNumber, AdjectiveAdverb, PronounForm |
| F11 | `corrupt`, `merge` and `report` commands; determinism and round-trip checks | ERRANT label agreement (needs F5) |
| C3 | All of C3, including the `cpu` / `cuda` / `rocm` torch extras, `just setup <build>`, device selection and the test markers | — |
| F13 | All of F13: alignment, windows, collator | — |
| F14 | All of F14; trained with `use_doc=False` | Doc-vector training (F19) |
| F15 | All of F15: decoding and metrics | — |
| F16 | Training loop with early stopping and checkpoints, run on the RX 7900 XT | Resuming interrupted runs, hyperparameter sweeps |
| F17 | `evaluate` on the synthetic val split: per-kind F0.5, detection-only F0.5, false alarms per 1,000 words on clean val text, and the SymSpell-only baseline for spelling | Real eval sets (F12), latency |
| PoC | `contcheck tag`: load the checkpoint, window the text, decode, print each flagged span with kind and probability | Replaced for production use by M6's `check()` |

Out of the PoC: F2 and F5 (ERRANT), F12 (real eval data), and M4 to M8.

## Order

```mermaid
flowchart LR
  F1 --> C2
  C2 --> F3 & F4 & F6 & F7 & F8 & F9 & F10
  C2 --> C3
  C3 --> F13 & F14 & F15
  F3 & F4 & F6 & F7 & F8 & F9 & F10 --> F11
  F11 & F13 & F14 & F15 --> F16
  F16 --> F17 & TAG[tag]
```

C3 only edits dependencies, fixtures and markers, so it runs right after C2, next to the
data features. The model features (F13, F14, F15) then build alongside the data pipeline.

| Step | Features (parallel within a step) | Sessions |
| --- | --- | --- |
| 1 | F1 | 1 |
| 2 | C2 | 1 |
| 3 | C3, then F13, F14, F15; and F3, F4, F6, F7, F8, F9, F10 | 5 |
| 4 | F11 | 1 |
| 5 | Generate data, then F16 and a training run | 1, plus compute |
| 6 | F17 and `tag` | 2 |

**Critical path:** F1 → C2 → F7 → F11 → data generation → F16 → training run → F17 /
`tag`.

## Running it

- **Branches:** one branch per milestone slice, stacked so each can be reviewed on its
  own: `feat/poc-m1-taxonomy` (F1) on `main`, `feat/poc-m2-rule-data` (C2, F3, F4,
  F6–F11) on it, and `feat/poc-m3-baseline` (C3, F13–F17, `tag`) on top. Each feature
  lands as one commit on its milestone branch; a change to a lower branch reaches the
  ones above it by merging it up, not by rebasing. Every branch is one pull request in
  the stack, merged bottom up after review. (This replaces the single `feat/poc-tagger`
  branch first planned, so review comes in milestone-sized pieces.)
- **Afterwards:** M1 to M3 continue from `main` with the remaining scope in the right-hand
  column, plus F2, F5 and F12.
- **Data and compute:** about 60,000 Wikipedia paragraphs grouped into documents, two
  corrupted variants each, gives roughly 120,000 training paragraphs. Generation runs on CPU
  (spaCy), roughly 15–30 minutes. Training ELECTRA-small for 3 epochs on the RX 7900 XT is
  estimated at 30–60 minutes. Data stays in `data/`, outside git.
- **Estimate:** about a day to a day and a half of agent sessions with up to five in
  parallel, plus 1–2 hours of compute. This is a planning estimate, not a measurement.

## Done when

- `just check` passes on every branch of the stack.
- `contcheck evaluate` reports per-kind F0.5 for all five PoC kinds on the synthetic val
  split, and the tagger beats the SymSpell-only baseline on Spelling.
- `contcheck tag` flags errors in a handful of hand-written sentences, one per PoC kind,
  and stays quiet on their corrected versions.

Synthetic scores overstate real-world quality (implementation plan, Phase 4). The PoC
proves the pipeline works end to end; M3's evaluation on real human errors judges the
model.

## First run (2026-10-24)

Measured on the dev machine with `configs/gen.yaml` and `configs/train.yaml` as committed.

- **Data:** 60,000 Wikipedia paragraphs from 10,222 articles; `corrupt` wrote 120,000
  records in 3.5 minutes on CPU; after `merge`, 106,291 train, 5,829 val and 6,098 test
  records, with no round-trip failures.
- **Training:** 106,968 windows on the RX 7900 XT with bfloat16 autocast. Val F0.5 peaked
  after epoch 1 (0.829) and fell after epoch 2 (0.821), so early stopping kept epoch 1. The
  whole run, two epochs with validation after each, took 5 minutes 20 seconds.
- **Synthetic val, span-level F0.5:**

  | Kind | Tagger | SymSpell only |
  | --- | --- | --- |
  | Spelling | 0.884 | 0.453 |
  | RealWord | 0.960 | — |
  | SubjectVerbAgreement | 0.881 | — |
  | VerbTense | 0.740 | — |
  | Determiner | 0.644 | — |
  | Overall | 0.829 | 0.377 |

  Detection-only F0.5 was 0.831, and the tagger made 4.4 false alarms per 1,000 clean words
  (SymSpell: 7.7).
- **Hand-written sentences:** `tag` flagged the error in one sentence per kind and in the
  example above (`go`, `buyed`, `there`), and flagged nothing in any corrected version. It
  also made one extra flag, Determiner at 0.53 on a correct word in the VerbTense sentence.

## Second run: realistic misspellings (2026-10-24)

The first model missed plain misspellings such as "libary" and "tomorow". Probing showed
two causes. Real misspellings follow sound and word structure and split into
normal-looking subwords (`occur ##ed`), while the corruptor's keyboard slips split into
junk. And because errors were never on neighbouring words, adjacent typos suppressed each
other. The Spelling corruptor gained realistic patterns, `adjacent_share: 0.1` put some
errors side by side, and `evaluate` gained a check that swaps 96 common real misspellings
into val text (1,132 probes, never next to a labeled error).

Both models on the first run's val set (same text):

| Measure | First run | Second run |
| --- | --- | --- |
| Overall span F0.5 | 0.829 | 0.815 |
| Spelling F0.5 | 0.884 | 0.879 |
| False alarms per 1,000 clean words | 4.42 | 5.06 |
| Real misspellings flagged, in val text | 0.623 | 0.681 |
| Real misspellings flagged, one fixed sentence | 0.34 | 0.32 |
| Typo caught when the next word also has one | 0.917 | 0.936 |
| That neighbouring typo caught | 0.827 | 0.858 |

The SymSpell baseline flags every probe, because each listed misspelling is a non-word
for it.

**Reading:** the patterns helped only a little (+6 points in context, nothing in a
neutral sentence), and they cost some precision. The tagger still misses "libary" and
"tomorow" in "Their going too the libary tomorow and he dont like it." The model has no
dictionary, so it cannot tell a plausible misspelling from a rare correct word.

**Decision:** non-word misspellings belong to the deterministic SymSpell pre-pass
(implementation plan, Phase 5a step 1), not the tagger. The patterns are reverted;
adjacent errors stay. The tagger's job is errors made of real words, the ones a
dictionary cannot see. Next: a held-out confusion check to test whether the model
generalizes beyond the pairs it trained on, then broader real-word error generation
and more registers than Wikipedia. M6 still needs a rule for combining SymSpell flags
with the tagger. Dropping flags the model rates as confident `O` would discard real
misspellings like the ones above, so that rule should be revisited.
The first run's model and data are kept locally as `runs/baseline-v1` and
`data/synthetic-v1`.

## Held-out confusion check (2026-10-24)

Does the tagger judge whether a word fits, or memorize the confusion pairs it trained on?
`held_out_confusion_share: 0.2` withheld a fifth of the ~50 hand-written confusion sets'
pairs from the RealWord corruptor. The data was regenerated and the model retrained (3
epochs, best at epoch 3, val F0.5 0.807). `evaluate` then swapped partners into val text:

| Probe | Flagged | Flagged as RealWord | Probes |
| --- | --- | --- | --- |
| Pairs seen in training | 0.930 | 0.924 | 1,041 |
| Pairs withheld from training | 0.245 | 0.161 | 347 |

**Reading:** the model memorizes pairs. RealWord F0.5 on synthetic val is 0.974, but a
wrong word from a pair it has not seen is caught a quarter of the time. This is the
baseline the broader real-word generators (generated confusables, masked-LM
substitutions, closed-class and inflection kinds) have to move. The held-out number,
not synthetic val, is the one to watch.

## Decisions

| Decision | Assumption |
| --- | --- |
| Clean corpus for the PoC | English Wikipedia (`wikimedia/wikipedia`, CC BY-SA 4.0) through `datasets`; public-domain fiction can be added later for past-tense narration |
| Hardware | Training on the RX 7900 XT with the `rocm` build; everything else on CPU |
