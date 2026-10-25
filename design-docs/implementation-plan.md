# Context-Aware Spell & Grammar Tagger — Implementation Plan

Oct 24, 2026 · Tyler Allen · Exported from the [source doc](https://claude.ai/artifact/9eoXKv4bmaXozatYscWYJ1).

## Overview

Build a local, CPU-fast checker that tags grammar and spelling errors per token, using a fine-tuned small encoder over a sliding window plus a frozen document-context vector. Corrections come later, chosen per error kind.

![Inference architecture: one document vector conditions every token tag](images/architecture.png)

The document vector is computed once and joins every token at the fusion head; SymSpell flags go straight to the error list.

**Pipeline at inference:**

1. SymSpell pre-pass flags non-word misspellings (dictionary lookup, microseconds).
2. Document context is computed once: frozen sentence embedding + tense histogram + a few stats.
3. The tagger runs over 256-token windows (stride 128), fusing the document vector into every token.
4. Overlapping predictions are merged (centre-most window wins) and BIO tags become `(span, kind, op)` errors.
5. A corrector picks a fix per error kind (Phase 6).

**Stack:** Python 3.14 managed by `uv`, with every command run through `just` (see `AGENTS.md`) — PyTorch (CPU, CUDA or ROCm build, chosen at install; see the feature plan's PyTorch builds), Hugging Face `transformers` + `datasets`, `sentence-transformers`, spaCy (`en_core_web_sm`), `errant`, `symspellpy`, `lemminflect`. Local LLM via Ollama or llama.cpp. Deployment via ONNX Runtime, callable from Python or another language (e.g. the `ort` crate in Rust). Dev tooling: `ruff`, `ty` and `pytest`, as in `AGENTS.md`. All of these resolve together for Python 3.14 on Linux x86_64 with current releases (checked 2026-10-24).

**Repo layout:** code follows the `src` layout in `AGENTS.md`, organized by domain. Tests mirror `src/contcheck/`: the tests for `src/contcheck/<path>/<name>.py` are `tests/unit/<path>/test_<name>.py` and, when they cross a real boundary, `tests/integration/<path>/test_<name>.py` (`just layout` enforces this).

```
contcheck/
  src/contcheck/
    taxonomy/         # ErrorKind, Operation, label map (single source of truth)
    analysis/         # spaCy adapter: POS, lemmas, tense histogram, doc stats
    lm/               # MaskedScorer: masked-LM predictions and scores in context
    spelling/         # SymSpell adapter: dictionary lookup and suggestions
    annotation/       # ERRANT adapter mapping edits through taxonomy
    corrupt/          # rule-based corruptors, one module per error kind, plus the applier
    llm_gen/          # local-LLM generation + ERRANT relabeling
    corpus/           # merge, dedupe, document-level splits, distribution report
    features/         # doc vector: embedding, tense histogram, cache
    model/            # ContextTagger, alignment, collator
    training/         # train, evaluate, export_onnx
    infer/            # windowing, span merging, SymSpell pre-pass
    correct/          # per-kind correctors
    cli/              # `contcheck` commands; the only composition root
  tests/
    unit/             # mirrors src/contcheck/, in process with fakes
    integration/      # mirrors src/contcheck/, real files, models or ONNX Runtime
  configs/            # generation and training YAML
  data/               # not committed
    raw/              # clean source paragraphs
    synthetic/        # corrupted JSONL with word-level labels
    eval/             # real human-error sets (BEA-2019, FCE, ...)
  design-docs/
    images/           # Graphviz .dot sources + rendered PNGs (`just diagrams`)
```

## Phase 0 — Taxonomy and label scheme

Define the error set once, in `contcheck.taxonomy`, and generate everything else (model label map, corruptor registry, app enum) from it. It follows ERRANT's categories so real datasets map onto it.

**Kinds (19):** Spelling, RealWord, Capitalization, WordBoundary, Apostrophe, NounNumber, SubjectVerbAgreement, VerbTense, VerbForm, AdjectiveAdverb, PronounForm, Determiner, Preposition, Conjunction, WordOrder, MissingWord, ExtraWord, SentenceBoundary, Punctuation.

**Operations (3):** Missing, Unnecessary, Replacement — kept as a separate field, not baked into the tag.

```python
# src/contcheck/taxonomy/kinds.py
from enum import Enum

class ErrorKind(str, Enum):
    SPELLING = "Spelling"
    REAL_WORD = "RealWord"
    # ... all 19

class Operation(str, Enum):
    MISSING = "M"; UNNECESSARY = "U"; REPLACEMENT = "R"

LABELS = ["O"] + [f"{p}-{k.value}" for k in ErrorKind for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}   # 39 labels
```

**Tagging rules:**

- Labels are word-level BIO: `B-VerbTense` on the first word of an error span, `I-VerbTense` on the rest.
- **Missing** errors have no word to tag. Tag the word *after* the gap (or before, at sentence end) and record `op = M`.
- One kind per span. If edits overlap, keep the one ERRANT ranks first and drop the rest from training.
- Start with the operation as metadata only. Add a second 3-way head for it once kind detection works.

**Mapping from ERRANT:** write `src/contcheck/taxonomy/errant_map.py` translating ERRANT codes (`R:VERB:TENSE`, `M:DET`, `R:SPELL`, `R:ORTH`, …) to `(ErrorKind, Operation)`. Codes with no clean mapping (`OTHER`) are dropped or reviewed by hand.

**Done when:** a unit test round-trips every kind through `LABELS`, and every ERRANT code is either mapped or explicitly listed as dropped.

## Phase 1 — Data

Train on synthetic errors from two generators, and keep real human errors for evaluation only. Every record carries clean text, corrupted text and word-level labels.

**Record format (JSONL):**

```json
{"doc_id": "wiki-00412", "clean": "...", "noisy": "...",
 "words": ["Yesterday", "I", "go", "..."],
 "labels": ["O", "O", "B-VerbTense", "..."],
 "edits": [{"start": 2, "end": 3, "kind": "VerbTense", "op": "R", "orig": "went"}],
 "source": "rule"}
```

### 1a. Clean corpus

1. Collect 50k–200k clean paragraphs, grouped into 2–5 paragraph documents (keep `doc_id` so document context is real).
2. Mix registers: encyclopedic, news, fiction (for past-tense narration), technical, informal.
3. Filter: drop paragraphs the SymSpell pass flags heavily, or that a strong grammar check rewrites. Errors left in "clean" text become false negatives in training.
4. Split by **document**, not sentence, so no document leaks across train/val/test.

### 1b. Rule-based corruptor (`contcheck.corrupt`)

One function per kind, each returning `(new_words, edit)` so labels are exact by construction.

- **Spelling:** keyboard-adjacent substitution, transposition, deletion, doubling, phonetic swaps (ph↔f, ie↔ei). Non-word misspellings are caught by the SymSpell pre-pass, not the tagger: the PoC showed a dictionary-less model cannot tell a plausible misspelling (occured, seperate) from a rare correct word. Spelling corruption stays in the data so the tagger is robust to typos, with a low weight; the tagger's effort goes to errors made of real words.
- **RealWord:** swap a word for a real word it is easily confused with. A few dozen hand-written sets would let the model memorize pairs, so the confusions are generated (`corrupt/confusables.py`), deterministically, when `contcheck corrupt` starts (about 3 s):
  - *hand-written sets* (their/there, then/than, affect/effect, …), kept and weighted highest;
  - *homophones* from the CMU Pronouncing Dictionary (BSD-2-Clause), first pronunciations only, bundled as `corrupt/data/homophones.txt` and rebuilt with `scripts/build_homophones.py`;
  - *look-alikes*: among the 25,000 most frequent SymSpell words, pairs one edit apart, or two when both have 8+ letters (at a quarter of the weight).

  Generated pairs drop forms of one lemma (the inflection corruptors' job), one word extending the other or a shared long stem (invented/inventor), British/American spellings, one-letter words, and words lemminflect does not know (names, abbreviations). A pair's weight is its source weight times `1 + log10` of its rarer word's frequency over the vocabulary's lowest; each word keeps its eight heaviest alternatives, scaled to a total of at most 3. The corruptor draws a confusable word *type* by weight (so repeated function words are not favoured), then an occurrence, then an alternative, and keeps the original's capitalization. `contcheck corrupt` splits the map with `held_out_confusion_share` and passes only the training pairs, weights kept (`ConfusionMap.restrict`), to the corruptor; `evaluate` derives the same split for its held-out check. Expect some label noise: look-alikes where both words fit the context (now/not, fired/hired).
- **RealWord (word choice):** confusion sets alone teach a fixed list of pairs, so a share of RealWord edits (`word_choice_share` in `configs/gen.yaml`) comes from a masked LM instead (`corrupt/word_choice.py`, mixed with the confusion sets by `corrupt/composite.py`). It masks a noun, verb, adjective or adverb, takes the model's top 100 words, and keeps those with the original's Penn tag that are `lemminflect` dictionary words, not a form of the original's lemma, not a stop word, not a neighbour's repeat, and that do not break a preceding *a*/*an*. The point is a contrastive filter: the model must expect the original (P ≥ 0.2), prefer it at least 100× over the candidate, and give the candidate itself at most 0.001, because many proposals are acceptable alternatives (*docks* for *harbour*) and would be false labels. On a manual review of 50 edits from `roberta-large` (the configured model; `distilroberta-base` is the cheaper default), 25 were clear errors, 14 borderline and 11 acceptable alternatives, mostly meaning changes the context allows (*wife* → *cousin*); `distilroberta-base` gave fewer clear errors. Embedding similarity and a neighbour-likelihood check did not separate the two groups. The scorer runs batched on the GPU in half precision behind the `MaskedScorer` protocol (`lm/`) with a per-run cache; roberta-large adds about 9 ms per paragraph (about 9 minutes on 60k paragraphs × 2 variants), so no precompute step is needed. GPU float differences can flip borderline candidates between machines, so byte-identical output holds per machine and model, not across them.
- **VerbTense / VerbForm / SubjectVerbAgreement:** spaCy POS + `lemminflect` to re-inflect one verb (VBD→VBZ, VBZ→VBP, VBN→VBD). For tense, pick verbs in documents whose tense histogram is >80% one tense, so the error is detectable.
- **Determiner / Preposition:** delete, insert, or swap from a closed set, only where the result is wrong in any context. Swapping *a* for *the*, inserting *the*, or dropping *the* usually leaves correct English, so Determiner only drops *a*/*an* before a singular noun, puts *a*/*an* before a plural, and uses the wrong *a*/*an* form.
- **NounNumber, Capitalization, Apostrophe, WordBoundary, Punctuation, ExtraWord, MissingWord, WordOrder:** simple token operations.

Sampling: 0–3 errors per paragraph, \~15% of paragraphs left clean, kind frequencies set by a config file so you can rebalance.

### 1c. LLM-generated errors (`contcheck.llm_gen`)

For naturalistic, multi-word errors the rules can't produce.

1. Run a local 7–14B instruct model (Qwen, Gemma or Llama class) via Ollama.
2. Prompt: rewrite the paragraph as a specific writer profile would (ESL intermediate, fast typist, child), keeping meaning and changing only grammar and spelling.
3. **Ignore the LLM's own description of its changes.** Run ERRANT on `(noisy, clean)` to extract and classify edits, then map through `errant_map.py`.
4. Reject outputs with too many edits, low word overlap with the source, or any unmapped edit type.

Target mix to start: \~70% rule-based, \~30% LLM.

### 1d. Real evaluation data

BEA-2019 (W&I+LOCNESS dev), FCE test, CoNLL-2014, JFLEG, GitHub Typo Corpus, Birkbeck spelling corpus. Convert each through ERRANT into the same JSONL format. Check licenses before any commercial use.

### 1e. Corruptor implementation

Each corruptor is registered by kind and returns an `Edit` (or `None` when the paragraph has no valid site). A shared applier turns edits into noisy words and BIO labels, so labeling logic lives in one place.

```python
# src/contcheck/corrupt/base.py
@dataclass
class Edit:
    start: int; end: int            # word indices in the CLEAN text
    kind: ErrorKind; op: Operation
    orig: list[str]; new: list[str]

CORRUPTORS: dict[ErrorKind, Callable] = {}

def register(kind):
    def deco(fn):
        CORRUPTORS[kind] = fn
        return fn
    return deco
```

```python
# src/contcheck/corrupt/spelling.py
@register(ErrorKind.SPELLING)
def spelling(ctx, rng):
    sites = [i for i, w in enumerate(ctx.words) if w.isalpha() and len(w) > 3]
    if not sites: return None
    i = rng.choice(sites); w = ctx.words[i]
    new = rng.choice([keyboard_sub, transpose, drop_char, double_char, phonetic])(w, rng)
    if new == w or new.lower() in DICTIONARY:   # must be a non-word, else it's RealWord
        return None
    return Edit(i, i + 1, ErrorKind.SPELLING, Operation.REPLACEMENT, [w], [new])
```

```python
# src/contcheck/corrupt/verbs.py
@register(ErrorKind.VERB_TENSE)
def verb_tense(ctx, rng):
    if ctx.tense_hist["past"] < 0.8: return None   # only where the error is detectable
    sites = [i for i, t in enumerate(ctx.tags) if t == "VBD"]
    if not sites: return None
    i = rng.choice(sites)
    target = "VBZ" if ctx.is_third_singular(i) else "VBP"   # keep agreement correct
    new = getInflection(ctx.lemmas[i], tag=target)
    if not new or new[0] == ctx.words[i]: return None
    return Edit(i, i + 1, ErrorKind.VERB_TENSE, Operation.REPLACEMENT, [ctx.words[i]], [new[0]])
```

**Applier rules (`src/contcheck/corrupt/apply.py`):**

- Sample 0–3 kinds per paragraph from the config weights; retry another kind when a corruptor returns `None`.
- Reject edits that overlap. Keep the rest apart by an untouched word, except in a small share of paragraphs (`adjacent_share` in the config) whose errors sit on neighbouring words: real writers make those, and a model that never sees them learns that an error's neighbours are correct. Each edit still gets its own label span, and a Missing edit always keeps untouched neighbours, since its label goes on one of them.
- Apply edits right to left so earlier indices stay valid, then compute labels on the noisy word list.
- **Missing** edits (a deleted word) put `B-<kind>` on the word that now follows the gap.
- Match casing of the original word, and re-attach punctuation tokens unchanged.
- Produce 2–3 corrupted variants per clean paragraph with different seeds; it multiplies data cheaply.

### 1f. LLM generation and relabeling

**Prompt (one call per paragraph and profile):**

```text
Rewrite the paragraph below as it might be written by {profile}.
Introduce {n} grammar or spelling mistakes typical of that writer.
Keep the meaning, sentence order and vocabulary the same otherwise.
Do not change style, tone or word choice beyond the mistakes.
Return only the rewritten paragraph, nothing else.

Paragraph:
{clean}
```

Profiles: "an intermediate English learner whose first language is {L1}" (rotate Spanish, Mandarin, Arabic, Hindi, Russian…), "a native speaker typing fast on a phone", "a 12-year-old student". Temperature \~0.8; `n` sampled 1–4.

```python
# src/contcheck/llm_gen/relabel.py
import errant
annotator = errant.load("en")

def relabel(noisy: str, clean: str):
    orig = annotator.parse(noisy)          # spans are in the NOISY text
    cor = annotator.parse(clean)
    edits = annotator.annotate(orig, cor)  # noisy -> clean = what to fix
    out = []
    for e in edits:
        mapped = ERRANT_MAP.get(e.type)    # e.g. "R:VERB:TENSE" -> (VERB_TENSE, R)
        if mapped is None:
            return None                    # unmapped edit: drop the whole sample
        out.append((e.o_start, e.o_end, *mapped))
    return out
```

**Filters:** 1–6 edits; noisy/clean length ratio 0.9–1.1; character similarity ≥ 0.85; no unmapped edit types; reject outputs that start with chatter ("Here is…").

**Throughput:** budget this before committing. A 7–14B model on CPU manages roughly tens of tokens per second, so 150k paragraphs can take weeks. Use a GPU if you have one, run a llama.cpp server with parallel slots, or start with \~20k LLM paragraphs and grow only if the ablation shows they help.

### 1g. Config and run commands

```yaml
# configs/gen.yaml
seed: 13
variants_per_paragraph: 2
errors_per_paragraph: {0: 0.15, 1: 0.40, 2: 0.30, 3: 0.15}
kind_weights:            # relative; rebalance after the first eval
  Spelling: 3
  RealWord: 2
  VerbTense: 2
  SubjectVerbAgreement: 2
  Determiner: 2
  Preposition: 1.5
  Punctuation: 1.5
  default: 1
llm:
  model: <your local instruct model>
  temperature: 0.8
  paragraphs: 20000
```

```bash
just cli corrupt   --config configs/gen.yaml --in data/raw --out data/synthetic/rule.jsonl
just cli generate  --config configs/gen.yaml --in data/raw --out data/synthetic/llm.jsonl
just cli merge     --out data/synthetic/train.jsonl   # dedupe, shuffle, keep doc splits
just cli report    data/synthetic/train.jsonl         # per-kind counts, errors/paragraph
```

These are subcommands of the `contcheck` CLI (`src/contcheck/cli/`); add a `just` recipe for any that become routine.

### 1h. Data quality checks

- **Determinism:** same seed and config produce byte-identical output.
- **Round trip:** reverting each record's edits on the noisy text reproduces the clean text exactly.
- **Label agreement:** run ERRANT on rule-based pairs too; its kind should match the corruptor's kind on ≥ 90% of edits. Low agreement for one kind means that corruptor is buggy.
- **Distribution report:** per-kind counts and errors per paragraph, compared against the config targets.
- **Manual review:** 50–100 samples per kind, read before the first training run.

**Done when:** 500k+ labeled training paragraphs, a per-kind count report, and spot checks of 100 random records show correct labels.

## Phase 2 — Document context vector

The document vector is \~392 numbers: a frozen 384-dim embedding plus explicit tense features, computed once per document and cached. The tense histogram carries the tense signal; the embedding mostly carries topic and register.

| Part | Dims | Source | Why |
| --- | --- | --- | --- |
| Sentence embedding | 384 | `all-MiniLM-L6-v2` or `bge-small-en-v1.5`, mean of paragraph embeddings | Topic and register, for real-word errors |
| Tense histogram | 4 | spaCy tags over finite verbs: share of VBD, VBP+VBZ, MD, other | Dominant tense of the document |
| Doc stats | \~4 | sentence count, mean sentence length, share of 1st-person pronouns, share of quoted text | Separates dialogue and narration |

```python
# src/contcheck/features/doc_vector.py
def doc_vector(text: str) -> np.ndarray:
    emb = embedder.encode(paragraphs(text)).mean(axis=0)   # (384,)
    doc = nlp(text)
    verbs = [t.tag_ for t in doc if t.tag_ in FINITE_TAGS]
    hist = tense_histogram(verbs)                          # (4,) sums to 1
    stats = doc_stats(doc)                                 # (4,)
    return np.concatenate([emb, hist, stats]).astype(np.float32)
```

**Important:** compute the training vector from the **noisy** text, not the clean one. At inference you only have noisy text, and a single tense error barely moves the histogram, so the signal survives.

**Caching:** store vectors in a `.npy` keyed by `doc_id`; recompute only when the feature code version changes.

**Done when:** vectors exist for every train/val/test document, and the tense histogram on a hand-picked past-tense story is >0.8 VBD.

## Phase 3 — Tagger model and training

Fine-tune ELECTRA-small with a fusion head; train end to end with the document vector as a fixed extra input. Get a no-context baseline working first, then add the document vector.

### 3a. Tokenization and alignment (`src/contcheck/model/align.py`)

- Tokenize pre-split words with `is_split_into_words=True`.
- Label the first subword of each word; set continuation subwords and special tokens to `-100`.
- Training windows: 256 tokens, stride 128, built with `return_overflowing_tokens=True`. Each window carries its document's `doc_id` so the collator attaches the right vector.

### 3b. Model (`src/contcheck/model/tagger.py`)

```python
class ContextTagger(nn.Module):
    def __init__(self, enc="google/electra-small-discriminator",
                 doc_dim=392, num_labels=39, fuse_dim=128, use_doc=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(enc)
        h = self.encoder.config.hidden_size
        self.use_doc = use_doc
        self.doc_proj = nn.Sequential(nn.LayerNorm(doc_dim),
                                      nn.Linear(doc_dim, fuse_dim), nn.GELU())
        in_dim = h + (fuse_dim if use_doc else 0)
        self.head = nn.Sequential(nn.Linear(in_dim, h), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(h, num_labels))

    def forward(self, input_ids, attention_mask, doc_vec, labels=None):
        tok = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        if self.use_doc:
            g = self.doc_proj(doc_vec).unsqueeze(1).expand(-1, tok.size(1), -1)
            tok = torch.cat([tok, g], dim=-1)
        logits = self.head(tok)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                               weight=CLASS_WEIGHTS, ignore_index=-100) if labels is not None else None
        return loss, logits
```

The `LayerNorm` on the document vector matters: embedding values and histogram shares are on different scales.

### 3c. Training settings (starting point)

| Setting | Value |
| --- | --- |
| Encoder learning rate | 3e-5 |
| Head + doc projection learning rate | 1e-3 |
| Batch size | 32 windows |
| Epochs | 3–5, early stop on val F0.5 |
| Warmup | 6% linear, then linear decay |
| Class weight for `O` | 0.2 (others 1.0) |
| Max grad norm | 1.0 |

Use Hugging Face `Trainer` with a custom collator, or a plain PyTorch loop; both are fine at this size. A single consumer GPU trains this in hours; CPU-only training works but is slow, so consider renting a GPU for this phase only.

### 3d. Order of experiments

1. `use_doc=False` baseline on rule-based data only.
2. Add LLM data.
3. `use_doc=True`.
4. Try DeBERTa-v3-small as the encoder if accuracy is short and the latency budget allows.

**Done when:** the baseline beats a SymSpell-only checker on spelling, and per-kind F0.5 is reported for every kind on the synthetic val set.

## Phase 4 — Evaluation

Judge the model on real human errors with span-level F0.5, broken down by kind. Synthetic val scores are for debugging only; they will look much better than real-world performance.

**Metrics (`src/contcheck/training/evaluate.py`):**

- **Span-level precision, recall, F0.5** per kind and overall. A hit = same span and same kind. F0.5 weights precision twice as much as recall, because false alarms annoy users more than misses.
- **Detection-only F0.5:** span correct, kind ignored. Shows whether misses are detection or classification problems.
- **Confusion matrix** across kinds (e.g. VerbTense vs VerbForm vs SubjectVerbAgreement).
- **False-alarm rate on clean text:** flags per 1,000 words on held-out clean documents. Track this every run.
- **Latency:** ms per 256-token window on your target CPU, p50 and p95.

**Ablations to run:**

1. With vs without the document vector, overall and on VerbTense alone. If they match, drop the vector.
2. Embedding only vs tense histogram only vs both.
3. Rule-only vs rule + LLM training data, scored on real errors.

**Targeted tense test set:** build \~300 documents by hand or from fiction where exactly one verb breaks a consistent tense, plus \~300 that legitimately mix tenses (dialogue, flashbacks). The model should flag the first set and stay quiet on the second.

**Done when:** a single command prints the per-kind table for every eval set and the ablation results land in a results log.

## Phase 5 — Inference and CPU deployment

Ship three artifacts — the tagger as int8 ONNX, the embedder as ONNX, and the SymSpell dictionary — behind one `check(text) -> list[Error]` function.

### 5a. Inference flow (`contcheck.infer`)

1. **Pre-pass:** SymSpell marks words not in the dictionary (skip numbers, URLs, code, all-caps, user dictionary words).
2. **Doc vector:** compute once per document; recompute on edit only if the text changed by more than a paragraph.
3. **Windows:** tokenize the whole document; windows of 256 with stride 128. Skip windowing if the document fits in one window.
4. **Merge:** for each token, keep logits from the window where it sits closest to the centre.
5. **Decode:** take each word's first-subword prediction, then group `B-X I-X…` into spans. Repair invalid sequences (`I-X` with no `B-X` becomes `B-X`).
6. **Threshold:** flag a span only if its probability exceeds a per-kind threshold tuned on val for F0.5.
7. **Combine with SymSpell:** a SymSpell flag the model rates as `O` with high confidence is usually a name or jargon; downgrade or drop it.

### 5b. Export (`src/contcheck/training/export_onnx.py`)

- `torch.onnx.export` with inputs `input_ids`, `attention_mask`, `doc_vec`; dynamic axes for batch and sequence length.
- Dynamic int8 quantization with `onnxruntime.quantization.quantize_dynamic`.
- Verify: quantized vs fp32 logits agree, and per-kind F0.5 drops by no more than \~1 point.
- Export the embedder the same way, or swap it for a Model2Vec static model if embedding latency matters.

### 5c. Latency budget (targets to measure, not guarantees)

| Step | Target on a modern laptop CPU |
| --- | --- |
| SymSpell pre-pass | < 1 ms per paragraph |
| spaCy tense histogram | < 20 ms per document |
| Embedder | < 50 ms per document |
| Tagger, int8 | < 15 ms per 256-token window |

For interactive use, re-tag only windows that overlap the edited region, and cache the doc vector.

**Done when:** `check()` runs end to end from ONNX files with no PyTorch installed, meets the latency targets, and matches PyTorch outputs on the test set.

## Phase 6 — Corrections

Route each detected error to a corrector by kind: rules and candidate scoring for most kinds, a small local LLM only for the structural ones. Every corrector returns ranked suggestions, never silently edits.

| Kind | Corrector | Candidates from | Ranked by |
| --- | --- | --- | --- |
| Spelling | SymSpell | edit distance ≤ 2 | masked-LM score in context + word frequency |
| RealWord, Determiner, Preposition, Conjunction | Confusion sets | closed sets per word | masked-LM score |
| VerbTense, VerbForm, SubjectVerbAgreement | `lemminflect` re-inflection | all inflections of the lemma | masked-LM score; for tense, prefer the doc's dominant tense |
| NounNumber, AdjectiveAdverb, PronounForm | Inflection / form tables | alternate forms | masked-LM score |
| Capitalization, Apostrophe, WordBoundary, Punctuation, ExtraWord | Rules | deterministic fix | — |
| WordOrder, MissingWord, SentenceBoundary | Local LLM (constrained) | LLM rewrite of the sentence | ERRANT check that only the flagged span changed |

**Masked-LM scoring:** mask the flagged word, score each candidate with a small masked LM (e.g. DistilRoBERTa) in the full sentence. ELECTRA's generator half also works and is tiny. The `MaskedScorer` protocol in `lm/` already does this for the word-choice corruptor (`MaskedSlot.candidates` with `top_k=0` scores only the candidates); correctors reuse it, adding an ONNX adapter so `infer` stays free of `torch`. It scores only words that are one model token, so multi-token candidates need an extension (for example pseudo-log-likelihood over their pieces).

**Optional reranker:** once enough candidates are logged, train LightGBM on features per candidate (edit distance, frequency, LM score, tagger confidence, same-lemma flag). This is where tree models fit well.

**Later upgrade:** GECToR-style edit tags (`$APPEND_the`, `$TRANSFORM_VERB_VBD_VBZ`) let the tagger output common fixes directly, skipping a separate corrector for those kinds.

**Done when:** top-1 suggestion accuracy is reported per kind on real eval data, and LLM-routed corrections are rejected whenever they change text outside the flagged span.

## Milestones and open questions

[feature-plan.md](feature-plan.md) splits these milestones into features, waves and dependencies, and tracks the open questions below as decisions with working assumptions.

The first useful milestone is M3: a no-context tagger on rule-based data, scored on real errors. Everything after it is an improvement you can measure against that baseline.

- [ ] **M1 — Taxonomy:** `ErrorKind`, `Operation`, `LABELS`, ERRANT mapping, round-trip tests.
- [ ] **M2 — Rule-based data:** clean corpus, document-level split, corruptors for Spelling, RealWord, VerbTense, SubjectVerbAgreement, Determiner first; the rest after.
- [ ] **M3 — Baseline tagger:** ELECTRA-small, `use_doc=False`, evaluated on BEA-2019 dev and the false-alarm set.
- [ ] **M4 — Document context:** doc vectors, `use_doc=True`, tense ablation on the targeted tense set.
- [ ] **M5 — LLM data:** Ollama generation, ERRANT relabeling, filters; retrain and compare.
- [ ] **M6 — CPU inference:** ONNX export, int8, windowing and merge, SymSpell pre-pass, latency report.
- [ ] **M7 — Corrections:** SymSpell + confusion sets + inflection with masked-LM ranking.
- [ ] **M8 — Structural fixes:** constrained local-LLM corrector for WordOrder, MissingWord, SentenceBoundary.

**Open questions:**

- Which language hosts the app — Python, or Rust/other calling ONNX Runtime? This decides whether spaCy can run at inference or the tense histogram needs a lighter POS tagger.
- English only, or other languages later? Multilingual changes the encoder choice (e.g. a multilingual MiniLM or mDeBERTa).
- Should the operation (M/U/R) get its own head in v1, or stay metadata until detection is solid?
- What latency counts as "fast": as-you-type (< 50 ms per edit) or on-demand checks (< 1 s per document)?
