# Latent Pager Memory: Externalizing Latent States Across Recursive Reads

## Autonomous Agent Handoff Documentation

**Version:** 1.0  
**Date:** 2026-02-24  
**Target Agent:** Codex 5.3 Extended Autonomous Coding Agent  
**Base Model:** `Qwen/Qwen3-1.7B` (HuggingFace)  
**License:** Apache 2.0

---

## 1. Executive Summary

This experiment implements and evaluates **Latent Pager Memory** — a system that stores compressed latent states (not text summaries) produced by a transformer's hidden layers as first-class objects in a programmatic environment. Instead of the conventional Recursive Language Model (RLM) approach of passing textual intermediate buffers between recursive reads of a large document, we store continuous-space "pages" of latent representations and later aggregate them for final answer decoding.

The core comparison is:

| Condition | Intermediate Representation | Aggregation |
|---|---|---|
| **Baseline (Text Buffer)** | Text summaries from each chunk | Concatenate summaries → fee LM |
| **Treatment (Latent Pager)** | Compressed hidden-state vectors per chunk | Neural aggregator → soft-prompt injection → LM decode |

---

## 2. Theoretical Motivation

### 2.1 From Two Source Papers

**Paper A — "Scaling Up Test-Time Compute with Latent Reasoning" (Recurrent Depth):**
The key insight is that meaningful reasoning happens in continuous latent space — information that may not be easily or faithfully verbalized into tokens. A depth-recurrent transformer iterates a shared core block in latent space before decoding. This proves that latent states carry reasoning-relevant information beyond what text can capture.

**Paper B — "Recursive Language Models" (RLMs):**
RLMs decompose massive inputs by recursively reading chunks and storing intermediate results (text buffers) in a REPL-like environment. This solves context-window limits and context rot, but intermediate buffers are lossy text summaries — information is destroyed at each summarization step.

### 2.2 The Synthesis — Laory

Treat latent vectors like "pages" in an out-of-core algorithm:

```
load chunk_i → forward pass → extract hidden states → compress → save latent page_i
...repeat for all chunks...
load all latent pages → aggregate → inject as soft prompt → decode final answer
```

**Why this should outperform text buffers:**
1. Text summaries are lossy compressions forced through the vocabulary bottleneck
2. Hidden states preserve distributional nuance, implicit relationships, and uncertainty signals
3. Aggregation in continuous space can perform weighted combination impossible with text concatenation
4. Reduces hallucination risk from multi-hop text-summary chains (each summary is a potential hallucination source)

---

## 3. Model Specification

### 3.1 Base Model

```
Model: Qwen/Qwen3-1.7B
Source: https://huggingface.co/Qwen/Qwen3-1.7B
Architecture: Qwen3ForCausalLM (dense transformer, decoder-only)
Framework: HuggingFace Transformers >= 4.51.0
```

**Expected architecture parameters** (verify from `c runtime):

| Parameter | Expected Value |
|---|---|
| `hidden_size` | 2048 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 |
| `intermediate_size` | ~6144 |
| `vocab_size` | 151936 |
| `max_position_embeddings` | 32768 |
| `hidden_act` | silu |
| `rms_norm_eps` | 1e-6 |
| `torch_dtype` | bfloat16 |

**IMPORTANT:** On first run, load the model and print `model.config` to verify all values. Use the actual `hidden_size` from `config.json` throughout (referred to as `D_model` below).

### 3.2 Compute Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 8 GB | 16+ GB (A100/L4/RTX 4090) |
| System RAM | 32 GB | 64 GB |
| Disk | 20 GB | 50 GB |
| CUDA | 11.8+ | 12.1+ |

Use `bfloat16` precision for all model operations. Enable `torch.compile` where stable. Use gradient checkpointing for the aggregator training phase.

---

## 4. Architecture Design

### 4.1 System Components

```
┌──────────────────────────┐
│                    LATENT PAGER SYSTEM                    │
│                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐   │
│  │  Chunker  │───▶│  Qwen3-1.7B  │───▶│ Page Compressor│  │
│  │          │    │  (frozen)    │    │  (trainable)  │   │
│  └──────────┘    └──────────────┘    └───────┬───────┘   │
│                                              │           │
│                                    ┌─────────▼─────────┐ │
│                                    │  Latent Page Store │ │
│                                    │  (in-memory dict)  │ │
│                                    └────                                │
│         ▼                                                │
│    Final Answer                                          │
└──────────────────────────────────────────────────────────┘
```

### 4.2 Component Specifications

#### 4.2.1 Document Chunker

```python
class DocumentChunker:
    """
    Splits input document into overlapping chunks that fit within
    the model's effective context window.
    """
    def __init__(
        self,
        tokenizer,
        chunk_size: int = 1024,       # tokens per chunk
        overlap: int = 128,            # overlap between consecutive chunks
        max_chunks: int = 64           # maximum chunks per document
    ):
        pass

    def chunk(self, document: str) -> list[dict]:
        """
        Returns list of:
        {
            "chunk_id": int,
            "text": str,
            "token_ids"unk_size=1024` keeps each chunk well within the 32K context, leaving room for the question prompt
- Overlap prevents information loss at chunk boundaries
- Truncate or sample if document produces > `max_chunks` chunks

#### 4.2.2 Latent State Extractor

```python
def extract_latent_states(
    model,                          # frozen Qwen3-1.7B
    input_ids: Tensor,              # [1, seq_len]
    attention_mask: Tensor,
    extraction_layers: list[int],   # which layers to extract from
    pooling: str = "mean"           # "mean" | "last_token" | "attention_weighted"
) -> Tensor:
    """
    Forward pass with output_hidden_states=True.
    Extract hidden states from specified layers.
    Pool across sequence dimension.

    Returns: [1, num_extraction_layers, D_model]
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
    # outputs.hidden_states is tuple of (num_layers+1) tensors, each [batch, seq_len, D_model]
    selected = torch.stack([outputs.hidden_states[l] for l in extraction_layers])  # [num_layers_selected, batch, seq, D_model]

    if pooling == "mean":
        mask = attention_mask.unsqueeze(0).unsqueeze(-1)  # [1, 1, seq, 1]
        pooled = (selected * mask).sum(dim=2) / mask.sum(dim=2)  # [num_layers_selected, batch, D_model]
    elif pooling == "last_token":
        last_idx = attention_mask.sum(dim=-1) - 1
        pooled = selected[:, :, last_idx, :]
    # else: attention_weighted (future extension)

    return pooled.squeeze(1)  # [num_layers_selected, D_model]
```

**Default extraction layers:** `[7, 14, 21, 27]` (quartile layers for a 28-layer model; adapt if actual `num_hidden_layers` differs). This captures progressively abstract representations.

#### 4.2.3 Page Compressor (Trainable)

```python
class PageCompressor(nn.Module):
    """
    Compresses multi-layer hidden states into a single fixed-size latent page vector.

    Input:  [num_extraction_layers, D_model]  (e.g., [4, 2048])
    Output: [D_page]                          (e.g., [512])
    """
    def __init__(self, num_layers: int, d_model: int, d_page: int = 512):
        super().__init__()
        self.flatten_dim = num_layers * d_model
        self.net = nn.Sequential(
            nn.Linear(self.flatten_dim, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_page),
            nn.LayerNorm(d_page)
        )

    def forward(self, multi_layer_states: Tensor) -> Tensor:
        flat = multi_layer_states.reshape(-1, self.flatten_dim)
        return self.net(flat)  # [batch, d_page]
```

**Key design choice:** `d_page = 512` (1/4 of `D_model`) provides significant compression while retaining representational capacity. This is a tunable hyperparameter.

#### 4.2.4 Latent Page Store

```python
class LatentPageStore:
    """
    In-memory store for compressed latent pages.
    Analogous to a virtual memory paging system.
    """
    def __init__(self):
        self.pages: dict[int, dict] = {}  # chunk_id -> page_data

    def write(self, chunk_id: int, page_vector: Tensor, metadata: dict):
        self.pages[chunk_id] = {
            "vector": page_vector.detach().cpu(),
            "metadata": metadata  # chunk text boundaries, extraction timestamp, etc.
        }

    def read_all(self) -> Tensor:
        """Returns all page vectors stacked: [num_pages, d_page]"""
        ordered = sorted(self.pages.keys())
        return torch.stack([self.pages[k]["vector"] for k in ordered])

    def read_by_ids(self, chunk_ids: list[int]) -> Tensor:
        return torch.stack([self.pages[cid]["vector"] for cid in chunk_ids])

    def num_pages(self) -> int:
        return len(self.pages)

    def clear(self):
        self.pages = {}
```

#### 4.2.5 Page Aggregator (Trainable)

```python
class PageAggregator(nn.Module):
    """
    Aggregates multiple latent pages into a fixed number of soft-prompt embeddings.

    Input:  [num_pages, d_page]
    Output: [num_soft_tokens, D_model]  — ready for injection into the LM
    """
    def __init__(
        self,
        d_page: int = 512,
        d_model: int = 2048,
        num_soft_tokens: int = 32,
        num_heads: int = 8,
        num_agg_layers: int = 2
    ):
        super().__init__()
        self.num_soft_tokens = num_soft_tokens

        # Project pages up to model dimension
        self.page_proj = nn.Linear(d_page, d_model)

        # Learnable query tokens that attend to pages
        self.query_tokens = nn.Parameter(torch.randn(num_soft_tokens, d_model) * 0.02)

        # Cross-attention layers: queries attend to pages
        agg_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )
        self.cross_attn = nn.TransformerDecoder(agg_layer, num_layers=num_agg_layers)

        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, ge_vectors: Tensor) -> Tensor:
        """
        page_vectors: [num_pages, d_page]
        returns: [num_soft_tokens, D_model]
        """
        # Project pages: [num_pages, D_model]
        memory = self.page_proj(page_vectors).unsqueeze(0)  # [1, num_pages, D_model]

        # Query tokens: [1, num_soft_tokens, D_model]
        queries = self.query_tokens.unsqueeze(0)

        # Cross-attend
        out = self.cross_attn(queries, memory)  # [1, num_soft_tokens, D_model]

        return self.output_norm(out.squeeze(0))  # [num_soft_tokens, D_model]
```

**Design rationale:** This is a Perceiver-style bottleneck. A fixed set of learned query tokens attends over a variable number of pages, producing a fixed-size soft prompt regardless of document length.

#### 4.2.6 Soft-Prompt Injector

```python
def inject_soft_prompt_and_generate(
    model,
    tokenizer,
    soft_prompt_embeds: Tensor,     # [num_soft_tokens, D_model]
    question_text: str,
    max_new_tokens: int = 256
) -> str:
    """
    Prepends soft-prompt embeddings to the question's token embeddings,
    then generates via the frozen LM.
    """
    question_ids = tokenizer(question_text, return_tensors="pt").input_ids.to(model.device)
    question_embeds = model.model.embed_tokens(question_ids)  # [1, q_len, D_model]

    soft_prompt = soft_prompt_embeds.unsqueeze(0).to(model.device)  # [1, num_soft, D_model]

    combined_embeds = torch.cat([soft_prompt, question_embeds], dim=1)  # [1, num_soft + q_len, D_model]

    # Create attention mask
    attn_mask = torch.ones(1, combined_embeds.shape[1], device=model.device)

    outputs = model.generate(
        inputs_embeds=combined_embeds,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## 5. Baseline System (Text Buffer RLM)

The baseline mimics the RLM text-buffer approach for fair comparison:

```python
class TextBufferBaseline:
    """
    For each chunk:
      1. Feed chunk + task prompt to LM
      2. Generate a text summary/extraction
      3. Store text in buffer
    After all chunks:
      4. Concatenate all text buffers (truncate if needed)
      5. Feed concatenated buffer + question to LM
      6. Generate final answer
    """

    def __init__(self, model, tokenizer, chunk_size=1024, max_buffer_tokens=4096):
        self.model = model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.max_buffer_tokens = max_buffer_tokens

    def process_chunk(self, chunk_text: str, task_prompt: str) -> str:
        prompt = f"{task_prompt}\n\nDocument section:\n{chunk_text}\n\nExtracted information:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def aggregate_and_answer(self, buffers: list[str], question: str) -> str:
        combined = "\n---\n".join(buffers)
        # Truncate to max_buffer_tokens if needed
        combined_ids = self.tokenizer(combined, truncation=True, max_length=self.max_buffer_tokens)
        combined_text = self.tokenizer.decode(combined_ids.input_ids, skip_special_tokens=True)

        prompt = f"Based on the following extracted information:\n{combined_text}\n\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
```

---

## 6. Dataset and Evaluation

### 6.1 Primary Dataset: Synthetic Long-Document QA (OOLONG-style)

Since the original OOLONG benchmark may not be publicly released or easily accessible, construct a synthetic equivalent:

#### 6.1.1 Dataset Construction

```python
"""
Synthetic OOLONG-style dataset construction.
Each sample consists of:
  - A long document (8K-64K tokens) composed of multiple passages
  - A question that requires information from 1-4 specific passages
  - A gold-standard answer
  - Metadata: which passages are evidence, distractor count, etc.
"""

TASK_TYPES = [
    "single_fact_extraction",      # answer in one passage
    "multi_hop_reasoning",         # chain across 2-3 passages
    "aggregation",                 # combine info from 3+ passages
    "contradiction_detection",     # find conflicting claims
    "temporal_ordering"            # order events from different passages
]
```

**Construction pipeline:**

1. **Source passages:** Use Wikipedia paragraphs, arXiv abstracts, or news articles (public domain / CC-licensed)
2. **Document assembly:** For each sample, select N evidence passages (1-4) and M distractor passages (8-30). Shuffle ordering. Concatenate to form the "long document"
3. **Question generation:** Use Qwen3-1.7B itself or a larger model to generate questions that require the evidence passages
4. **Answer generation:** Generate gold answers from evidence passages only
5. **Validation:** Verify that the question is not answerable from distractors alone

**Target dataset size:**

| Split | Samples | Document Length (tokens) |
|---|---|---|
| Train | 2000 | 8K – 32K |
| Validation | 300 | 8K – 32K |
| Test | 500 | 8K – 64K |

#### 6.1.2 Alternative: Use Existing Benchmarks

If construction is infeasible, use these public alternatives:

1. **LongBench** (THUDM): Multi-task long-context benchmark  
   - HuggingFace: `THUDM/LongBench`
   - Relevant subsets: `narrativeqa`, `qasper`, `multifieldqa_en`, `musique`

2. **SCROLLS** (Tau et al.): Long-document understanding tasks  
   - HuggingFace: `tau/scrolls`
   - Relevant subsets: `qasper`, `quality`, `narrative_qa`

3. **QuALITY** (Pang et al.): Multiple-choice long-document QA  
   - Long articles with comprehension questions

**Priority order:** Synthetic OOLONG-style > LongBench > SCROLLS > QuALITY

### 6.2 Evaluation s

#### 6.2.1 Primary Metrics (Success Criteria)

| Metric | Definition | Target |
|---|---|---|
| **Task Accuracy** | Exact match or F1 on answer extraction | Latent > Text baseline by ≥ 3 points |
| **ROUGE-L** | Longest common subsequence overlap with gold answer | Latent ≥ Text baseline |
| **Hallucination Rate** | % of generated claims not supported by source document | Latent < Text baseline by ≥ 10% relative |
| **Global Consistency** | For multi-query over same doc: consistency of answers | Latent > Text baseline |

#### 6.2.2 Secondary Metrics (Diagnostic)

| Metric | Definition | Purpose |
|---|---|---|
| **Information Retention** | Probe test: can the aggregated representation recover specific facts? | Measures compression quality |
| **Latent Reconstruction Loss** | MSE between compressed and original hidden states (via decoder probe) | Validates compressor isn't destroying info |
| **Compute Cost** | Total FLOPs / wall-clock for full pipeline | Must be within 1.5x of text baseline |
| **MFootprint** | Peak GPU memory during inference | Track scalability |
| **Pages-vs-Accuracy Curve** | Accuracy as function of number of chunks/pages | Shows scaling behavior |

#### 6.2.3 Hallucination Detection Method

```python
def compute_hallucination_rate(generated_answer: str, source_document: str, gold_answer: str) -> float:
    """
    Decompose generated answer into atomic claims.
    For each claim, check if it is:
      (a) supported by the source document → not hallucinated
      (b) supported by the gold answer → not hallucinated
      (c) neither → hallucinated

    Implementation options (in order of preference):
      1. Use an NLI model (e.g., `cross-encoder/nli-deberta-v3-base`) to check
         entailment between source doc and each claim
      2. Use Qwen3-1.7B itself as a judge with a verification prompt
      3. N-gram overlap heuristic (least reliable)

    Returns: fraction of claims that are hallucinated
    """
    pass
```

#### 6.2.4 Global Consistency Check

```python
def global_consistency(answers: list[str], document: str) -> float:
    """
    Given multiple questions about the same document, check that
    answers are mutually consistent.

    Method: For each pair of answers, check for contradictions
    using NLI or self-consistency prompting.

    Returns: fraction of answer pairs that are consistent
    """
    pass
```

---

## 7. Experiment Protocol

### 7.1 Phase 1: Infrastructure Setup

**Steps:**

1. Install dependencies:
   ```bash
   pip install torch>=2.1 transformers>=4.51 datasets accelerate bitsandbytes
   pip install rouge-score nltk scikit-learn tensorboard wandb
   ```

2. Download and verify model:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model = AutoModelForCausalLM.from_pretrained(
       "Qwen/Qwen3-1.7B",
       torch_dtype=torch.bfloat16,
       device_map="auto"
   )
   tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
   print(model.config)  # RECORD ALL VALUES
   ```

3. Verify hidden state extraction works:
   ```python
   test_input = tokenizer("Hello world", return_tensors="pt").to(model.device)
   with torch.no_grad():
       out = model(**test_input, output_hidden_states=True)
   print(f"Num hidden state layers: {len(out.hidden_states)}")
   print(f"Hidden state shape: {out.hidden_states[-1].shape}")
   # Expected: [1, seq_len, D_model]
   ```

4. Prepare dataset (see Section 6.1)

**Phase 1 Checkpoint:** All components loadable, hidden states extractable, dataset ready. Log all config values.

### 7.2 Phase 2: Baseline Evaluation

**Steps:**

1. Run TextBufferBaseline on the full test set
2. Record: accuracy, ROUGE-L, hallucination rate, latency, memory
3. Run with multiple chunk sizes: {512, 1024, 2048}
4. Record all results to `results/baseline/`

**Phase 2 Checkpoint:** Baseline numbers established. If baseline accuracy < 10% on any task, the task may be too hard for the 1.7B model — consider simplifying or switching datasets.

### 7.3 Phase 3: Latent Pager Training

**What is trained:*Only the `PageCompressor` and `PageAggregator` modules. The base Qwen3-1.7B model is **frozen** throughout.

**Training objective:**

```python
# For each training sample (document, question, gold_answer):
#   1. Chunk the document
#   2. Extract hidden states for each chunk (frozen model, no grad)
#   3. Compress each chunk's hidden states via PageCompressor (trainable)
#   4. Store in LatentPageStore
#   5. Aggregate via PageAggregator (trainable)
#   6. Inject soft prompt + question into frozen model
#   7. Compute cross-entropy loss against gold_answer tokens

loss = cross_entropy(
    logits_from_soft_prompt_generation,
    gold_answer_token_ids
)
```

**Training hyperparameters:**

| Hyperparameter | Value | Notes |
|---|---|---|
| Learning rate | 1e-4 | AdamW, with linear warmup (500 steps) + cosine decay |
| Batch size | 4 | Effective; use gradient accumulation if needed |
| Epochs | 20 | With early stopping |
| `d_page` | 512 | Sweep: {256, 512, 1024} |
| `num_soft_tokens` | 32 | Sweep: {16, 32, 64} |
| `num_extraction_layers` | 4 | Layers {7, 14, 21, 27} |
| Pooling strategy | mean | Also test: last_token |
| `num_agg_layers` | 2 | Cross-attention decoder layers |
| Weight decay | 0.01 | |
| Gradient clipping | 1.0 | Max norm |

**Training monitoring:**
- Log to TensorBoard / W&B: loss, validation accuracy, learning rate
- Save checkpoint every epoch
- Track gradient norms for compressor and aggregator separately

### 7.4 Phase 4: Evaluation and Comparison

Run the trained Latent Pager system on the test set. Compute all metrics from Section 6.2. Compare against baseline.

**Required output files:**

```
results/
├── baseline/
│   ├── metrics.json          # All metrics
│   ├── predictions.jsonl     # Per-sample predictions
│   └── config.json           # Baseline hyperparameters
├── latent_pager/
│   ├── metrics.json
│   ├── predictions.jsonl
│   ├── config.json
│   ├── training_curves.png   # Loss / accuracy over training
│   _sweep.json
│       └── pooling_comparison.json
└── comparison/
    ├── summary_table.md      # Side-by-side metrics
    ├── significance_tests.json
    └── analysis.md           # Written analysis of results
```

### 7.5 Phase 5: Ablation Studies

Run the following ablation experiments (each varies one factor):

| Ablation | Values to Test | Hypothesis |
|---|---|---|
| `d_page` | {128, 256, 512, 1024, 2048} | Higher d_page retains more info but may overfit |
| `num_soft_tokens` | {8, 16, 32, 64, 128} | More tokens → more expressive but slower decode |
| Extraction layers | {last_only, quartiles, all_layers} | Multi-layer captures more abstraction levels |
| Pooling | {mean, last_token} | Last token may carry more "summary" info |
| Number of chunks | {4, 8, 16, 32, 64} on same docs | Tests scalability of aggregator |
| Aggregator depth | {1, 2, 4} layers | Deeper aggregator may help with many pages |

---

## 8. Hypotheses and Predictions

### H1: Latent pages reduce hall** The latent pager system will produce answers with ≥10% lower hallucination rate (relative) compared to text-buffer baseline.

**Rationale:** Text summaries are generated outputs — each is a potential hallucination source. Latent pages preserve the original model's internal representation without generation, removing one hallucination-inducing step.

**Measurement:** Hallucination rate as defined in Section 6.2.3.

**Prediction:** Hallucination rate drops from ~25-35% (text baseline, expected for 1.7B model on long docs) to ~18-28% (latent pager).

### H2: Latent pages improve multi-hop accuracy

**Hypothesis:** On questions requiring information from 2+ document sections, latent pager will achieve ≥5% higher F1 than text buffer.

**Rationale:** Text summaries of individual chunks discard cross-chunk relational information. Latent states preserve implicit associations that the aggregator can exploit.

**Measurement:** F1 score on multi-hop subset of test data.

### H3: Global consistency improves wient aggregation

**Hypothesis:** When asked multiple questions about the same document, the latent pager system will produce more mutually consistent answers.

**Rationale:** All questions see the same aggregated latent representation (deterministic), whereas text-buffer answers depend on the quality of each independent summarization pass.

**Measurement:** Consistency metric from Section 6.2.4.

### H4: Information retention scales with d_page

**Hypothesis:** Probe accuracy (can the latent page recover specific facts?) will increase monotonically with `d_page` up to `D_model`, then plateau.

**Rationale:** Higher-dimensional latent pages have more capacity. At `d_page = D_model` the compressor is essentially an identity-like mapping.

**Measurement:** Fact probe accuracy as a function of `d_page`.

### H5: Compute cost is comparable or lower

**Hypothesis:** Total inference FLOPs for the latent pager system will be ≤1.5x the text-buffer baseline.

**Rationale:** The text baseline requires N generation pses (one per chunk summary) + 1 final pass. The latent pager requires N forward passes (cheaper — no generation) + 1 final generation pass + small aggregator overhead.

**Measurement:** Wall-clock time and estimated FLOPs.

---

## 9. Success Criteria

### 9.1 Experiment is a SUCCESS if ALL of the following hold:

| Criterion | Threshold | Metric |
|---|---|---|
| S1 | Latent pager accuracy (F1) ≥ text baseline accuracy | Task F1 on test set |
| S2 | Latent pager hallucination rate < text baseline hallucination rate | Hallucination metric |
| S3 | Latent pager compute cost ≤ 2x text baseline | Wall-clock time |
| S4 | Aggregator training converges (loss decreases monotonically after warmup) | Training loss curve |

### 9.2 Experiment is a STRONG SUCCESS if additionally:

| Criterion | Threshold |
|---|---|
| S5 | Accuracy improvement ≥ 3 F1 points |
| S6 | Hallucination reduction ≥ 10% relative |
| S7 | Improvement is consistent across all task types |
| S8 | Scaling curve: accuracy increases withs (more chunks of the same doc) |

### 9.3 Experiment is a PARTIAL SUCCESS if:

- S1 holds but S2 does not (latent pages help accuracy but not hallucination)
- S2 holds but S1 does not (latent pages reduce hallucination at cost of accuracy)
- Results are task-type-dependent (works for aggregation but not single-hop)

### 9.4 Experiment is a FAILURE if:

| Criterion | Condition |
|---|---|
| F1 | Latent pager accuracy < text baseline by > 3 F1 points |
| F2 | Aggregator training does not converge after 20 epochs |
| F3 | Latent pager hallucination rate > text baseline |
| F4 | System OOMs on test samples consistently |

---

## 10. Stop Criteria

### 10.1 Early Stopping During Training

```python
PATIENCE = 5  # epochs without improvement
MIN_DELTA = 0.001  # minimum improvement to count

# Stop training if:
# - Validation loss has not improved by MIN_DELTA for PATIENCE consecutive epochs
# - Training loss is NaN or Inf
# - Gradient norm exceeds 100.0 for 3 consecutive steps (instability)
# - Validation accuracy drops by > 5% from best (catastrophic forgetting)
```

### 10.2 Experiment-Level Stop Criteria

**STOP the entire experiment and report findings if:**

1. **Phase 1 blocker:** Model cannot be loaded with `output_hidden_states=True` → report incompatibility
2. **Phase 2 blocker:** Text baseline accuracy < 5% on all tasks → model is too weak for these tasks; simplify dataset
3. **Phase 3 blocker:** Aggregator training loss does not decrease after 1000 steps → architecture bug or learning rate issue; debug, try LR in {1e-3, 1e-4, 1e-5}. If none work after 3 attempts, report failure
4. **Phase 3 blocker:** OOM during training → reduce batch size to 1, enable gradient checkpointing, reduce `num_soft_tokens` to 8. If still OOM, report hardware limitation
5. **Phase 4 blocker:** Statistical significance test (paired bootstrap, p < 0.05) shows no difference between latent pager and baseline on ANY metric → report null result
6. **Budget exhaustion:** If total experiment wall-clock exceeds 72 hours of stop and report partial results

### 10.3 Hyperparameter Search Stop

For each ablation sweep:
- Run at most 5 values per hyperparameter
- If the first 3 values show no clear trend, skip remaining values and move on
- If a sweep reveals a clear optimum, use it for subsequent experiments

---

## 11. Repository Structure

```
latent-pager-memory/
├── README.md                     # This document
├── requirements.txt
├── setup.py
├── configs/
│   ├── default.yaml              # Default hyperparameters
│   ├── ablation_d_page.yaml
│   ├── ablation_soft_tokens.yaml
│   └── ablation_pooling.yaml
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── latent_extractor.py   # Hidden state extraction
│   │   ├── page_compressor.py    # PageCompressor module
│   │   ├── page_aggregator.py    # PageAggregator module
│   │   ├── page_store.py         # LatentPageStoretrator
│   ├── baseline/
│   │   ├── __init__.py
│   │   └── text_buffer.py        # TextBufferBaseline
│   ├── data/
│   │   ├── __init__.py
│   │   ├── chunker.py            # DocumentChunker
│   │   ├── dataset_builder.py    # Synthetic OOLONG-style dataset
│   │   └── data_loader.py        # PyTorch DataLoader wrappers
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # Accuracy, ROUGE, hallucination
│   │   ├── consistency.py        # Global consistency checker
│   │   ├── probes.py             # Information retention probes
│   │   └── significance.py       # Paired bootstrap tests
│   └── training/
│       ├── __init__.py
│       ├── trainer.py            # Training loop for compressor + aggregator
│       └── scheduler.py          # LR scheduler, early stopping
├── scripts/
│   ├── 01_setup_and_         # Phase 5
│   └── 06_generate_report.py     # Final comparison report
├── results/                      # All outputs (see Section 7.4)
├── checkpoints/                  # Model checkpoints
└── logs/                         # Training logs
```

---

## 12. Implementation Order and Priority

Execute scripts in numbered order. Each script should be independently runnable and should check for the existence of prior outputs.

| Priority | Script | Estimated Time | Dependencies |
|---|---|---|---|
| P0 | `01_setup_and_verify.py` | 10 min | None |
| P0 | `02_run_baseline.py` | 2-6 hours | Phase 1 outputs |
| P0 | `03_train_latent_pager.py` | 8-24 hours | Phase 1 + 2 outputs |
| P0 | `04_evaluate.py` | 2-6 hours | Trained model |
| P1 | `05_ablations.py` | 12-36 hours | Trained model |
| P1 | `06_generate_report.py` | 5 min | All prior outputs |

**P0 = must complete. P1 = complete if time permits.**

---

## 13. Failure Modes and Mitigations

| Failure Mode | Detection | Mitigatressor destroys information | Probe accuracy near random | Increase `d_page`, add skip connection, try autoencoder pre-training |
| Aggregator doesn't learn cross-page relationships | Multi-hop accuracy = single-hop accuracy | Increase `num_agg_layers`, add positional encoding to pages |
| Soft-prompt injection is ignored by frozen LM | Model output doesn't change with different soft prompts | Try prefix-tuning formulation, inject at multiple layers |
| Training instability (NaN/Inf) | Loss monitoring | Reduce LR, add gradient clipping, check for exploding norms in compressor |
| OOM | CUDA OOM error | Reduce batch size, chunk size, `num_soft_tokens`; use 8-bit model loading |
| Baseline is too strong (no room for improvement) | Baseline accuracy > 90% | Use harder tasks or longer documents |
| Baseline is too weak (floor effect) | Baseline accuracy < 10% | Use easier tasks or shorter documents |

---

## 14. Logging and Reproducibility

- **Random seeds:** Set `torch.manual_seed(42)`, `numpy.random.seed(42)`, `random.seed(42)` at the start of every script
- **Log all hyperparameters** to a JSON/YAML file before each run
- **Log environment:** Python version, PyTorch version, CUDA version, transformers version, GPU model
- **Save raw predictions:** Every sample's prediction should be saved for post-hoc analysis
- **Deterministic operations:** Set `torch.use_deterministic_algorithms(True)` where possible (disable if it causes CUDA errors)
- **Git:** If running in a repo, commit before each phase and tag the commit

---

## 15. Key Implementation Notes for the Agent

1. **Qwen3-1.7B access to hidden states:** Use `output_hidden_states=True` in the forward call. Hidden states are returned as `outputs.hidden_states` — a tuple of `(num_layers + 1)` tensors (including embedding layer output at index 0).

2. **Embedding access for soft-prompt injection:** The embedding layer is at `model.model.embed_tokens`. Use this to get token embeddings, then concatenate soft-prompt embeddings before passing to `model.generate` via `inputs_embeds`.

3. **Frozen model:** Always wrap Qwen3-1.7B operations in `torch.no_grad()` and ensure `model.eval()`. Only the `PageCompressor` and `PageAggregator` parameters should require gradients.

4. **Memory management:** After extracting hidden states from a chunk, immediately detach and move to CPU. Only move to GPU when aggregating/training. Call `torch.cuda.empty_cache()` between chunks if memory is tight.

5. **Tokenizer:** Qwen3 uses a SentencePiece-based tokenizer. Use `tokenizer.apply_chat_template()` for prompt formatting if using the instruct variant. For the base model, direct tokenization is fine.

6. **Generation:** Set `presence_penalty=1.5` if generating with the instruct model to avoid repetition (per Qwen3 best practices).

---

## 16. Final Deliverables

Upon completion, the agent must produce:

1. **All code** in the repository structure above, runnable end-to-end
2. **`results/comparison/summary_table.md`** — side-by-side metrics comparison
3. **`results/comparison/analis.md`** — written analysis (2-3 paragraphs) of whether each hypothesis (H1-H5) is supported
4. **`results/latent_pager/training_curves.png`** — training loss and validation accuracy curves
5. **`checkpoints/best_model.pt`** — best aggregator + compressor weights
6. **A final verdict:** SUCCESS / STRONG SUCCESS / PARTIAL SUCCESS / FAILURE with justification referencing specific metrics from Section 9

---

*End of handoff documentation.*
