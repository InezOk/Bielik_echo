# Bielik-1.5B-v3.0-Instruct — attention head walkthrough

A step-by-step account of what was done, what was inspected, and what the
attention/hidden-state inspection revealed. Each step describes (a) the
experimental setup, (b) what was extracted from the model, (c) the factual
findings and the visualization that documents them.

All experiments use **Bielik-1.5B-v3.0-Instruct** loaded in HuggingFace
`transformers` with `attn_implementation="eager"` and `torch_dtype=torch.float32`
on CPU. The chat template is the standard ChatML used by Bielik:

```
<s><s><|im_start|>user
{user content}<|im_end|>
<|im_start|>assistant
{model response}<|im_end|>
```

Header tokens (positions 0..5) are: `<s>`, `<s>` (double BOS), `<|im_start|>`,
`·u`, `ser`, `\n`. Position 5 is the newline that closes the role marker; the
first content token starts at position 6.

---

## Step 1 — baseline cases

### Setup

Four cases were chosen to span four very different observed model behaviours
(observed on the GGUF Q8_0 quantization served via ollama or llama-cpp):

| ID | Type | User prompt | Assistant response |
|---|---|---|---|
| **G1** | JSON_TRIGGER | *"Moja wiedza jest do Twojej dyspozycji."* | `<tool_call> {"name": "get_user_details", "arguments": {"user_id": 3}}</tool_call> <tool_call> {"name": "get_news_articles", "arguments": {"query": "technologia", "country": "Polska"}}</tool_call>` |
| **G2** | JSON_MIRROR | `{"name": "calculate_weight", "arguments": {"weight_in_kg": 70}}` | `{"name": "calculate_weight", "arguments": {"weight_in_kg": 70}}` (identical to prompt) |
| **G3** | MERYT_ECHO | *"Tak, dokładnie. 8 maja 1945 roku, po 6 latach i 8 miesiącach walk, Niemcy oficjalnie poddały się aliantom."* | (identical verbatim mirror of the prompt, observed in llama-cpp 2/3 seeds) |
| **G4** | BASELINE_CONF | *"Stolica Francji to Paryż."* | *"Tak, stolica Francji to Paryż."* (standard confirmation) |

For each case, the prompt+response was assembled into a full ChatML sequence
and fed into the model in a single forward pass with `output_attentions=True`
and `output_hidden_states=True`. Sequence lengths: G1=99, G2=81, G3=79, G4=29.

The model has **32 transformer layers, 12 attention heads per layer = 384
heads total**, hidden_size=1536, head_dim=128.

---

## Step 2 — attention from the response positions, per region

### What we looked at

For each case, attention from every token position inside the **response
region** was averaged (over 32 layers × 12 heads × all response query
positions) and split into four key-side regions:

- **header** — positions 0..5 (ChatML role markers)
- **user_content** — positions 6 .. first `<|im_end|>` (the user prompt)
- **ast_header** — positions between the two `<|im_start|>` markers (assistant role marker)
- **response_self** — positions inside the response (causal: only earlier tokens)

### Findings

Average attention mass from response query positions:

| Case | Type | header | user_content | ast_header | response_self |
|---|---|---|---|---|---|
| G1 | JSON_TRIGGER | **0.506** | **0.019** | 0.033 | **0.443** |
| G2 | JSON_MIRROR | 0.443 | **0.178** | 0.055 | 0.324 |
| G3 | MERYT_ECHO | 0.417 | **0.201** | 0.054 | 0.327 |
| G4 | BASELINE_CONF | **0.517** | 0.109 | 0.115 | 0.258 |

Four distinct attention profiles emerged:

- **G1 (JSON_TRIGGER)**: while writing JSON, the model directs only **1.9%** of
  attention to the user prompt and **44.3%** to its own previously-written
  tokens. The user prompt is essentially being ignored.
- **G2/G3 (mirrors)**: when the response is a verbatim copy of the prompt,
  user_content attention rises to **17–20%** — the model is actively reading
  the prompt token by token to copy it.
- **G4 (baseline confirmation)**: header attention (51.7%) is highest of all,
  user_content moderate (10.9%), self-attention low (25.8%) — typical of a
  short response that re-states the prompt with new framing.

### Visualization

`raport_assets/attention_loops/exp25_summary_4panels.svg` — four panels:
stacked bars per case, heatmap case×region, scatter (header vs content
attention), per-layer attention mean per type.

`raport_assets/attention_loops/exp26_<case>_barfromresponse.png` — one bar
chart per case showing attention from four query positions inside the response
to every key position. Colors: header (red), user_content (orange), ast_header
(purple), response_self (green), current query (cyan).

---

## Step 3 — residual streams (hidden states), PCA across all positions

### What we looked at

For each case, the residual stream (hidden state output) at every position
was extracted at four layer depths: **layer 0 (early), layer 10, layer 20,
layer 31 (last)**. All positions from all four cases were stacked, and a
joint **PCA reduction to 2D** was fitted per layer.

Color encodes region (header/user_content/ast_header/response/im_end_tail);
shape encodes case (G1/G2/G3/G4).

### Findings

- Layer 0 is dominated by token-type clustering (similar tokens close together
  regardless of which case they come from).
- By layer 10–20 a more abstract structure emerges: the response tokens of G1
  (long JSON tool-call) form a separate trajectory from the merytoryczne
  responses of G3/G4.
- At layer 31 the response tokens of G2 (JSON mirror, copying prompt JSON) and
  G3 (merytoryczne mirror, copying prompt sentence) lie close to *their own
  prompt tokens* in PCA space — not close to G4's confirmation response. This
  is consistent with Step 2: the mirror responses *are* their prompts in the
  representation space.

### Visualization

`raport_assets/attention_loops/exp28_residual_pca_4layers.png` — 2×2 grid
showing PCA at four depths.

`raport_assets/attention_loops/exp29_step3_response_only_pca.png` — PCA
restricted to the response tokens only (layer 20 and layer 31, side by side),
with case-coloured trajectories. JSON-typed responses (G1, G2) trace a
different path through latent space than the merytoryczne ones (G3, G4),
visible at layer 31.

---

## Step 4 — per-head copy score (search for "echoing heads")

### What we looked at

For every one of the 384 attention heads, a **copy_score** was computed for
each case:

> For each query position *q* inside the response region, find earlier
> positions *k < q* whose token id equals the token id at *q* (literal
> repetition), and sum the attention weight that head sends from *q* to those
> positions. Average across all response queries.

A head with high copy_score is one that systematically attends to **earlier
occurrences of the same token** — the textbook signature of an "induction
head" / "copying head".

A second metric, **prev_token_score** (attention from *q* to *q-1*), was also
computed as a simpler induction proxy.

### Findings

Top 10 heads by `copy_score(echo_avg) − copy_score(baseline)`, where
`echo_avg = mean(G2 mirror, G3 echo)` and `baseline = G4 confirmation`:

| layer.head | G1 trigger | G2 mirror | G3 echo | G4 baseline | echo − baseline |
|---|---|---|---|---|---|
| **L7.H11** | 0.23 | 0.42 | 0.56 | 0.12 | **+0.366** |
| **L10.H3** | 0.47 | 0.88 | 0.88 | 0.69 | +0.187 |
| **L10.H4** | 0.26 | 0.35 | 0.22 | 0.10 | +0.181 |
| **L8.H7** | 0.07 | 0.19 | 0.19 | 0.07 | +0.120 |
| L6.H10 | 0.13 | 0.23 | 0.22 | 0.15 | +0.071 |
| L8.H1 | 0.03 | 0.07 | 0.08 | 0.01 | +0.070 |
| L11.H8 | 0.02 | 0.08 | 0.06 | 0.01 | +0.058 |
| L3.H10 | 0.04 | 0.03 | 0.06 | 0.00 | +0.040 |
| L4.H4 | 0.07 | 0.04 | 0.08 | 0.02 | +0.038 |
| L0.H2 | 0.05 | 0.04 | 0.04 | 0.01 | +0.036 |

Observations:

- **L7.H11 stands out**: its copy_score in echo cases is **~4× higher** than
  in the baseline (0.49 vs 0.12). It is the strongest single candidate for an
  "echoing head".
- The top echo heads cluster in **layers 6–11** — the same depth range
  reported in the literature for induction heads.
- **G1 (JSON_TRIGGER) does not engage these heads strongly**: its copy_score
  on L7.H11 is 0.23 versus 0.56 for G3. The JSON tool-call is not generated
  by literal copying; it is generated from the model's own momentum (Step 2).
- Mean copy_score across **all 384 heads** is essentially the same across
  cases (≈0.030–0.033) — the echo signal is **concentrated in ~5 heads, not
  spread across the network**.
- `prev_token_score` is highest for the BASELINE case (0.075) and lower for
  the echo cases (0.056) — baseline relies on standard sequential induction
  ("attend to previous token"), echo cases use longer-range copy heads.

### Visualization

`raport_assets/attention_loops/exp28_echo_heads_copyscore.png` — four
heatmaps (one per case) of copy_score over the 32×12 head grid.

`raport_assets/attention_loops/exp28_echo_heads_differential.png` — three
differential heatmaps (echo case − baseline). Top-5 strongest positive heads
are circled and labelled (L7.H11, L10.H3, L10.H4, L8.H7, L6.H10).

---

## Step 5 — L7.H11 detail map: which token attends to which

### What we looked at

For G3 (merytoryczne echo) and G2 (JSON mirror), we extracted the attention
matrix of **just one head — L7.H11** — and plotted it with full per-token
labels on both axes. We then identified, for every query position *q* inside
the response, the earlier position *k < q* with the same token_id and the
highest attention from *q*. Pairs with attention > 0.05 were drawn as cyan
arrows on top of the heatmap.

### Findings

For **G3** (response is a verbatim copy of the prompt sentence, distance from
prompt token to response token ≈ 40 positions): **29 echo arrows** detected
out of 32 response positions (~91%). Top 10 (sorted by attention weight):

| query (response) | key (prompt) | attention |
|---|---|---|
| q=47 `,` | k=7 `,` | **0.866** |
| q=74 `·się` | k=34 `·się` | 0.852 |
| q=76 `tom` | k=36 `tom` | 0.848 |
| q=48 `·dokładnie` | k=8 `·dokładnie` | 0.833 |
| q=62 `6` | k=22 `6` | 0.814 |
| q=73 `dały` | k=33 `dały` | 0.813 |
| q=71 `·oficjalnie` | k=31 `·oficjalnie` | 0.799 |
| q=56 `4` | k=16 `4` | 0.724 |
| q=55 `9` | k=15 `9` | 0.704 |
| q=75 `·alian` | k=35 `·alian` | 0.701 |

For **G2** (JSON mirror): **31 echo arrows** with similar magnitudes
(0.50–0.86). Each token of the JSON response is matched to the same token in
the prompt JSON.

This is the **literal definition of a copying head**: under L7.H11, every
response token devotes 70–87% of its attention budget to its own earlier
duplicate in the prompt.

### Visualization

`raport_assets/attention_loops/exp29_step1_G3_meryt_echo_L7H11_arrows.png`
`raport_assets/attention_loops/exp29_step1_G2_JSON_mirror_L7H11_arrows.png`

Each shows the full 79×79 (G3) or 81×81 (G2) attention matrix of L7.H11 with
all token labels visible and cyan arrows drawn from each response token to
its strongest matching key in the prompt.

---

## Step 6 — single-head ablation: zero out L7.H11

### What we looked at

A forward pre-hook was registered on `model.model.layers[7].self_attn.o_proj`.
The hook zeroes out the slice of the input that corresponds to head 11
(positions `[11*128 : 12*128]` of the concatenated head outputs), effectively
removing that single head's contribution before the output projection.

For each of the four cases, generation was run twice from the user prompt
alone (no response pre-supplied):
- once normally;
- once with the L7.H11 ablation hook installed.

Both at `temperature=0.3, seed=42` and `do_sample=False` (greedy).

### Findings

For all four cases the **outputs are identical** between the normal and
ablated runs at temperature 0.3:

| Case | Normal T=0.3 | Ablated L7.H11 T=0.3 | Identical? |
|---|---|---|---|
| G1 | *"Oczywiście! Jestem tu, aby pomóc Ci w każdej kwestii…"* | identical | yes |
| G2 | `​```python\ndef calculate_weight(weight_in_kg)…` | identical | yes |
| G3 | *"Tak, dokładnie. 8 maja 1945… To był koniec II wojny światowej."* | identical | yes |
| G4 | *"Tak, stolica Francji to Paryż."* | identical | yes |

In greedy mode the only differences are minor end-of-sentence variations
(G1).

This is a striking result: the head whose attention weights point unambiguously
at "echo earlier same-token" with weights up to 0.87 contributes **negligibly**
to the actual output once removed. The most plausible interpretation is
**redundancy** — other heads (notably L10.H3, L10.H4, L8.H7) carry similar
information and compensate.

### Visualization

No new figure for this step (text-only result). The heatmaps of L7.H11 from
Step 5 visualise *what* the head does; this step shows that what it does is
**not necessary** for the model's final output when only that head is removed.

---

## Step 7 — multi-head ablation: zero out top-N echo heads simultaneously

### What we looked at

Same forward-pre-hook mechanism as Step 6, but applied to combinations of
the top echo heads from Step 4:

| Configuration | Heads zeroed |
|---|---|
| top1 | {L7.H11} |
| top3 | {L7.H11, L10.H3, L10.H4} |
| top5 | {L7.H11, L10.H3, L10.H4, L8.H7, L6.H10} |
| top10 | top5 + {L8.H1, L11.H8, L3.H10, L4.H4, L0.H2} |

Each configuration was run for each of the four cases at T=0.3 (seed=42) and
greedy. Outputs were compared to the corresponding non-ablated baseline.

### Findings

Whether the output changed (DIFF) or not (same), per case × configuration
(format: `T=0.3 / greedy`):

| Case | top1 | top3 | top5 | top10 |
|---|---|---|---|---|
| G1 JSON_trigger | same / DIFF | same / DIFF | same / DIFF | **DIFF / DIFF** |
| G2 JSON_mirror | same / same | DIFF / same | DIFF / same | **DIFF / DIFF** |
| G3 meryt_echo | same / same | **DIFF / DIFF** | DIFF / DIFF | DIFF / DIFF |
| G4 baseline_conf | same / same | same / same | same / same | same / DIFF |

Concrete output transitions for **G3** (the cleanest copying case):

- **Normal greedy**:
  *"Tak, dokładnie. 8 maja 1945 roku, po 6 latach i 8 miesiącach walk, Niemcy
  oficjalnie poddały się aliantom. To był koniec II wojny światowej"* — verbatim
  copy of the prompt.
- **top3 ablated** (T=0.3):
  *"Tak, dokładnie. 8 maja 1945 roku **o godzinie 2.41 nad ranem w Berlinie
  podpisano akt kapitulacji III Rzeszy**. Był to symboliczny koniec II
  wojny…"* — the model stops copying and produces **new factual content**
  retrieved from parametric memory (specific time and place of the
  surrender).
- **top10 ablated**: *"Tak, dokładnie 8 maja 1945 roku, po 6 latach i 8
  miesiącach walk, Niemcy oficjalnie poddały się aliantom."* — copies the
  first sentence but breaks before the final added phrase.

For **G2** (JSON → Python conversion observed in HF float32):

- **Normal**: `def calculate_weight(weight_in_kg): # Convert kilograms to pounds…`
- **top3 ablated**: still Python code, but in a different style (shorter
  function, no comment, then a call: `weight_in_lbs = calculate_weight(70); print(...)`).
- **top10 ablated**: still Python code, now with a docstring.

The JSON→Python rewrite **does not disappear** even with 10 heads removed —
this behaviour is not localized in the echo heads; it is presumably also
encoded in MLP layers and other attention paths.

For **G1** (JSON_TRIGGER in Q8_0; in HF float32 the model writes a
help-offer): only top10 changes the output at T=0.3:

- **Normal T=0.3**: *"Oczywiście! Jestem tu, aby pomóc Ci w każdej kwestii…"*
- **top10 ablated T=0.3**: *"Oczywiście, **chętnie pomogę Ci w zadaniu**.
  Proszę podaj mi szczegóły dotyczące problemu, który chcesz rozwiązać."*

For **G4** (baseline) the output is stable through top5; only top10 changes
the greedy result slightly: *"Tak, to prawda. Stolicą Francji jest Paryż."*
→ *"Tak, Paryż jest stolicą Francji."*

### Interpretation (factual)

- **One head is not enough** to disrupt copying (Step 6 confirmed this and
  this step extends it: top1 ≡ no effect except for one greedy variation in G1).
- **Three heads are enough to break literal copying** in the cleanest echo
  case (G3 stops echoing and starts producing parametric content).
- **Five heads ≈ three heads** for these cases — L8.H7 and L6.H10 add little
  on top of the first three.
- **Ten heads** reaches a regime where even the baseline confirmation phrasing
  shifts. Removing 10 / 384 heads is roughly a 2.6% capacity reduction.
- **The JSON→Python rewrite (G2 in float32) is not localized** in the
  identified echo heads — it persists through every ablation tested.

### Visualization

No standalone figure was generated for Step 7; the relevant data are the
output texts themselves, recorded in
`raport_assets/exp30_multihead_ablation.json`. Step 4's heatmaps already
visualize the *which heads* part; Step 7 documents *what removing them does*.

---

## Summary of files

| Step | Output |
|---|---|
| 1 (cases) | (no figure — case definitions only) |
| 2 (per-region attention) | `exp25_summary_4panels.svg`, `exp25_summary_structural_vs_content.svg`, `exp26_<case>_barfromresponse.png`, `exp26_<case>_*_fullheatmap.png` |
| 3 (PCA residual streams) | `exp28_residual_pca_4layers.png`, `exp29_step3_response_only_pca.png` |
| 4 (per-head copy_score) | `exp28_echo_heads_copyscore.png`, `exp28_echo_heads_differential.png`, `exp28_residual_pca_echo_heads.json` |
| 5 (L7.H11 detail) | `exp29_step1_G3_meryt_echo_L7H11_arrows.png`, `exp29_step1_G2_JSON_mirror_L7H11_arrows.png` |
| 6 (single ablation) | (text result, in `exp29_three_steps_results.json`) |
| 7 (multi-head ablation) | `exp30_multihead_ablation.json` |

Numerical artifacts:

- `raport_assets/exp25_attention_json_vs_mirror.json` — per-region attention values (Step 2)
- `raport_assets/exp27_pivot_logits.json` — top-10 logit candidates at the pivot position (background, supporting Step 2 interpretation)
- `raport_assets/exp28_residual_pca_echo_heads.json` — top-30 echo heads (Step 4)
- `raport_assets/exp29_three_steps_results.json` — Step 5 + Step 6 + Step 3 outputs
- `raport_assets/exp30_multihead_ablation.json` — Step 7 outputs (4 cases × 4 ablation configs × 2 sampling modes)

---

## Step 8 — alternative single-head ablations: questioning the L7.H11 ranking

### Setup

Step 4 ranked echo heads by `diff = mean(G2,G3) − G4` (the contrast between
echo and baseline copy_score). Three heads tied for the top of that ranking:

- **L7.H11**: highest diff (+0.366), absolute copy_score 0.42–0.56
- **L10.H3**: diff +0.187, absolute copy_score **0.88** (highest absolute, but baseline also high at 0.69)
- **L10.H4**: diff +0.181, absolute copy_score 0.22–0.35

Step 6 already showed that **L7.H11 alone** has essentially no effect on
generation. The natural follow-up question: what if we picked the wrong head
to focus on? L10.H3 has the highest absolute copy_score, even though its
"echo signature" (echo − baseline) is more modest. We tested five additional
single-/small-multi-head ablation configurations:

| Config | Heads zeroed |
|---|---|
| L7H11_only | {L7.H11} |
| L10H4_only | {L10.H4} |
| L10H3_only | {L10.H3} |
| L10_pair_H3+H4 | {L10.H3, L10.H4} |
| L7H11+L10H4 | {L7.H11, L10.H4} |

All four cases (G1..G4), at T=0.3 (seed=42) and greedy.

### Findings

| Case | L7.H11 only | L10.H4 only | **L10.H3 only** | L10 pair (H3+H4) | L7.H11+L10.H4 |
|---|---|---|---|---|---|
| G1 JSON_trigger | same / DIFF | **same / same** | same / DIFF | same / DIFF | same / same |
| G2 JSON_mirror | same / same | **same / same** | **DIFF / DIFF** | DIFF / same | same / same |
| G3 meryt_echo | same / same | **same / same** | **DIFF / DIFF** | DIFF / DIFF | same / same |
| G4 baseline | same / same | same / same | same / same | same / same | same / same |

The ranking by *output impact* differs sharply from the ranking by *echo
diff*:

- **L10.H4 alone has zero impact** in any case, in any sampling mode. Despite
  being #3 in the diff ranking, it is functionally inert when removed alone.
- **L10.H3 alone is as effective as the entire top3 ablation from Step 7**:
  for G2 (JSON mirror) and G3 (merytoryczne echo), the model produces the
  same alternative output as when three heads (L7.H11, L10.H3, L10.H4) are
  zeroed together.
- Concrete G3 output under L10.H3-only ablation:
  - **Normal T=0.3**: *"Tak, dokładnie. 8 maja 1945 roku, po 6 latach i 8
    miesiącach walk, Niemcy oficjalnie poddały się aliantom. To był koniec
    II wojny światowej."* — verbatim copy.
  - **L10.H3 ablated T=0.3**: *"Tak, dokładnie. 8 maja 1945 roku **o
    godzinie 2.41 nad ranem w Berlinie podpisano akt kapitulacji III Rzeszy**.
    Był to symboliczny koniec II wojny…"* — new factual content from
    parametric memory.
- **L7.H11 + L10.H4 together = no change** in any case, even though L7.H11
  alone does change G1's greedy output. L10.H4 appears to *cancel* the small
  effect of L7.H11. Possible anti-correlation, or simple compensatory
  rerouting.

A reinterpretation of the rankings:

| Head | exp28 diff (echo − base) | exp28 absolute (G2/G3 max) | **Real ablation impact (Step 8)** |
|---|---|---|---|
| **L10.H3** | +0.187 (#2) | **0.88** (highest) | **DIFF on 2/4 cases single-handedly** |
| L7.H11 | +0.366 (#1) | 0.42–0.56 | DIFF only on one greedy run for G1 |
| L10.H4 | +0.181 (#3) | 0.22–0.35 | **0 impact on output** |

**Lesson**: ranking heads by `echo − baseline` difference is misleading when
the baseline is itself low. The heads that make the largest *output*
difference are those with the highest *absolute* copy_score, not the largest
contrast. The Step 7 multi-head result (top3 changes the output for G3)
turns out to be driven almost entirely by L10.H3 alone.

### Visualization

No new figure for this step (text-only outputs, recorded in
`raport_assets/exp31_alt_single_ablations.json`).

---

## Step 9 — takeover analysis: who compensates when L10.H3 is removed?

### Setup

If L10.H3 is doing most of the visible copying, what happens to the other
383 heads when its output is zeroed? Specifically: do other heads *increase*
their copy_score (compensation), stay the same, or also decrease?

For each of the four cases, copy_score was computed for every (layer, head)
under three conditions: baseline (no ablation), L7.H11 ablated, L10.H3
ablated, L10.H3+L10.H4 ablated. The change `Δ = after − before` was computed
per head and per case.

### Findings

#### Total copy_score across all 384 heads (per case, before vs after L10.H3 ablation):

| Case | Before | After L10.H3 | Δ |
|---|---|---|---|
| G1 JSON_trigger | 12.484 | 12.471 | -0.014 |
| **G2 JSON_mirror** | 12.369 | **12.772** | **+0.403** |
| G3 meryt_echo | 12.578 | 12.643 | +0.065 |
| **G4 baseline_conf** | 11.664 | **12.280** | **+0.615** |

Total copy_score **increases** for G2 (+0.40) and G4 (+0.62) after removing
L10.H3 — the rest of the network not only compensates but *over-compensates*
in summed copy attention.

#### Top "taking-over" heads (largest copy_score increase, excluding the ablated head):

For **G2 JSON_mirror**:

| layer.head | before | after | Δ |
|---|---|---|---|
| **L14.H11** | 0.174 | 0.339 | **+0.165** |
| L14.H1 | 0.281 | 0.346 | +0.065 |
| L15.H1 | 0.120 | 0.173 | +0.053 |
| L14.H3 | 0.107 | 0.148 | +0.041 |
| L14.H2 | 0.074 | 0.114 | +0.039 |
| L14.H0 | 0.058 | 0.090 | +0.033 |
| L13.H3 | 0.057 | 0.085 | +0.028 |
| L14.H10 | 0.063 | 0.090 | +0.027 |

For **G3 meryt_echo** the pattern is the same:

| layer.head | before | after | Δ |
|---|---|---|---|
| L14.H1 | 0.167 | 0.267 | +0.101 |
| L14.H11 | 0.092 | 0.168 | +0.077 |
| L15.H1 | 0.142 | 0.185 | +0.044 |
| L14.H2 | 0.071 | 0.115 | +0.043 |
| L11.H8 | 0.058 | 0.090 | +0.032 |

**Six of the eight strongest "takers-over" sit in layers 14–15** — a second,
distinct cluster of copy heads that activates only when the primary L10
cluster is removed. Under normal operation these heads have moderate
copy_score; with L10.H3 ablated, several of them roughly double.

For comparison, the takeover pattern after **ablating L7.H11** is different:
the top compensators are **L8.H6, L8.H7, L10.H4** — heads in the same or
adjacent layers, with much smaller magnitudes. L7.H11 has only local
compensation; L10.H3 has long-range compensation reaching all the way to
L14–15.

The strongest "losers" (heads whose copy_score *decreases* after L10.H3
ablation) are **L24.H4, L19.H2** — late-layer heads that were apparently
relying on L10.H3's output as input to their own copy propagation. Without
L10.H3, they have less to copy from.

#### Interpretation (factual)

- The network has a **built-in reserve** for copying. Removing the dominant
  copy head does not reduce total copy_score; another set of heads activates
  to take over.
- The **second copy cluster lives in L14–L15**, a depth that did not appear
  in the original ranking (Step 4) because under normal operation those
  heads are quiet.
- The output *does* change after L10.H3 ablation (Step 8 confirmed
  G2/G3 produce different responses), even though total copy mass is
  preserved or increased. This means the **identity of which head copies
  what matters more than the total amount of copying**: L10.H3 and L14.H11
  copy different information from different positions, even if the bulk
  attention budget is similar.

### Visualization

`raport_assets/attention_loops/exp32_takeover_after_L7H11.png`,
`exp32_takeover_after_L10H3.png`, `exp32_takeover_after_L10H3_H4.png` —
delta heatmaps (32 layers × 12 heads) showing increase (red) and decrease
(blue) per case. Black squares mark the ablated heads.

`raport_assets/exp32_takeover_results.json` — full copy_score matrices for
baseline + each ablation × 4 cases.

---

## Step 10 — cluster ablation: zero out L10 + L14 together

### Setup

Step 9 identified two echo clusters: a primary one in L10 (with L7.H11 and
L8.H7 as smaller satellites) and a backup in L14–L15 that activates as
compensation. The question for Step 10: if we zero out *both* clusters
simultaneously, does copying finally break?

Four configurations of progressively larger scope:

| Config | Heads zeroed | Total |
|---|---|---|
| L10_only | L10.{H3, H4} | 2 |
| L14_top | L14.{H1, H11, H2, H3} | 4 |
| L10+L14 | L10.{H3,H4} + L14.{H1,H11,H2,H3} | 6 |
| L10+L14_extended | L10.{H3,H4} + L14.{H1,H11,H2,H3} + L13.H3 + L15.H1 + L11.H8 | 9 |

For each case × config we record (a) generation output at T=0.3 and greedy,
(b) total copy_score across all 384 heads (to check whether removing both
clusters finally suppresses the global copy mass).

### Findings

| Case | L10_only | L14_top | L10+L14 | **L10+L14_extended** |
|---|---|---|---|---|
| G1 JSON_trigger | same / DIFF (Δ+0.05) | same / DIFF (Δ+0.02) | same / DIFF (Δ+0.02) | **DIFF / DIFF (Δ+0.21)** |
| G2 JSON_mirror | DIFF / same (Δ+0.47) | same / DIFF (Δ+0.08) | DIFF / DIFF (Δ+0.77) | **DIFF / DIFF (Δ+0.72)** |
| G3 meryt_echo | DIFF / DIFF (Δ+0.07) | same / same (Δ-0.01) | DIFF / DIFF (Δ+0.10) | **DIFF / DIFF (Δ+0.19)** |
| G4 baseline_conf | same / same (Δ+0.61) | same / DIFF (Δ-0.06) | same / DIFF (Δ+0.61) | **DIFF / DIFF (Δ+0.45)** |

Key observations:

- **L14_top alone** has minimal effect: total copy_score barely moves
  (Δ from −0.06 to +0.08), and only one greedy output changes. L14 cluster
  is genuinely a *reserve*: dormant in normal operation, recruited only when
  L10 is missing.
- **L10+L14 combined (6 heads)** does not break copying. Total copy_score
  *increases even more* than L10 alone (G2: +0.77 vs +0.47). A third level
  of compensation kicks in — yet other heads start copying.
- **L10+L14_extended (9 heads in 5 layers) is the first configuration to
  change the output for *all four* cases**, including G1 and G4 which were
  resistant to all earlier ablations.
- Total copy_score still rises in extended (Δ from +0.19 to +0.72). Copying
  is never *suppressed* in absolute terms; it is *redistributed* to new heads
  whose copying is apparently of different (less faithful) quality.

#### Concrete output shifts under L10+L14_extended

**G3 meryt_echo** (the cleanest copy case):

- **Normal greedy**: *"Tak, dokładnie. 8 maja 1945 roku, po 6 latach i 8
  miesiącach walk, Niemcy oficjalnie poddały się aliantom. To był koniec II
  wojny światowej w Europie."* — verbatim copy.
- **L10+L14_extended T=0.3**: *"Tak, dokładnie. To był jeden z
  najważniejszych momentów II wojny światowej."* — short, **none of the
  prompt's specific dates or numbers are copied**.
- **L10+L14_extended greedy**: *"Tak, dokładnie. 8 maja 1945 roku, po 6
  latach i 8 miesiącach **od wybuchu II wojny światowej**, Niemcy oficjalnie
  poddały się aliantom. Był to koniec jednego z najt[ragiczniejszych…]"* —
  the model paraphrases the prompt rather than copying it verbatim
  (insertion of "od wybuchu II wojny światowej").

**G2 JSON_mirror** greedy:

- Normal: *"The weight of 70 kg is approximately 154.32 pounds."*
- L10+L14_extended: *"The function `calculate_weight` takes a single argument
  `weight_in_kg`, which represents the weight in kilograms. Here's an example
  of how you can use the func…"* — model now *describes* the function in
  English prose instead of producing the numerical conversion.

**G4 baseline**:

- Normal: *"Tak, stolica Francji to Paryż."*
- L10+L14_extended T=0.3: *"Tak, to prawda. Paryż jest stolicą Francji i
  jest jednym z najbardziej znanych miast na świecie. Jest to miasto pełne
  zabytków, kultury i historii."* — the model produces a longer elaboration
  rather than the short confirmation.

### Interpretation (factual)

- A 9-head ablation (≈2.3% of attention capacity) is needed to noticeably
  shift the output of every case in our small sample.
- Even at this scale, **total copy_score does not drop** — it grows. Copying
  in this network is not a localized property of a few heads; it is a
  distributed circuit with multiple layers of redundancy.
- What ablation *does* change is the **type of copy**: from verbatim copy
  of specific tokens (dates, numbers, code structure) to paraphrase,
  elaboration, or translation into a different mode (Polish prompt → English
  description).
- L10 → L14 → L13/L15/L11 is a chain of progressively recruited reserves.
  The full picture is consistent with the literature on induction circuits
  in transformers: **copying is implemented by a circuit, not by a head**.

### Visualization

`raport_assets/exp33_cluster_ablation.json` — outputs for 4 cases × 4
ablation configs × 2 sampling modes, plus per-case total_copy_score before
and after each ablation. No standalone heatmap was generated for this step;
the per-config delta heatmaps from Step 9 already cover the spatial
distribution of changes.
