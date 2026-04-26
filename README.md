# Bielik-1.5B-v3.0-Instruct — attention head walkthrough

A step-by-step interpretability investigation into how **Bielik-1.5B-v3.0-Instruct**
produces three different kinds of repetitive output:

- **JSON tool-calls** generated for prompts that are not requests for tools
- **Verbatim mirroring** of long factual sentences
- **Help-offer attractors** (role reversal when the user speaks in assistant voice)

The report walks through 10 experimental steps, from per-region attention
analysis through PCA of residual streams, identification of "echoing heads",
single- and multi-head ablations, takeover analysis, and final cluster ablation.

## How to view

Open **[index.html](https://InezOk.github.io/Bielik_echo/)** in a browser.

The HTML is fully self-contained except for image links to the
`raport_assets/attention_loops/` folder. The standalone markdown source is
in [`raport_attention_heads_walkthrough.md`](raport_attention_heads_walkthrough.md).

## What's inside

- `index.html` — the rendered walkthrough (10 steps + figure gallery)
- `raport_attention_heads_walkthrough.md` — markdown source of the same report
- `raport_assets/attention_loops/` — 23 figures (PNG, SVG)
- `raport_assets/*.json` — raw numerical data used in each step (top heads,
  copy_score matrices, ablation outputs, pivot logits, etc.)

## Setup notes

All experiments use HuggingFace `transformers`, `attn_implementation="eager"`,
`torch_dtype=torch.float32` on CPU. The model has 32 transformer layers ×
12 attention heads = 384 heads total, hidden_size 1536, head_dim 128.

Selected examples are in Polish (matching the original analysis); the
explanatory text and tables are in English.
