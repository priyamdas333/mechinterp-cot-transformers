# When Do Transformers Actually Reason?
### A Mechanistic Study of Chain-of-Thought in Small Models

> **Core question:** Does chain-of-thought training change a transformer's *internal computation*, or does it just improve accuracy through pattern matching?

> **Key finding:** Removing intermediate reasoning tokens degrades CoT model accuracy by ~65%, establishing causal necessity. Logit lens reveals gradual answer buildup in CoT vs. sudden final-layer jump in direct models. We propose a **hybrid computation hypothesis** — CoT induces both explicit token-based reasoning and implicit latent compression.

---

## Overview

Most chain-of-thought research measures whether CoT improves accuracy. This project asks a deeper question: **what happens inside the model?**

We fine-tune GPT-2 on two-digit addition under two conditions — direct answer and chain-of-thought with explicit carry reasoning — then apply four mechanistic analyses to determine whether reasoning tokens are causally necessary or merely decorative.

```
Direct:  23 + 48 = 71
CoT:     23 + 48 = 3+8=11, write 1 carry 1; 2+4+1=7 → 71
```

### Analyses Performed

| Analysis | Question | Method |
|----------|----------|--------|
| **Causal Interventions** | Are reasoning tokens necessary? | Token removal, shuffling, corruption, partial ablation |
| **Logit Lens** | Does the answer build gradually? | Project intermediate hidden states to vocabulary |
| **Attention Analysis** | Where do answer tokens look? | Categorize tokens, measure attention flow |
| **Representation Analysis** | Do reasoning tokens cluster distinctly? | PCA of hidden states by token category |

---

## Results

<p align="center">
  <img src="figures/fig2_causal_interventions.png" width="700" alt="Causal intervention results"/>
</p>

| Intervention | CoT Accuracy |
|:------------|:------------|
| Full CoT (intact) | High |
| Reasoning tokens removed | **Dramatic drop (~65%)** |
| Reasoning tokens shuffled | Significant drop |
| Carry step removed | Largest single-step drop |
| Direct model (no CoT) | Moderate |

**Interpretation:** Reasoning tokens are causally necessary — the model cannot bypass them. The carry sub-step is the most critical component, which aligns with it being the algorithmic bottleneck (information propagation from ones to tens column).

<p align="center">
  <img src="figures/fig6_summary_dashboard.png" width="750" alt="Summary dashboard"/>
</p>

---

## Project Structure

```
├── run.py                          # Full pipeline: train → evaluate → analyze → visualize
├── requirements.txt
├── LICENSE
│
├── src/                            # Core modules
│   ├── data_generation.py          # Synthetic arithmetic data (direct + CoT)
│   ├── dataset.py                  # Tokenization + HuggingFace dataset prep
│   ├── training.py                 # LossTrackingTrainer + training loop
│   └── evaluation.py               # Autoregressive generation + exact-match accuracy
│
├── analysis/                       # Mechanistic analysis modules
│   ├── attention_analysis.py       # Token categorization + attention flow measurement
│   ├── logit_lens.py               # Layer-wise prediction probing
│   ├── causal_interventions.py     # 5 intervention types + embedding corruption
│   ├── representation_analysis.py  # PCA of hidden states by token category
│   └── visualize.py                # All plotting functions
│
├── notebooks/
│   └── CoT_Mechanistic_Study.ipynb # Complete Colab notebook (runs end-to-end)
│
├── figures/                        # Generated figures
│   ├── fig1_training_curves.png
│   ├── fig2_causal_interventions.png
│   ├── fig3_attention_patterns.png
│   ├── fig4_logit_lens.png
│   ├── fig5_partial_ablation.png
│   └── fig6_summary_dashboard.png
│
└── checkpoints/                    # Saved models (git-ignored, created at runtime)
```

---

## Quick Start

### Option 1: Run on Google Colab (Recommended)

Open `notebooks/CoT_Mechanistic_Study.ipynb` in Google Colab with a T4 GPU. The notebook runs end-to-end in ~30–45 minutes and produces all figures.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Unity333A/cot-mechanistic-study/blob/main/notebooks/CoT_Mechanistic_Study.ipynb)

### Option 2: Run Modular Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python run.py

# With custom settings
python run.py --epochs 3 --batch-size 8 --intervention-samples 100

# Skip training, only run analysis (requires saved models in checkpoints/)
python run.py --analysis-only
```

---

## Technical Details

### Model
- **Base:** GPT-2 (124M parameters, 12 layers, 12 heads, 768-dim)
- **Fine-tuning:** From pretrained `gpt2` checkpoint using HuggingFace Trainer
- **Training:** AdamW (lr=5e-5, weight decay=0.01), 5 epochs, linear warmup

### Dataset
- 10,000 two-digit addition examples (A + B where A, B ∈ [10, 99])
- 90/10 train/test split
- Same problems, two formats (direct vs. CoT)

### Five Causal Interventions
1. **Full removal:** Delete all reasoning tokens → measures overall causal necessity
2. **Shuffling:** Randomize token order → tests whether sequential structure matters
3. **Text corruption:** Replace reasoning digits with random digits → tests value sensitivity
4. **Partial ablation:** Remove carry / ones / tens step individually → identifies critical components
5. **Embedding corruption:** Inject Gaussian noise via PyTorch forward hooks → tests representational reliance

---

## Key Findings

1. **Reasoning tokens are causally necessary.** Removing them causes ~65% accuracy drop — not mere decoration.

2. **Order matters.** Shuffled reasoning performs worse than intact but better than removed — the model uses both content and sequence.

3. **The carry step is most critical.** Partial ablation shows removing the carry causes the largest single-step accuracy drop, aligning with it being the algorithmic bottleneck.

4. **CoT builds answers gradually.** Logit lens shows progressive answer probability increase across layers in CoT, vs. sudden jump in direct model.

5. **Hybrid computation.** CoT induces a combination of explicit sequential reasoning (through generated tokens) and implicit latent compression (in the residual stream), evidenced by converging attention, PCA, and corruption analyses.

---

## References

- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," NeurIPS 2022
- nostalgebraist, "Interpreting GPT: the Logit Lens," 2020
- Elhage et al., "A Mathematical Framework for Transformer Circuits," 2021
- Nanda et al., "Progress Measures for Grokking via Mechanistic Interpretability," ICLR 2023
- Feng et al., "Towards Revealing the Mystery Behind Chain of Thought," NeurIPS 2023
- Lanham et al., "Measuring Faithfulness in Chain-of-Thought Reasoning," 2023

---

## Author

**Priyam Das**
- Email: priyamdas3334@gmail.com
- GitHub: [Unity333A](https://github.com/Unity333A)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
