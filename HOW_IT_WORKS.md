# How CSD-BMIA Works

A one-glance guide to what this project does and how the code fits together.

---

## The problem in one line

When a trained model meets corrupted images (noise, blur, weather…), popular
test-time adaptation methods **collapse** — they start predicting the *same
class* for everything and accuracy crashes.

```mermaid
flowchart LR
    A[Corrupted images] --> B[Entropy-minimization TTA]
    B --> C{High learning rate?}
    C -- yes --> D[COLLAPSE: predicts one class<br/>accuracy to near 0]
    C -- no --> E[Works, but fragile]
    style D fill:#ffd6d6,stroke:#c0392b,color:#000
```

---

## Our idea: maximize Mutual Information instead of minimizing entropy

Collapse is the *global minimum* of entropy. It is **not** a minimum of Mutual
Information — MI stays a full `log K` away from collapse. So an MI objective can
never be pulled into collapse.

```mermaid
flowchart LR
    A[Corrupted batch] --> B[Frozen backbone<br/>only BatchNorm affine trainable]
    B --> C["Loss = H(Y|X) − H(Y) + λ·‖θ − θ₀‖²"]
    C --> D[MI term:<br/>confident BUT diverse predictions]
    C --> E[Anchor term:<br/>stay near source weights θ₀]
    D & E --> F[ORCA monitors dom%<br/>adjusts λ every batch]
    F --> G[Adapted model<br/>zero collapse]
    style G fill:#d6f5d6,stroke:#27ae60,color:#000
```

---

## The adaptation loop (what runs each batch)

```mermaid
flowchart TD
    S[Start: load model, save θ₀] --> L{next batch?}
    L -- yes --> F[forward pass]
    F --> M[compute MI loss + anchor]
    M --> BK[backward + update BN affine only]
    BK --> O[measure dom% = most-predicted class share]
    O --> D{dom% > τ ?}
    D -- yes, collapse risk --> UP[λ ← λ × 1.10  tighten anchor]
    D -- no, stable --> DN[λ ← λ × 0.95  relax anchor]
    UP --> L
    DN --> L
    L -- no --> R[report accuracy, MI, collapse]
```

**In plain words:**
1. Freeze the whole network except the BatchNorm scale/shift parameters.
2. For each incoming batch, nudge those parameters to **maximize mutual
   information** — predictions become confident *and* spread across classes.
3. An **anchor** keeps the parameters close to the original model.
4. **ORCA** watches how concentrated the predictions are and tightens or
   relaxes the anchor automatically — no manual tuning.
5. Result: the model adapts to the corruption **without ever collapsing**.

---

## What's in this repository

```mermaid
flowchart TD
    ROOT[CSD-BMIA] --> RM[README.md — overview]
    ROOT --> HW[HOW_IT_WORKS.md — this file]
    ROOT --> TH[theory/ — theorem + proofs]
    ROOT --> EX[experiments/ — 20 notebooks]
    EX --> E1[imagenet_c_* — 1000-class benchmark]
    EX --> E2[cifar100c_* — main comparison]
    EX --> E3[eata_collapse_analysis — why baselines fail]
    EX --> E4[medmnist / tinyimagenet / vit — generalization]
    EX --> E5[batch_size / multiseed / threshold — ablations]
```

Each notebook is self-contained: **Cell 1** installs dependencies, **Cell 2**
runs the experiment and prints a results table.

---

## The one-sentence takeaway

> Replace entropy minimization with **mutual-information maximization + an
> adaptive anchor**, and test-time adaptation becomes **collapse-free** — even
> at high learning rates and small batches where every other method breaks.

Author: **Anish Kumar Thakur** — Alumni 2025, Dept. of CSE, ABV-IIITM (2026).
