# Theorems — Behavioural Rate–Distortion for Quantized Language Models

**Date:** 2026-08-02
**Status:** formalisation for the pivoted direction (`docs/pivot_plan_2026-08.md`). Supersedes the
FPR-Decoupling Theorem in `docs/math.md`, which was adjudicated near-tautological by all three
panel reviewers (`docs/theory_panel_2026-08.md` §5).
**Numerical verification:** every closed form below was checked against direct numerical solution;
results are quoted inline.

---

## 0. Honest preface — what is and is not novel

Individually, almost nothing here is new mathematics. Theorem 1 is signal-detection algebra.
Theorem 4 is the ROC-invariance fact. Theorem 5 is the data-processing inequality. Theorem 6 is
linear algebra. The triage in `docs/theory_panel_2026-08.md` §3 classified each and found **no
single statement in category (iii)**.

The contribution is the **package**, and specifically three properties of it:

1. **The assumptions are measurable** on a 6 GB GPU, and we report the measurements rather than
   assuming them (A1 via `eval/isotropy.py`, A2 via `gaussianity_gap`, A3 by fitting rather than
   assuming the exponent).
2. **η is measurable without behavioural labels** — from weights and benign activations only. This
   makes the law *predictive*, not descriptive, which is what separates it from the existing
   empirical quantization scaling laws (arXiv:2508.18609 fits a smooth power law by non-linear
   least squares; it has no mechanism and predicts no collapse point).
3. **It makes a falsifiable shape claim.** A smooth power law in `log(bit-width)` structurally
   cannot produce a cliff. Theorem 2 says the cliff exists and is a threshold-crossing artifact,
   and predicts *where*. Those two claims are distinguishable on data.

Anything below marked **(i)** must be presented as a lemma or scope statement, never as a
contribution. Presenting Theorem 6 as a "Safety Restoration Theorem" would repeat exactly the
mistake that sank the FPR-Decoupling Theorem.

---

## 1. Setup and assumptions

Let $M$ be a model, $\ell$ a layer, and $z(x)\in\mathbb{R}^{D}$ the residual stream at the readout
token for prompt $x$. Let $q$ be a post-training quantization scheme mapping weights
$W \mapsto W_q$, with perturbation $E_q = W - W_q$, and write the induced activation perturbation

$$\varepsilon_q(x) \;=\; z_q(x) - z(x).$$

A **behavioural readout** is a unit direction $\hat r\in\mathbb{R}^{D}$ (or a rank-$k$ subspace
$U$) with margin $m(x)=\langle z(x),\hat r\rangle$, together with a **composition rule** $C$
describing how per-instance decisions aggregate into the observed behaviour.

> **A1 (Isotropic projected perturbation).** $\mathbb{E}[\langle\varepsilon_q(x),\hat r\rangle]=0$
> and $\operatorname{Var}[\langle\varepsilon_q(x),\hat r\rangle]=\sigma_q^{2}$, with $\sigma_q^{2}$
> not depending on the choice of $\hat r$ — i.e. the perturbation carries no preferred alignment
> with the behavioural subspace.
> *Testable:* `eval/isotropy.py` (concentration $z$-scores against a matched-magnitude isotropic
> null; excess kurtosis of the perturbation vs the direction).
> *Current evidence:* on Llama-3.2-3B FP16→NF4 the perturbation has excess kurtosis $0.20$ against
> a direction with excess kurtosis $19.34$, and concentration $z\in[1.7,1.9]$ — consistent with
> isotropy. **Gated on Stage 0** (defect D2): sampling noise is itself isotropic, so this is not yet
> established.

> **A2 (Equal-variance Gaussian margins).** Class-conditional margins are Gaussian with a common
> variance $s^{2}$: $m\mid\text{neg}\sim\mathcal N(\mu_-,s^2)$, $m\mid\text{pos}\sim\mathcal N(\mu_+,s^2)$.
> *Testable:* `gaussianity_gap` (|empirical AUC − Gaussian-model AUC|). Values $\gtrsim 0.05$
> invalidate the closed-form predictions. A heavy-tail-contaminated case measured a gap of $0.22$
> ($d'=0.25$, model AUC $0.57$, empirical AUC $0.79$) — this failure mode is real and must be
> screened for, not assumed away.

> **A3 (Exponential noise growth).** $\sigma_q^{2}\propto \beta^{-b}$ for effective bit-width $b$,
> with $\beta=4$ for uniform non-saturating scalar quantization.
> **A3 is fitted, never assumed.** GGUF k-quants use super-block scales, group quantization and
> outlier handling, any of which can break it. `eval/noise_spectrum.fit_eta_vs_bits` fits both the
> coefficient and the base, and reports a confidence interval on the exponent.

Define the **noise-to-signal ratio on the behavioural subspace**

$$\eta_q \;=\; \frac{\sigma_q^{2}}{s^{2}}.$$

---

## 2. Theorem 1 — Discriminability decay

> **Theorem 1.** Under A1 and A2,
> $$d'(q) \;=\; d'_{0}\,\bigl(1+\eta_q\bigr)^{-1/2}, \qquad d'_0 = \frac{\mu_+-\mu_-}{s}.$$

**Proof.** The quantized margin is $m_q(x)=\langle z(x)+\varepsilon_q(x),\hat r\rangle
= m(x)+\langle\varepsilon_q(x),\hat r\rangle$. By A1 the additive term has mean $0$ and variance
$\sigma_q^2$ and is independent of class, so the class means are unchanged and each class variance
becomes $s^{2}+\sigma_q^{2}$. Hence
$d'(q)=(\mu_+-\mu_-)/\sqrt{s^{2}+\sigma_q^{2}} = d'_0\,s/\sqrt{s^{2}+\sigma_q^{2}}
= d'_0(1+\eta_q)^{-1/2}$. $\blacksquare$

**Class: (ii)** — true, but the content is entirely in A1/A2. Implemented as
`discriminability.predict_d_prime`; the inverse `implied_eta` lets a behaviourally-measured $\eta$
be compared against a weights-measured one. **That comparison is the core validation: agreement
supports the mechanism, disagreement falsifies it.**

### Corollary 1.1 — Threshold composition
At a fixed operating false-positive rate $\alpha$,
$$\mathrm{TPR}(q)=\Phi\!\bigl(d'(q)-z_{1-\alpha}\bigr).$$

### Corollary 1.2 — Chain composition (the reasoning law)
For a $T$-step conjunction with per-step success $p(\eta)=\Phi(d'(\eta)-z)$, chain success is
$P_T(\eta)=p(\eta)^{T}$. Fixing a constant *relative* retention $c$ (an absolute bar would be
unreachable for large $T$ from ceiling effects alone) and linearising $\log p$ about $\eta=0$,

$$\frac{d\log p}{d\eta}\Big|_{0} = -\kappa_1,
\qquad \kappa_1=\frac{d'_0}{2}\cdot\frac{\varphi(d'_0-z)}{\Phi(d'_0-z)},$$

$$\eta^{*}_T=\frac{\log(1/c)}{T\,\kappa_1}
\qquad\Longrightarrow\qquad
\boxed{\;b^{*}(T)=b^{*}(1)+\log_{\beta}T + O(1)\;}$$

For $\beta=4$: **each 4× increase in chain length costs exactly one additional bit.**

*Verification.* $\kappa_1$ closed form $=0.430424$ against numerical derivative $0.430424$ (exact).
Solving $P_T$ directly without linearisation, with $d'_0=2.5$, $\alpha=0.05$, $\eta_4=0.30$,
$c=0.5$:

| $T$ | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| $b^{*}(T)$ | 2.561 | 3.213 | 3.765 | 4.281 | 4.786 | 5.287 | 5.788 |

The additive offset relative to $b^{*}(1)+\log_4 T$ converges to $\approx +0.227$ — hence the
$O(1)$ term. **The exact statement is in the increments:** $b^{*}(4T)-b^{*}(T) \to 1.000$ bit
(measured $1.021, 1.006, 1.002$ for $T=4\!\to\!16$, $8\!\to\!32$, $16\!\to\!64$).

> **Corollary 1.2a (width invariance).** The transition width in $\eta$ scales as $1/(T\kappa_1)$,
> but $d\eta/db\propto\eta(b)$ and $\eta^{*}_T\propto 1/T$, so the two factors of $T$ cancel: the
> reasoning cliff is **displaced to higher bit-width by $\log_\beta T$ but is not sharper**. This
> contradicts the natural intuition that longer chains cliff more abruptly, and is therefore a
> useful falsifier.

**Class: (ii)/(iii) boundary.** $P_T=p^{T}$ alone is textbook series-system reliability and must
**never** be headlined. The claim is the coupling to the noise model. Assumes conditionally
independent, comparable steps — false for real chain-of-thought, which self-corrects and has
correlated errors. Report where it bends.

### Corollary 1.3 — Tail composition (rare failures)
For a failure at threshold $\Delta$ (in units of $s$) with baseline rate $p_0=\Phi(-\Delta)$,
$$p_q=\Phi\!\left(-\frac{\Delta}{\sqrt{1+\eta}}\right),
\qquad
\boxed{\;\frac{\log p_q}{\log p_0}\;\longrightarrow\;\frac{1}{1+\eta}\;}$$

**Proof.** $\log\Phi(-t)\sim -t^{2}/2$ as $t\to\infty$. Substituting $t\mapsto t/\sqrt{1+\eta}$
scales the exponent by $1/(1+\eta)$. $\blacksquare$

*Verification.* For $\eta=0.25$ (target $0.800$) the ratio runs $0.864, 0.839, 0.823, 0.811$ at
$p_0=10^{-2},10^{-4},10^{-8},10^{-16}$ — converging as claimed.

Equivalently $R(\eta)=p_q/p_0\approx\sqrt{1+\eta}\,\exp\!\bigl[(\Delta^{2}/2)\,\eta/(1+\eta)\bigr]$.
Measured inflation at $\eta=0.5$: $\times1.5$ at $p_0=10^{-1}$, rising to $\times24.9$ at $10^{-5}$.

**This is the mechanistic explanation of "quality is not a safety proxy."** Perplexity and MMLU are
*bulk* statistics; safety and security failures are *tail* statistics. The rarer the event, the
more violently it inflates at the same $\eta$.

**Class: (i) for the Mills-ratio algebra, (ii) in application.** *Scope limit:* requires genuine
tail depth ($\Delta\gtrsim 2\text{–}3$). The project's own safety numbers (TPR $0.80\to0.27$) are
**not** in that regime — they are fully explained by Theorem 1 with no tail asymptotics. Conflating
the two would dress one phenomenon as two theorems.

---

## 3. Theorem 2 — The cliff is a threshold-crossing artifact

> **Theorem 2.** Under A1–A3, an observed behaviour $B(b)$ is the composition of a **smooth,
> strictly monotone** $d'(b)$ with a **sigmoidal** readout. $d'(b)$ has no discontinuity in $b$;
> the apparent cliff is the sigmoid's transition, located at
> $$b^{*} \;=\; 4-\log_{\beta}\!\left(\frac{(d'_0/z_{1-\alpha})^{2}-1}{\eta_4}\right).$$

**Proof.** $d'(b)=d'_0(1+\eta_4\beta^{4-b})^{-1/2}$ is smooth and strictly increasing in $b$. By
Corollary 1.1 the behaviour is $\Phi(d'(b)-z_{1-\alpha})$, whose steepest descent is where the
argument crosses zero, i.e. $d'(b^{*})=z_{1-\alpha}$. Solving for $b^{*}$ gives the display.
$\blacksquare$

With $d'_0=2.5$, $\alpha=0.05$, $\eta_4=0.30$: $b^{*}=2.94$ bits — i.e. **Q3_K_M**, which is where
the blueprint pre-registered the cliff and where the literature reports it. **The theory derives
the pre-registered location instead of assuming it.**

**Class: (iii) candidate**, and the paper's sharpest differentiator. It is a claim about *shape*,
directly opposed to the smooth-power-law form of arXiv:2508.18609, and therefore decidable on data.

### Corollary 2.1 — Ordering across domains
Collapse bit-width increases with chain length $T$ and with rarity (decreasing $p_0$). Under the
worked constants this predicts the order in which sectors break as precision falls:

| Sector | $b^{*}$ |
|---|---|
| Reasoning, $T=64$ | 5.27 |
| Security, $p_0=10^{-5}$ | 5.11 |
| Security, $p_0=10^{-3}$ | 4.71 |
| Reasoning, $T=16$ | 4.31 |
| Reasoning, $T=4$ | 3.39 |
| Safety, single-shot | 2.56 |

**This ordering is the headline testable prediction:** one GGUF ladder, three sectors, check the order.

---

## 4. Theorem 3 — Isotropy is a property of the quantizer

> **Theorem 3 (conditional).** A1 is a property of the *quantization algorithm*, not of
> quantization per se. Algorithms minimising unweighted reconstruction error (RTN, NF4, vanilla
> GGUF k-quants) have no term coupling $E_q$ to any behavioural direction, so A1 is plausible and
> Theorems 1–2 apply. Activation-aware algorithms (AWQ, GPTQ) explicitly protect high-salience
> channels, inducing $\mathbb{E}\langle\varepsilon_q,\hat r\rangle\neq 0$ or direction-dependent
> $\sigma_q^{2}$ — violating A1, so **no cliff is predicted**.

**Status: conjecture with a decisive cheap test**, not a proved theorem. It is stated because it
*resolves a live contradiction*: arXiv:2606.10154 reports refusal falls of 12–68 pp, while
arXiv:2606.29581 (8 models, 5 families, 144 configurations, ~2.0M scored responses, six-judge
ensemble) finds AWQ INT4 stays within ~1.6 pp of FP16 for 7 of 8 models. Under Theorem 3 both are
correct and the disagreement is explained by quantizer family.

*Test:* run `eval/isotropy.py` on an AWQ/GPTQ checkpoint of the same model. Prediction: NF4/GGUF
isotropic (small $|z|$), AWQ/GPTQ anisotropic (large $|z|$). Decisive either way — if AWQ is also
isotropic, the literature split must come from judges or corpora instead, which is equally worth
knowing.

---

## 5. Theorems 4–5 — What cannot be undone

> **Theorem 4 (Recalibration invariance).** For any strictly increasing $g$, the ROC of $g\circ m$
> equals the ROC of $m$; hence $d'$ is unchanged. **No threshold or affine recalibration can
> restore discriminability lost to quantization.**

**Proof.** $\{x: m(x)\ge\tau\}=\{x: g(m(x))\ge g(\tau)\}$, so every threshold on $m$ has an image
threshold on $g\circ m$ yielding the identical $(\mathrm{FPR},\mathrm{TPR})$ pair. As $\tau$ ranges
over $\mathbb{R}$ both curves trace the same set. $\blacksquare$

**Class: (i).** Stated because it forecloses a specific wrong hope: fixing defect D0 (the inverted
calibration tail) is necessary but **cannot** recover cliff-induced TPR loss. Calibration bugs and
discriminability loss are different failures with different fixes.

*Empirical corroboration.* Decomposing the measured FP16→NF4 damage into components parallel
(correctable bias) and orthogonal (input-dependent) to $\hat r$: $-0.0279$ vs $0.2345$ — a ratio of
$8.4$, so **$99\%$ of the damage sits in the component recalibration cannot touch.**

> **Theorem 5 (Post-hoc impossibility).** With the Markov chain $Y\to z(x)\to z_q(x)$,
> $I(Y;z_q)\le I(Y;z)$. Moreover, two distinct FP16 weight matrices lying in the same quantization
> cell map to identical $W_q$; hence no function of $W_q$ alone can distinguish their behaviours,
> and **uniform exact recovery from a frozen quantized checkpoint is impossible.**

**Class: (i)** — standard DPI plus a collision argument. Essential as a scope boundary so that
Theorem 6 is not misread as beating information theory.

*Corollary (deterministic error).* For fixed input and fixed quantized weights, $z_q(x)$ is
**deterministic** — same input, same error, every time. Sampling randomises only final token
selection, downstream of the corrupted representation. **Therefore self-consistency and repeated
sampling of the same prompt cannot average $\varepsilon_q$ away**; there is nothing stochastic to
average. (A $d'\sqrt{T}$ recovery conjecture was proposed and refuted on this basis.)

---

## 6. Theorems 6–7 — What can be recovered, and at what cost

> **Theorem 6 (Sidecar achievability).** Theorem 5 binds only when the corrector sees $z_q(x)$
> alone. Augmenting it with $\Delta W = W-W_q$ (free at build time) and the layer input $a(x)$
> (already computed in the forward pass) escapes the bottleneck. For a linear layer feeding the
> readout,
> $$\langle\varepsilon_q(x),\hat r\rangle=\hat r^{\top}(W-W_q)\,a(x)=v\cdot a(x),
> \qquad v:=\hat r^{\top}(W-W_q),$$
> so $m_{\text{corr}}(x)=m_q(x)+v\cdot a(x)$ restores that layer's contribution **exactly**.

$v$ is a fixed, precomputable row vector: one dot product at inference, no training. Storage is
$O(k\,d_{\text{in}}L)$ — for $k=1$, $d=3072$, $L\approx30$ that is $\approx$ **180 KB**, negligible
against a multi-billion-parameter model.

**Class: (i)** as linear algebra. Its value is as a **mechanism test**, not a product: Q-resafe
(ICML 2025, arXiv:2506.20251), Q-realign (arXiv:2601.08089) and CAQ (arXiv:2511.07842) already
occupy the practical safety-restoration niche with learned, weight-updating methods. Do not compete
on ASR numbers. **Report the recovery fraction against a random-direction matched-norm control.**

> **Theorem 7 (Behavioural disagreement bound).** If the corrected margin satisfies
> $|m_{\text{corr}}(x)-m(x)|\le\epsilon$ for all $x$, then the decision-disagreement rate against
> full precision at threshold $\tau$ obeys
> $$\Pr[\text{disagree}] \;\le\; F(\tau+\epsilon)-F(\tau-\epsilon)\;\approx\;2\epsilon f(\tau),$$
> where $F,f$ are the FP16 margin CDF and density.

**Proof.** A decision flips only if $m$ and $m_{\text{corr}}$ straddle $\tau$, which requires
$|m(x)-\tau|\le\epsilon$. $\blacksquare$

*Verification* ($m\sim\mathcal N(0,1)$, $\tau=0.7$, uniform error on $[-\epsilon,\epsilon]$):

| $\epsilon$ | disagreement | bound | $2\epsilon f(\tau)$ |
|---|---|---|---|
| 0.001 | 0.00018 | 0.00062 | 0.00062 |
| 0.01 | 0.00158 | 0.00625 | 0.00625 |
| 0.05 | 0.00778 | 0.03122 | 0.03123 |
| 0.2 | 0.03177 | 0.12448 | 0.12490 |

Bound holds throughout, conservative by $\approx4\times$ for uniform error (worst-case tight).

**Class: (ii), possibly a novel synthesis** — it is the step that converts a *representation*
guarantee into a *behavioural* one, which is what makes a rate–distortion statement meaningful for
a language model rather than for a matrix.

> **Conjecture 8 (Behavioural rate–distortion).** For a causally sufficient $k$-dimensional
> behavioural statistic, bounded activations and Lipschitz downstream score, a quantized sidecar of
> $\Theta\!\bigl(k\,d\,L\log(R/\epsilon)\bigr)$ bits suffices to hold decision disagreement below
> $F(\tau+\epsilon)-F(\tau-\epsilon)$, and no method using $W_q$ alone attains it.

Achievability follows from Theorems 6–7 plus a covering argument. **The converse — that no scheme
does better — is open** and needs a genuine Wyner–Ziv-style rate-distortion argument. State it as a
conjecture and an appendix direction; **do not claim it.** It is not attemptable inside a 3–6 month
empirical timeline.

---

## 7. What would falsify this

| # | Falsifier | Consequence |
|---|---|---|
| F1 | Split-half / paired control shows the FP16→NF4 rotation is inside the noise floor | A1 unproven; §2–§3 become conditional |
| F2 | `gaussianity_gap` $\gtrsim 0.05$ on real margins | A2 fails; closed-form TPR predictions void (Theorem 1 survives for $d'$ only) |
| F3 | Fitted exponent CI excludes $\beta=4$ | A3 fails for k-quants; Theorem 2's $b^{*}$ formula needs the measured base |
| F4 | AWQ perturbation is *also* isotropic | Theorem 3 refuted; literature split must be explained otherwise |
| F5 | Measured $\eta$ from weights disagrees with $\eta$ implied by $d'$ decay | Theorem 1's mechanism refuted — the central validation |
| F6 | $b^{*}(4T)-b^{*}(T)\neq 1$ bit | Corollary 1.2 refuted (likely via correlated reasoning steps) |
| F7 | Sidecar recovery indistinguishable from a random-direction control | Theorem 6's causal-sufficiency premise fails for this readout |
| F8 | Sector ordering (Cor. 2.1) does not hold on a real ladder | Composition framework refuted |

Every one of these is runnable on an RTX 3050 + Colab. **F1 gates all the others.**

---

## 8. Verdicts from the first full ladder (2026-08-03)

Run `artifacts/runs/20260803-075641_dafdc44_local-ladder-rtn-qwen1.5b` — Qwen2.5-1.5B-Instruct,
layer 14, $n=250$/class, per-group asymmetric round-to-nearest at 8→2 bits, group 64.
Full write-up: [`results_local_ladder.md`](results_local_ladder.md).

| # | Verdict | Evidence |
|---|---|---|
| **F1** | **Does not fire.** A1 stands. | Rotation replicates at 8/8 rungs across disjoint prompt halves, $z=15$–$26$ against a chance SD of $1/\sqrt{1536}$ |
| **F2** | **Does not fire.** A2 stands. | Every `gaussianity_gap` $\le 0.005$; the equal-variance Gaussian ROC model fits |
| **F3** | **FIRES.** | Fitted base $\hat\beta = 4.355$, 95 % CI $[4.114,\,4.611]$, $R^2 = 0.9989$, $n=7$. The interval excludes 4 |
| **F4** | Not yet decided. | RTN and NF4 are themselves **not** isotropic (null rejected at 7/8 rungs), which contradicts Theorem 3's premise before AWQ is even reached |
| **F5** | **FIRES, decisively.** | $\eta(\text{weights})/\eta(d'\text{ decay})$ spans $374\times$ and falls monotonically from 64.7 to 0.173 |

### What survives

**Assumption A3 survives in form and dies in its constant.** $\eta$ is exponential in bit-width to
three digits of $R^2$; the base is simply not 4. Everywhere Theorem 2 writes $4^{4-b}$ it should
write $\hat\beta^{\,4-b}$ with $\hat\beta$ measured per (model, quantizer). The closed form

$$b^{*} \;=\; 4 - \log_{\hat\beta}\!\left(\frac{(d'_0/z_{1-\alpha})^2 - 1}{\eta_4}\right)$$

is unchanged in structure — `collapse_bits_threshold_closed_form` already takes `base` as an
argument, so no code changes, only a discipline: **never pass the default.**

The interval is **model-conditional**. The seven rungs share a checkpoint, a direction, and an
activation sample, so the OLS residuals are not independent. "Excludes 4" is a diagnostic, not a
pre-registered rejection; a defensible claim needs repeated estimates and a covariance-aware or
synchronised block bootstrap.

### What breaks

**Theorem 1's mechanism does not hold on this model.** The theorem asserts that quantization
inflates variance along the behavioural direction and that this inflation *is* the degradation:
$d'(q) = d'_0(1+\eta_q)^{-1/2}$. Measured: $\eta$ grows by a factor of 7400 from 8 bits to 2 bits
while held-out $d'$ moves from $0.4129$ to $0.3914$ — a fifth of one standard deviation.

The failure is not a loose constant. Rearranging the theorem, the implied $\eta$ must satisfy
$\eta_{\text{beh}} = (d'_0/d'_q)^2 - 1$. Comparing it rung by rung against the weight-space
measurement gives a ratio that is not constant but **monotone decreasing across two orders of
magnitude**. That shape is itself informative: the map from weight-space noise to margin-space
noise is strongly **compressive**, and increasingly so as the perturbation grows.

A compressive transfer is what one would expect if the readout renormalises away the inflation. The
margin used here, $m(x) = \langle a(x), \hat r\rangle / \lVert a(x)\rVert$, is norm-normalised,
which would make it *structurally* insensitive to exactly the isotropic inflation Theorem 1 is
built on.

**That explanation was tested and rejected.** `scripts/analyse_margin_normalisation.py` recomputes
held-out $d'$ from the same saved activations using the raw projection
$\langle a(x), \hat r\rangle$ with no normalisation. Both readouts are flat:

| | FP16 | RTN 2-bit | drop |
|---|---|---|---|
| normalised $\langle a,\hat r\rangle/\lVert a\rVert$ | 0.4129 | 0.3914 | $+0.25\,\mathrm{sd}$ |
| raw $\langle a,\hat r\rangle$ | 0.4071 | 0.3909 | $+0.19\,\mathrm{sd}$ |

The normalisation is exonerated; $\eta$ fails to reach the behavioural readout by either route.

The same table contains the more interesting number. At 2 bits the mean activation norm inflates by
$1.86\times$ and the **mean projection onto the direction goes negative** — $-3.87\times$ its FP16
value, i.e. the readout direction has reoriented past orthogonal — and $d'$ is *still* $0.39$.
Separability survives a coordinate system that has effectively reversed.

That reframes the result. The quantized model has not lost the harmful/benign distinction; it has
**moved where the distinction lives**, and a difference-in-means direction refitted on the quantized
model finds it again at full strength. Theorem 1 measures the FP16 direction's *continued validity*,
which decays fast, and mistakes it for the *information*, which does not decay at all. The two are
different quantities, and conflating them is the actual error.

**Theorem 1 should therefore be restated with an explicit transfer function**

$$d'(q) \;=\; \frac{d'_0}{\sqrt{1 + g(\eta_q)}}, \qquad g(0)=0,\; g'>0,$$

where the original theorem is the special case $g = \mathrm{id}$. The data reject $g=\mathrm{id}$
and constrain $g$ to be concave over the measured range. Measuring $g$ — not assuming it — is the
next theorem, and it is a more interesting one than the assumed-identity version, because $g$ is
where the model's own robustness lives.

### The caveat that limits all of the above

$d'_0 = 0.413$ is a **label ceiling**, not a property of the model. "Refused" labels the hh-rlhf
*rejected response*, not this model's behaviour. Two consequences. First, $b^{*}$ is undefined
because $d'_0 < z_{0.95} = 1.645$: the full-precision system is already below the operating
threshold the collapse bound is stated against. Second, and more seriously, a probe with little
discriminability to begin with has little to lose, so the flat $d'$ curve is consistent both with
"quantization does not degrade this behaviour" and with "this probe cannot see the degradation."

Separating those two is the single highest-value experiment remaining, and it needs labels derived
from the target model's own completions. Until then, F5's verdict should be stated as *the
mechanism is refuted for this readout*, not *for this behaviour*.

---

*Companion code: `eval/discriminability.py` (Thm 1, Cor 1.1), `eval/isotropy.py` (A1, Thm 3),
`eval/noise_spectrum.py` (A3, η), `eval/noise_floor.py` (F1), `scripts/run_local_ladder.py`
(the full F1–F5 ladder). Composition rules (Cor 1.2/1.3) and the sidecar (Thm 6–7) are not yet
implemented — Stages 4–6 of `docs/pivot_plan_2026-08.md`.*
