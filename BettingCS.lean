import Mathlib

/-!
# Ku-Betting-CS — machine-checked algebraic core (P0)

Backs `math-spine.md` §2–§4: the small-bet GROW objective, its exact
projection-form maximizer, and the (1−κ) e-power price identity — at the
inner-product-space level. The probability layer (martingale/Ville validity,
local asymptotics KL ≈ χ²/2) lives on paper in the math-spine; the Ville
argument itself is reused from the existing s=1 Lean bound in the GSA track.

Reading of the objects:
* `z`   — the centered likelihood-ratio deviation `L − 1` in `L²(P₀)`;
* `K`   — the bet subspace `span{φ₁,…,φ_m}` (centered basis functions);
* `w`   — a candidate bet direction (an element of `K`);
* `2⟪z,w⟫ − ‖w‖²` — twice the quadratic (small-bet) GROW objective
  `λᵀβ − ½λᵀFλ` written coordinate-free (`β_k = ⟪z,φ_k⟫` exactly, `F` = Gram);
* `p`   — the orthogonal projection of `z` onto `K` (the optimal bet);
* `κ = ‖p‖²/‖z‖²` — the captured fraction.

Theorem inventory (0 sorry):
1.  `grow_decomp`        — exact decomposition `2⟪z,w⟫ − ‖w‖² = ‖p‖² − ‖p−w‖²`;
2.  `grow_le`            — GROW upper bound `… ≤ ‖p‖²` (sup over the bet space);
3.  `grow_eq_iff`        — the maximizer is unique: equality ⟺ `w = p`;
4.  `sq_norm_decomp`     — Pythagoras `‖z‖² = ‖p‖² + ‖z−p‖²`;
5.  `captured_le`        — `‖p‖² ≤ ‖z‖²` (κ ≤ 1);
6.  `captured_fraction_nonneg` / `captured_fraction_le_one` — `κ ∈ [0,1]`;
7.  `deficit_eq_half_sq_dist` — e-power deficit `½‖z‖² − ½‖p‖² = ½‖z−p‖²`;
8.  `eprice_identity`    — bookkeeping form `½‖z‖² − ½‖p‖² = (1−κ)·½‖z‖²`;
9.  `exists_isProjOn`    — existence of `p` for subspaces with an orthogonal
                           projection (in particular any finite-dimensional `K`);
10. `abs_log_one_add_sub_quad_le` — `|log(1+b) − (b − b²/2)| ≤ |b|³/(1−|b|)`;
11. `abs_log_one_add_sub_quad_le_of_le_half` — `… ≤ 2|b|³` for `|b| ≤ ½`;
12. `quadEnvelope_eq` / `quadEnvelope_pos` — the exact quadratic envelope
    `1+h+h²/2=((h+1)²+1)/2` is strictly positive on the whole real line;
13. `quadEnvelopeFactor_pos` — its variance-compensated e-factor is positive.
14. `quadraticFamily_nonneg` / `quadraticFamily_pos` — the family
    `1+h+c h²` is nonnegative for `c≥1/4` and positive for `c>1/4`;
15. `quadraticFamily_localPenalty` — after matching the second moment, the
    quadratic log-growth penalty is `−v/2`, independently of `c`.
    (the quantified small-bet approximation layer, math-spine §2).

Build procedure (documented in `Ku_Science/README.md`, DSGE Lean host with
prebuilt Mathlib cache, toolchain leanprover/lean4:v4.26.0, mathlib v4.26.0;
NB in this mathlib the real inner-product notation is `⟪x, y⟫` — scoped in
`RealInnerProductSpace` — and the subtype-valued projection is
`K.orthogonalProjection` with `K.starProjection` its `E → E` composite):

    cp BettingCS.lean \
      ../../DSGE/papers/dsge-spectral/lean/DescriptorBasis/BettingCS.lean
    (cd ../../DSGE/papers/dsge-spectral/lean && \
      lake build DescriptorBasis.BettingCS)
    rm ../../DSGE/papers/dsge-spectral/lean/DescriptorBasis/BettingCS.lean

Source of truth: `Ku_Science/Ku_Betting_CS/BettingCS.lean` (this file).
-/

open scoped RealInnerProductSpace

namespace KuBettingCS

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- `p` is the foot of the orthogonal decomposition of `z` on the bet
subspace `K`: it lies in `K` and the residual `z − p` is orthogonal to `K`.
(For finite-dimensional `K` — the only case the paper needs — such `p`
exists and is unique; see `exists_isProjOn`.) -/
def IsProjOn (z p : E) (K : Submodule ℝ E) : Prop :=
  p ∈ K ∧ ∀ u ∈ K, ⟪z - p, u⟫ = 0

/-- Against bets in `K`, `z` and its projection `p` are indistinguishable. -/
theorem inner_eq_of_isProjOn {z p : E} {K : Submodule ℝ E}
    (h : IsProjOn z p K) {w : E} (hw : w ∈ K) :
    ⟪z, w⟫ = ⟪p, w⟫ := by
  have h2 := h.2 w hw
  rw [inner_sub_left] at h2
  linarith

/-- **Exact GROW decomposition.** For any bet `w` in the bet subspace,
`2⟪z,w⟫ − ‖w‖² = ‖p‖² − ‖p − w‖²`: the quadratic growth objective is the
projected mass minus the squared distance to the optimal bet. -/
theorem grow_decomp {z p : E} {K : Submodule ℝ E}
    (h : IsProjOn z p K) {w : E} (hw : w ∈ K) :
    2 * ⟪z, w⟫ - ‖w‖ ^ 2 = ‖p‖ ^ 2 - ‖p - w‖ ^ 2 := by
  rw [inner_eq_of_isProjOn h hw, norm_sub_sq_real]
  ring

/-- **Small-bet GROW optimality (upper bound).** No bet in the subspace can
exceed the projected mass: `2⟪z,w⟫ − ‖w‖² ≤ ‖p‖²`. In coordinates this is
`λᵀβ − ½λᵀFλ ≤ ½βᵀF⁻¹β` — the normal-equations `F·K = B` form of the school. -/
theorem grow_le {z p : E} {K : Submodule ℝ E}
    (h : IsProjOn z p K) {w : E} (hw : w ∈ K) :
    2 * ⟪z, w⟫ - ‖w‖ ^ 2 ≤ ‖p‖ ^ 2 := by
  linarith [grow_decomp h hw, sq_nonneg ‖p - w‖]

/-- **Uniqueness of the optimal bet.** The GROW bound is attained iff the bet
is the projection itself. -/
theorem grow_eq_iff {z p : E} {K : Submodule ℝ E}
    (h : IsProjOn z p K) {w : E} (hw : w ∈ K) :
    2 * ⟪z, w⟫ - ‖w‖ ^ 2 = ‖p‖ ^ 2 ↔ w = p := by
  constructor
  · intro heq
    have hd := grow_decomp h hw
    have h3 : ‖p - w‖ ^ 2 = 0 := by linarith
    have h4 : p - w = 0 := norm_eq_zero.mp (sq_eq_zero_iff.mp h3)
    exact (sub_eq_zero.mp h4).symm
  · intro heq
    subst heq
    have hd := grow_decomp h hw
    simp only [sub_self, norm_zero] at hd
    norm_num at hd
    linarith

/-- **Pythagoras for the bet decomposition.** `‖z‖² = ‖p‖² + ‖z − p‖²`. -/
theorem sq_norm_decomp {z p : E} {K : Submodule ℝ E}
    (h : IsProjOn z p K) :
    ‖z‖ ^ 2 = ‖p‖ ^ 2 + ‖z - p‖ ^ 2 := by
  have hzp : ⟪p, z - p⟫ = 0 := by
    have h2 := h.2 p h.1
    rwa [real_inner_comm] at h2
  have hz : p + (z - p) = z := by abel
  calc ‖z‖ ^ 2 = ‖p + (z - p)‖ ^ 2 := by rw [hz]
    _ = ‖p‖ ^ 2 + 2 * ⟪p, z - p⟫ + ‖z - p‖ ^ 2 := norm_add_sq_real _ _
    _ = ‖p‖ ^ 2 + ‖z - p‖ ^ 2 := by rw [hzp]; ring

/-- The captured mass never exceeds the total: `‖p‖² ≤ ‖z‖²` (i.e. κ ≤ 1). -/
theorem captured_le {z p : E} {K : Submodule ℝ E}
    (h : IsProjOn z p K) :
    ‖p‖ ^ 2 ≤ ‖z‖ ^ 2 := by
  linarith [sq_norm_decomp h, sq_nonneg ‖z - p‖]

omit [InnerProductSpace ℝ E] in
/-- The captured fraction is nonnegative: `0 ≤ κ`. -/
theorem captured_fraction_nonneg (z p : E) :
    0 ≤ ‖p‖ ^ 2 / ‖z‖ ^ 2 := by positivity

/-- The captured fraction is at most one: `κ ≤ 1`. -/
theorem captured_fraction_le_one {z p : E} {K : Submodule ℝ E}
    (h : IsProjOn z p K) (hz : z ≠ 0) :
    ‖p‖ ^ 2 / ‖z‖ ^ 2 ≤ 1 := by
  have hz2 : (0 : ℝ) < ‖z‖ ^ 2 := pow_pos (norm_pos_iff.mpr hz) 2
  rw [div_le_one hz2]
  exact captured_le h

/-- **The e-power deficit is a squared distance.** The gap between the oracle
quadratic growth `½‖z‖²` and the best polynomial growth `½‖p‖²` is exactly
half the squared distance from the LR deviation to the bet space. -/
theorem deficit_eq_half_sq_dist {z p : E} {K : Submodule ℝ E}
    (h : IsProjOn z p K) :
    ‖z‖ ^ 2 / 2 - ‖p‖ ^ 2 / 2 = ‖z - p‖ ^ 2 / 2 := by
  linarith [sq_norm_decomp h]

omit [InnerProductSpace ℝ E] in
/-- **The (1−κ) price identity.** Bookkeeping form of the deficit: with
`κ = ‖p‖²/‖z‖²`, the e-power price of restricting bets to the polynomial
space is `(1−κ)` times the oracle quadratic growth `½‖z‖²`. -/
theorem eprice_identity {z : E} (p : E) (hz : z ≠ 0) :
    ‖z‖ ^ 2 / 2 - ‖p‖ ^ 2 / 2
      = (1 - ‖p‖ ^ 2 / ‖z‖ ^ 2) * (‖z‖ ^ 2 / 2) := by
  have hz2 : (‖z‖ : ℝ) ^ 2 ≠ 0 := pow_ne_zero 2 (norm_ne_zero_iff.mpr hz)
  field_simp

/-- **Existence of the optimal bet.** For a subspace admitting an orthogonal
projection (in particular any finite-dimensional bet space, the paper's case),
the decomposition point exists. -/
theorem exists_isProjOn (K : Submodule ℝ E) [K.HasOrthogonalProjection]
    (z : E) : ∃ p, IsProjOn z p K :=
  ⟨K.starProjection z, K.starProjection_apply_mem z,
    fun u hu => Submodule.starProjection_inner_eq_zero z u hu⟩

/-!
## The quantified small-bet approximation layer (math-spine §2)

`log(1+b)` deviates from its quadratic proxy `b − b²/2` by at most a cubic
term; this is the entire licence for replacing log-growth by the quadratic
GROW objective above.
-/

/-- **Second-order log Taylor bound.** For `|b| < 1`,
`|log(1+b) − (b − b²/2)| ≤ |b|³/(1−|b|)`. -/
theorem abs_log_one_add_sub_quad_le {b : ℝ} (hb : |b| < 1) :
    |Real.log (1 + b) - (b - b ^ 2 / 2)| ≤ |b| ^ 3 / (1 - |b|) := by
  have hb' : |(-b)| < 1 := by rwa [abs_neg]
  have h := Real.abs_log_sub_add_sum_range_le hb' 2
  have hs : (∑ i ∈ Finset.range 2, (-b) ^ (i + 1) / (i + 1))
      = -(b - b ^ 2 / 2) := by
    simp only [Finset.sum_range_succ]
    push_cast
    ring
  rw [hs] at h
  have h1 : (1 : ℝ) - -b = 1 + b := by ring
  rw [h1] at h
  have h2 : -(b - b ^ 2 / 2) + Real.log (1 + b)
      = Real.log (1 + b) - (b - b ^ 2 / 2) := by ring
  rw [h2] at h
  rwa [abs_neg] at h

/-- **Working form of the small-bet licence.** For `|b| ≤ ½` the quadratic
proxy is accurate to `2|b|³`; taking expectations bounds the growth error by
`2·E|b|³` — the third-moment price quoted in the math-spine. -/
theorem abs_log_one_add_sub_quad_le_of_le_half {b : ℝ} (hb : |b| ≤ 1 / 2) :
    |Real.log (1 + b) - (b - b ^ 2 / 2)| ≤ 2 * |b| ^ 3 := by
  have hb1 : |b| < 1 := lt_of_le_of_lt hb (by norm_num)
  have h := abs_log_one_add_sub_quad_le hb1
  have hpos : (0 : ℝ) < 1 - |b| := by linarith
  have hbound : |b| ^ 3 / (1 - |b|) ≤ 2 * |b| ^ 3 := by
    rw [div_le_iff₀ hpos]
    nlinarith [pow_nonneg (abs_nonneg b) 3]
  linarith

/-!
## Exact quadratic-envelope factor (main theorem's algebraic layer)

The expectation step is a conditional-moment calculation in the paper.  Lean
checks the pointwise fact that removes the no-bankruptcy obstruction: the
quadratic envelope is positive for every real score, without clipping.
-/

/-- The unnormalised quadratic envelope used by the exact e-factor. -/
noncomputable def quadEnvelope (h : ℝ) : ℝ := 1 + h + h ^ 2 / 2

/-- Completing the square for the quadratic envelope. -/
theorem quadEnvelope_eq (h : ℝ) :
    quadEnvelope h = ((h + 1) ^ 2 + 1) / 2 := by
  simp only [quadEnvelope]
  ring

/-- **No bankruptcy on the real line.** `1+h+h²/2` is strictly positive for
every real `h`; its minimum is `1/2` at `h=-1`. -/
theorem quadEnvelope_pos (h : ℝ) : 0 < quadEnvelope h := by
  rw [quadEnvelope_eq]
  nlinarith [sq_nonneg (h + 1)]

/-- The variance-compensated factor remains strictly positive whenever its
variance proxy is nonnegative. -/
theorem quadEnvelopeFactor_pos (h v : ℝ) (hv : 0 ≤ v) :
    0 < quadEnvelope h / (1 + v / 2) := by
  have hden : 0 < 1 + v / 2 := by linarith
  exact div_pos (quadEnvelope_pos h) hden

/-!
## Admissible quadratic family

The extension replaces `1+h+h²/2` by `1+h+c h²`.
These lemmas check the global positivity threshold and the elementary
second-order cancellation used in the local-equivalence argument.  They do
not assert global growth optimality.
-/

/-- `1+h+c h²` is nonnegative on the real line when `c ≥ 1/4`. -/
theorem quadraticFamily_nonneg (h c : ℝ) (hc : 1 / 4 ≤ c) :
    0 ≤ 1 + h + c * h ^ 2 := by
  have hs : 0 ≤ (h + 2) ^ 2 := sq_nonneg (h + 2)
  have hh : 0 ≤ h ^ 2 := sq_nonneg h
  nlinarith

/-- `1+h+c h²` is strictly positive on the real line when `c > 1/4`. -/
theorem quadraticFamily_pos (h c : ℝ) (hc : 1 / 4 < c) :
    0 < 1 + h + c * h ^ 2 := by
  have hs : 0 ≤ (h + 2) ^ 2 := sq_nonneg (h + 2)
  have hh : 0 ≤ h ^ 2 := sq_nonneg h
  by_cases hz : h = 0
  · subst hz
    norm_num
  · have hhpos : 0 < h ^ 2 := sq_pos_of_ne_zero hz
    nlinarith

/-- Matching `E(h²)` to `v` cancels the coefficient `c` at quadratic order:
`(c−1/2)v−cv=−v/2`. -/
theorem quadraticFamily_localPenalty (c v : ℝ) :
    (c - 1 / 2) * v - c * v = -v / 2 := by
  ring

end KuBettingCS
