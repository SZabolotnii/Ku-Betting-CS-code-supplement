#!/usr/bin/env python3
"""verify_width.py — Ku_Betting_CS P1: числовий sanity-check width-теореми (W1/W3).

НЕ pre-registered гейт (формальний гейт BET-2 — окремо, за Spec, з порогами
в заголовку ДО прогону). Це перевірка тверджень math-spine §11:

  W1 (інваріантність): дворядна ставка (лінійна + контрольна змінна
     (x-m)^2 - s^2) дає e-CI, чия ПІВШИРИНА відноситься до першопорядкової
     як sqrt(g2), бо log-wealth поля обох методів у власній локальній
     параметризації m = mu + u*w_n^(M) збігаються до ОДНОГО граничного
     гауссового поля N(L(2u-1), 2L). Перевіряємо відношення ширин при
     фіксованому alpha=0.05 (інваріантність робить його вірним і тут,
     не лише при alpha->0).
  CL (кліп-вартість): T1-кліп + точне P0-перецентрування зберігає
     валідність точно; втрата росту третьомоментна -> ширина не рухається.
  W3 (plug-in / R2): предиктивний плагін moment'ів у lambda не змінює
     відношення ширин; валідність — асимптотична (floor-кліп без
     перецентрування дає малий контрольований леак, репортуємо).

Конструкція e-CI (fixed-horizon, по два односторонні e-тести):
  для кожного кандидата m: ставка "вгору" тюнована на альтернативу
  mu = m + w^(M), ставка "вниз" — дзеркально; reject сторони при
  W_n >= 2/alpha (Ville/Markov на рівні alpha/2); CI = hull прийнятих m.

Seed: 2026. Залежності: numpy. Час прогону ~1-2 хв.
"""

import numpy as np

trapezoid = getattr(np, "trapezoid", None) or np.trapz
rng = np.random.default_rng(2026)
FAILURES = []

ALPHA = 0.05
THRESH = 2.0 / ALPHA          # поріг на сторону (alpha/2)
L = np.log(THRESH)            # log(40) ~ 3.689
N = 3000
R = 300
CLIP = 0.45
WARMUP = 100                  # plug-in: перші кроки без ставки (предиктивно)


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------- закони
class Law:
    """Центрований закон з відомими моментами, семплером і щільністю."""

    def __init__(self, name, mu2, mu3, mu4, sampler, grid, pdf):
        self.name, self.mu2, self.mu3, self.mu4 = name, mu2, mu3, mu4
        self.sampler, self.grid, self.pdf = sampler, grid, pdf
        g3 = mu3 / mu2 ** 1.5
        g4 = mu4 / mu2 ** 2 - 3.0
        self.g2 = 1.0 - g3 ** 2 / (2.0 + g4)
        self.F = np.array([[mu2, mu3], [mu3, mu4 - mu2 ** 2]])


def make_gamma(k=2):
    grid = np.linspace(-k, 120.0, 400001)
    from math import gamma as G
    pdf = np.where(grid + k > 0,
                   (grid + k) ** (k - 1) * np.exp(-(grid + k)) / G(k), 0.0)
    return Law(f"Gamma(k={k})-centered", k, 2 * k, 3 * k * k + 6 * k,
               lambda n: rng.gamma(k, 1.0, n) - k, grid, pdf)


def make_mix():
    W_, A_, S_ = np.array([0.8, 0.2]), np.array([-0.6, 2.4]), np.array([0.8, 1.2])
    raw = np.zeros(5)
    zm = [1.0, 0.0, 1.0, 0.0, 3.0]
    from math import comb
    for wi, ai, si in zip(W_, A_, S_):
        raw += wi * np.array([sum(comb(r, j) * ai ** (r - j) * si ** j * zm[j]
                                  for j in range(r + 1)) for r in range(5)])
    mu2, mu3, mu4 = raw[2], raw[3], raw[4]
    grid = np.linspace(-12.0, 18.0, 400001)
    pdf = np.zeros_like(grid)
    for wi, ai, si in zip(W_, A_, S_):
        pdf += wi * np.exp(-0.5 * ((grid - ai) / si) ** 2) / (si * np.sqrt(2 * np.pi))

    def sampler(n):
        comp = rng.choice(2, size=n, p=W_)
        return A_[comp] + S_[comp] * rng.standard_normal(n)

    return Law("GaussMix(.8,.2)", mu2, mu3, mu4, sampler, grid, pdf)


# ------------------------------------------------ ставки і e-CI (oracle)
def lam_two(law, w):
    """GROW-тюн дворядної ставки на альтернативу offset w (може бути <0)."""
    return np.linalg.solve(law.F, np.array([w, w * w]))


def eta_clip(law, lam1, lam2):
    """E0[clip(b)] квадратурою — точне перецентрування T1."""
    b = lam1 * law.grid + lam2 * (law.grid ** 2 - law.mu2)
    return float(trapezoid(law.pdf * np.clip(b, -CLIP, CLIP), law.grid))


def run_oracle(law, X, method):
    """Fixed-horizon e-CI для методу 'lin'|'two'. Повертає (halfwidths, cover)."""
    w1 = np.sqrt(2.0 * law.mu2 * L / N)
    w = w1 * (np.sqrt(law.g2) if method == "two" else 1.0)
    grid_m = np.linspace(-3.5 * w1, 3.5 * w1, 101)
    if method == "two":
        lu = lam_two(law, +w)
        ld = lam_two(law, -w)
    else:
        lu = np.array([+w / law.mu2, 0.0])
        ld = np.array([-w / law.mu2, 0.0])
    eu = eta_clip(law, lu[0], lu[1])
    ed = eta_clip(law, ld[0], ld[1])

    acc = np.zeros((R, grid_m.size), dtype=bool)
    for j, m in enumerate(grid_m):
        B = X - m
        for lam, eta, side in ((lu, eu, 0), (ld, ed, 1)):
            b = np.clip(lam[0] * B + lam[1] * (B * B - law.mu2),
                        -CLIP, CLIP) - eta
            S = np.log1p(b).sum(axis=1)
            if side == 0:
                rej_up = S >= L
            else:
                rej_dn = S >= L
        acc[:, j] = ~(rej_up | rej_dn)
    return _hull_stats(acc, grid_m)


def _hull_stats(acc, grid_m):
    halfw, cover, empty = [], 0, 0
    for r in range(acc.shape[0]):
        idx = np.flatnonzero(acc[r])
        if idx.size == 0:
            empty += 1
            continue
        lo, hi = grid_m[idx[0]], grid_m[idx[-1]]
        halfw.append(0.5 * (hi - lo))
        if lo <= 0.0 <= hi:
            cover += 1
    n_ok = acc.shape[0] - empty
    return (np.mean(halfw) if halfw else np.nan,
            cover / max(n_ok, 1), empty)


# ------------------------------------------------ plug-in (R2, Gamma)
def run_plugin(law, X, method):
    """Предиктивний плагін моментів у lambda; floor-кліп -0.9 без
    перецентрування (леак репортується окремо)."""
    w1 = np.sqrt(2.0 * law.mu2 * L / N)
    w = w1 * (np.sqrt(law.g2) if method == "two" else 1.0)
    grid_m = np.linspace(-3.5 * w1, 3.5 * w1, 101)

    # кумулятивні суми степенів (предиктивні: зсув на 1)
    P1 = np.cumsum(X, axis=1)
    P2 = np.cumsum(X ** 2, axis=1)
    P3 = np.cumsum(X ** 3, axis=1)
    P4 = np.cumsum(X ** 4, axis=1)
    idx = np.arange(1, N + 1, dtype=float)

    acc = np.zeros((R, grid_m.size), dtype=bool)
    for j, m in enumerate(grid_m):
        # предиктивні центральні моменти про m (до i-1 включно)
        c1 = np.roll(P1, 1, axis=1); c1[:, 0] = 0.0
        c2 = np.roll(P2, 1, axis=1); c2[:, 0] = 0.0
        c3 = np.roll(P3, 1, axis=1); c3[:, 0] = 0.0
        c4 = np.roll(P4, 1, axis=1); c4[:, 0] = 0.0
        cnt = np.maximum(np.roll(idx, 1), 1.0); cnt[0] = 1.0
        m1 = (c1 - cnt * 0.0) / cnt - m  # середнє (X_j - m)
        # сирі моменти (X_j - m)^r через біном
        r1 = c1 / cnt - m
        r2 = c2 / cnt - 2 * m * c1 / cnt + m * m
        r3 = (c3 / cnt - 3 * m * c2 / cnt + 3 * m * m * c1 / cnt - m ** 3)
        r4 = (c4 / cnt - 4 * m * c3 / cnt + 6 * m * m * c2 / cnt
              - 4 * m ** 3 * c1 / cnt + m ** 4)
        mu2h = np.maximum(r2, 1e-3)
        F11, F12 = mu2h, r3
        F22 = np.maximum(r4 - mu2h ** 2, 1e-3)
        B = X - m
        for sgn in (+1.0, -1.0):
            ww = sgn * w
            if method == "two":
                det = F11 * F22 - F12 ** 2
                ok = det > 1e-6
                l1 = np.where(ok, (F22 * ww - F12 * ww * ww) / det, ww / mu2h)
                l2 = np.where(ok, (-F12 * ww + F11 * ww * ww) / det, 0.0)
            else:
                l1, l2 = ww / mu2h, np.zeros_like(mu2h)
            b = l1 * B + l2 * (B * B - mu2h)
            b[:, :WARMUP] = 0.0
            b = np.maximum(b, -0.9)          # floor-кліп (леак -> звіт)
            S = np.log1p(b).sum(axis=1)
            if sgn > 0:
                rej_up = S >= L
            else:
                rej_dn = S >= L
        acc[:, j] = ~(rej_up | rej_dn)
    return _hull_stats(acc, grid_m)


# ---------------------------------------------------------------- прогін
print("=" * 74)
print(f"W1/W3 sanity: fixed-horizon e-CI, n={N}, R={R}, alpha={ALPHA} "
      f"(поріг {THRESH:.0f}/сторону), кліп ±{CLIP} + точне перецентрування")
print("=" * 74)

for law in (make_gamma(2), make_mix()):
    X = law.sampler(R * N).reshape(R, N)
    w1_theory = np.sqrt(2.0 * law.mu2 * L / N)
    hw_lin, cov_lin, e1 = run_oracle(law, X, "lin")
    hw_two, cov_two, e2 = run_oracle(law, X, "two")
    ratio = hw_two / hw_lin
    print(f"\n  {law.name}: g2={law.g2:.4f}, sqrt(g2)={np.sqrt(law.g2):.4f}, "
          f"w1_theory={w1_theory:.4f}")
    print(f"    lin: halfwidth={hw_lin:.4f} ({hw_lin/w1_theory:.2f}x w1), "
          f"coverage={cov_lin:.3f}, empty={e1}")
    print(f"    two: halfwidth={hw_two:.4f}, coverage={cov_two:.3f}, empty={e2}")
    print(f"    RATIO two/lin = {ratio:.4f}  (прогноз sqrt(g2) = {np.sqrt(law.g2):.4f})")
    check(f"{law.name}: coverage lin >= 0.93", cov_lin >= 0.93, f"{cov_lin:.3f}")
    check(f"{law.name}: coverage two >= 0.93", cov_two >= 0.93, f"{cov_two:.3f}")
    check(
        f"{law.name}: |ratio - sqrt(g2)| <= 0.06",
        abs(ratio - np.sqrt(law.g2)) <= 0.06,
        f"ratio={ratio:.4f} vs {np.sqrt(law.g2):.4f}",
    )
    check(f"{law.name}: empty-CI реплік < 2%", (e1 + e2) < 0.02 * 2 * R,
          f"{e1}+{e2}")

# plug-in (R2) — Gamma
law = make_gamma(2)
X = law.sampler(R * N).reshape(R, N)
hw_lin, cov_lin, e1 = run_plugin(law, X, "lin")
hw_two, cov_two, e2 = run_plugin(law, X, "two")
ratio = hw_two / hw_lin
print(f"\n  PLUG-IN (R2) {law.name}: предиктивні moment'и, warmup={WARMUP}, "
      f"floor-кліп -0.9 без перецентрування")
print(f"    lin: halfwidth={hw_lin:.4f}, coverage={cov_lin:.3f}, empty={e1}")
print(f"    two: halfwidth={hw_two:.4f}, coverage={cov_two:.3f}, empty={e2}")
print(f"    RATIO two/lin = {ratio:.4f}  (прогноз {np.sqrt(law.g2):.4f})")
check("plug-in: coverage lin >= 0.91", cov_lin >= 0.91, f"{cov_lin:.3f}")
check("plug-in: coverage two >= 0.91", cov_two >= 0.91, f"{cov_two:.3f}")
check(
    "plug-in: |ratio - sqrt(g2)| <= 0.08",
    abs(ratio - np.sqrt(law.g2)) <= 0.08,
    f"ratio={ratio:.4f}",
)

print("=" * 74)
if FAILURES:
    print(f"ПІДСУМОК: {len(FAILURES)} FAIL: {FAILURES}")
    raise SystemExit(1)
print("ПІДСУМОК: ALL PASS (sanity W1/W3; формальний гейт BET-2 — окремо, за Spec)")
