#!/usr/bin/env python3
"""verify_stitch.py — Ku_Betting_CS P1: sanity-check W2 (stitched time-uniform CS).

НЕ pre-registered гейт (формальний BET-2 — окремо, за Spec). Перевіряє
math-spine §11 W2 на Gamma(2) (інваріантнісний аргумент законо-агностичний;
два закони вже покриті W1-скриптом verify_width.py):

  V  (validity):     stitched CS точно anytime-валідна — частка реплік, де
                     істина ВИКЛЮЧЕНА хоч раз до N, <= alpha (очік. << —
                     Markov+union слак).
  I  (інваріантність): криві відмови P_M(u, n_cp) двох методів у ВЛАСНИХ
                     u-одиницях (offset = u * s_M(n)) збігаються — пряма
                     перевірка методо-незалежності поля/накладки.
  R  (ratio):        50%-край u*_M і відношення ширин
                     (u*_two*s_two)/(u*_lin*s_lin) ~= sqrt(g2) на обох
                     чекпойнтах.

Конструкція (per side, per method): епохи k, n in [eta^k, eta^{k+1});
бюджет alpha_k = (alpha/2)/(zeta(s)(k+1)^s); ставка епохи k — T1
(кліп +-c, точне перецентрування), GROW-тюн на w_k = sqrt(2 sigma^2 g_M
L_k / eta^{k+1}), L_k = log(1/alpha_k); процес епохи живе з кроку 1;
reject кандидата у момент n, якщо БУДЬ-ЯКИЙ епох-процес перетнув 1/alpha_k
до n. Σ_k alpha_k = alpha/2 на сторону => точна Ville+union валідність.

Seed 2026. Залежності: numpy. Час ~1 хв.
"""

import numpy as np

trapezoid = getattr(np, "trapezoid", None) or np.trapz
rng = np.random.default_rng(2026)
FAILURES = []

ALPHA = 0.05
ETA = 2.0
S_EXP = 1.4
ZETA_S = 3.105527  # zeta(1.4), hardcoded
K_MIN, K_MAX = 5, 13          # епохи: n in [32, 16384); CS тривіальна до n=32
N = 10_000
R = 400
CLIP = 0.45
U_SET = np.array([0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4])
CHECKPOINTS = {1000: 9, 10_000: 13}   # n_cp -> k(n_cp) (eta=2)

# Gamma(2) centered: sigma^2=2, mu3=4, mu4=24
MU2, MU3, MU4 = 2.0, 4.0, 24.0
G2 = 1.0 - (MU3 / MU2 ** 1.5) ** 2 / (2.0 + (MU4 / MU2 ** 2 - 3.0))
F = np.array([[MU2, MU3], [MU3, MU4 - MU2 ** 2]])

# квадратура центрованої Gamma(2) для точного перецентрування
XG = np.linspace(-2.0 + 1e-9, 60.0, 400001)
PDF = (XG + 2.0) * np.exp(-(XG + 2.0))


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def alpha_k(k):
    return (ALPHA / 2.0) / (ZETA_S * (k + 1) ** S_EXP)


def L_k(k):
    return float(np.log(1.0 / alpha_k(k)))


def w_k(k, g):
    return float(np.sqrt(2.0 * MU2 * g * L_k(k) / ETA ** (k + 1)))


def lam_of(method, w):
    """GROW-тюн ставки на альтернативу offset w (w може бути <0)."""
    if method == "lin":
        return np.array([w / MU2, 0.0])
    return np.linalg.solve(F, np.array([w, w * w]))


def eta_recenter(lam):
    """E0[clip(b)] під нулем (закон shape-інваріантний по кандидату m)."""
    b = lam[0] * XG + lam[1] * (XG ** 2 - MU2)
    return float(trapezoid(PDF * np.clip(b, -CLIP, CLIP), XG))


def epoch_bet(Bc, lam, eta):
    """Кліпнута перецентрована ставка на масиві (x - m)."""
    b = lam[0] * Bc + lam[1] * (Bc * Bc - MU2)
    return np.clip(b, -CLIP, CLIP) - eta


X = rng.gamma(2.0, 1.0, (R, N)) - 2.0
print("=" * 74)
print(f"W2 sanity: stitched CS, Gamma(2)-centered, g2={G2:.3f}, "
      f"eta={ETA:.0f}, s={S_EXP}, epochs k={K_MIN}..{K_MAX}, "
      f"n={N}, R={R}, alpha={ALPHA}, кліп ±{CLIP}")
print("=" * 74)

# ------------------------------------------------------------- V: validity
print("\nV. Time-uniform validity (істина m=0, обидві сторони, всі епохи)")
for method, g in (("lin", 1.0), ("two", G2)):
    rej = np.zeros(R, dtype=bool)
    for k in range(K_MIN, K_MAX + 1):
        Lk = L_k(k)
        for sgn in (+1.0, -1.0):
            lam = lam_of(method, sgn * w_k(k, g))
            b = epoch_bet(X, lam, eta_recenter(lam))
            S = np.cumsum(np.log1p(b), axis=1)
            rej |= (S.max(axis=1) >= Lk)
    viol = float(rej.mean())
    check(f"validity {method}: P(істина виключена хоч раз до N) <= alpha",
          viol <= ALPHA, f"виміряно {viol:.4f} (бюджет {ALPHA})")

# --------------------------------------------- I + R: інваріантність, край
print("\nI/R. Криві відмови у власних u-одиницях + 50%-край")
ratios = {}
for n_cp, k_cp in CHECKPOINTS.items():
    print(f"\n  Чекпойнт n={n_cp} (епоха k={k_cp}, L_k={L_k(k_cp):.3f}, "
          f"B(n) теор. = {(np.sqrt(ETA**(k_cp+1)/n_cp)+np.sqrt(n_cp/ETA**(k_cp+1)))/2:.4f})")
    curves = {}
    for method, g in (("lin", 1.0), ("two", G2)):
        s_scale = np.sqrt(2.0 * MU2 * g * L_k(k_cp) / n_cp)
        P = []
        for u in U_SET:
            m = -u * s_scale
            B = X - m
            rej = np.zeros(R, dtype=bool)
            for k in range(K_MIN, k_cp + 1):
                lam = lam_of(method, +w_k(k, g))   # up-ставка (до істини)
                b = epoch_bet(B, lam, eta_recenter(lam))
                S = np.cumsum(np.log1p(b[:, :n_cp]), axis=1)
                rej |= (S.max(axis=1) >= L_k(k))
            P.append(float(rej.mean()))
        curves[method] = (np.array(P), s_scale)
        row = " ".join(f"{p:.3f}" for p in P)
        us = " ".join(f"{float(u):.1f}" for u in U_SET)
        print(f"    {method}: s_M={s_scale:.4f};  P(u)= [{row}]  (u = {us})")

    diff = float(np.max(np.abs(curves["lin"][0] - curves["two"][0])))
    check(f"n={n_cp}: інваріантність — max_u |P_lin(u)-P_two(u)| <= 0.12",
          diff <= 0.12, f"виміряно {diff:.3f}")

    # u*-смуги: асимптотична (широка на малому n — преасимптотика під
    # сильною асиметрією + union-підрізання краю ранніми епохами; вузька
    # на великому). Ratio: на малому n вимагаємо лише суттєвий виграш у
    # правильний бік; точний sqrt(g2) — асимптотичне твердження.
    lo, hi = (0.78, 1.20) if n_cp < 5000 else (0.85, 1.15)
    ustars = {}
    for method in ("lin", "two"):
        P, s_scale = curves[method]
        idx = np.flatnonzero(P >= 0.5)
        if idx.size == 0 or idx[0] == 0:
            ustars[method] = np.nan
        else:
            i = idx[0]
            u0, u1, p0, p1 = U_SET[i - 1], U_SET[i], P[i - 1], P[i]
            ustars[method] = u0 + (0.5 - p0) * (u1 - u0) / (p1 - p0)
        check(f"n={n_cp}: u*_{method} у [{lo}, {hi}]",
              lo <= ustars[method] <= hi, f"u*={ustars[method]:.3f}")
    ratio = (ustars["two"] * curves["two"][1]) / (ustars["lin"] * curves["lin"][1])
    ratios[n_cp] = ratio
    if n_cp < 5000:
        check(f"n={n_cp}: преасимптотичний ratio < 0.90 (виграш реалізовано частково)",
              ratio < 0.90,
              f"ratio={ratio:.4f} (−{100*(1-ratio):.0f}% ширини; sqrt(g2)={np.sqrt(G2):.4f} — асимптота)")
    else:
        check(f"n={n_cp}: |width-ratio - sqrt(g2)| <= 0.06",
              abs(ratio - np.sqrt(G2)) <= 0.06,
              f"ratio={ratio:.4f} vs sqrt(g2)={np.sqrt(G2):.4f}")

n_small, n_big = min(ratios), max(ratios)
check("збіжність: |ratio(n_big) - sqrt(g2)| < |ratio(n_small) - sqrt(g2)|",
      abs(ratios[n_big] - np.sqrt(G2)) < abs(ratios[n_small] - np.sqrt(G2)),
      f"{abs(ratios[n_big]-np.sqrt(G2)):.4f} < {abs(ratios[n_small]-np.sqrt(G2)):.4f}")

print("=" * 74)
if FAILURES:
    print(f"ПІДСУМОК: {len(FAILURES)} FAIL: {FAILURES}")
    raise SystemExit(1)
print(f"ПІДСУМОК: ALL PASS — stitched CS: точна anytime-валідність, "
      f"інваріантні криві, ratio {ratios[n_small]:.3f}->{ratios[n_big]:.3f} "
      f"vs sqrt(g2)={np.sqrt(G2):.3f} (формальний BET-2 — окремо, за Spec)")
