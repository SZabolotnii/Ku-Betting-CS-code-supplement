#!/usr/bin/env python3
"""verify_p1_close.py — Ku_Betting_CS P1: закриття двох останніх позицій.

НЕ pre-registered гейт. Перевіряє math-spine §11:

  A (Теорема CS, clipped-subspace): у кліпнутій сім'ї
     C = {clip(l^T phi, -c, c) - eta(l)} (точно fair під P0):
     A1: числовий sup росту по C >= ріст кліпнутого l* (тривіально, санітарно);
     A2: кліпнутий l* майже-оптимальний у C; rel gap = O(d) -> перевіряємо
         малість при d=0.05σ І спадання зі зменшенням d;
     A3: тотожність у χ²-ВАЛЮТІ: 2 sup_C / chi2 ~= kappa (обидві сторони
         пінчингу — кліп і не з'їдає понад 1-kappa, і не обходить
         підпростірну ціну нелінійністю); асимптотичне твердження ->
         малість при d=0.05σ + збіжність;
     A4: QMD-межа (чесність KL-форми): Gamma(2)-location має нескінченну
         Fisher-інформацію (E[T^-2]=inf) => 2KL/chi2 далеко від 1 навіть
         локально — документуємо; тому Теорема 1-kappa формулюється в
         χ²-валюті, KL-форма — лише під припущенням (A)/QMD.

  B (Проп. W3, plug-in дрейф): для R2-ставки з оракульним l2 і предиктивним
     mu2-hat дрейф D_n = l2 * Sum_i (mu2 - mu2_hat_i) має
     B1: sd(D_n) ~= |l2| sqrt(2 nu n), nu = Var((x-m)^2)  (формула масштабу);
     B2: E[D_n] ~ 0 (леак — флуктуація, не зсув; running mean незміщений).

Закон: Gamma(2) centered (sigma^2=2, mu3=4, mu4=24, nu=20). Seed 2026.
"""

import numpy as np

trapezoid = getattr(np, "trapezoid", None) or np.trapz
rng = np.random.default_rng(2026)
FAILURES = []

MU2, MU3, MU4 = 2.0, 4.0, 24.0
NU = MU4 - MU2 ** 2                      # Var((x-m)^2) = 20
F = np.array([[MU2, MU3], [MU3, MU4 - MU2 ** 2]])
CLIP = 0.45

XG = np.linspace(-2.0 + 1e-9, 80.0, 300001)
P0 = (XG + 2.0) * np.exp(-(XG + 2.0))
PHI1, PHI2 = XG, XG ** 2 - MU2


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def growth_of(lam, qpdf):
    """Точний ріст E_Q log(1+b_c) кліпнутої перецентрованої ставки."""
    b = np.clip(lam[0] * PHI1 + lam[1] * PHI2, -CLIP, CLIP)
    eta = float(trapezoid(P0 * b, XG))
    return float(trapezoid(qpdf * np.log1p(b - eta), XG))


def sup_over_family(lam_star, qpdf):
    """Числовий sup росту по C: груба сітка навколо/поза l* + 2 рефайни."""
    best_l, best_g = np.array(lam_star), growth_of(lam_star, qpdf)
    # груба сітка: l1 у [0, 2.5 l1*], l2 у [4 l2*, -2 l2*] (перетинає 0)
    g1 = np.linspace(0.0, 2.5 * lam_star[0], 31)
    lo2, hi2 = sorted((4.0 * lam_star[1], -2.0 * lam_star[1]))
    g2 = np.linspace(lo2, hi2, 31)
    for _ in range(3):
        for a in g1:
            for b2 in g2:
                g = growth_of((a, b2), qpdf)
                if g > best_g:
                    best_g, best_l = g, np.array((a, b2))
        d1 = (g1[1] - g1[0]) * 1.5
        d2 = (g2[1] - g2[0]) * 1.5
        g1 = np.linspace(best_l[0] - d1, best_l[0] + d1, 21)
        g2 = np.linspace(best_l[1] - d2, best_l[1] + d2, 21)
    return best_g, best_l


print("=" * 74)
print("A. Теорема CS (clipped-subspace): Gamma(2), кліп ±0.45, m=0")
print("=" * 74)
gaps, devs = {}, {}
for dfrac in (0.05, 0.10):
    d = dfrac * np.sqrt(MU2)
    qpdf = np.interp(XG - d, XG, P0, left=0.0, right=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_int = np.where(qpdf > 0, qpdf * np.log(np.where(qpdf > 0, qpdf, 1.0) / P0), 0.0)
    kl = float(trapezoid(kl_int, XG))
    chi2 = float(trapezoid((qpdf - P0) ** 2 / P0, XG))
    beta = np.array([d, d * d])
    kappa = float(beta @ np.linalg.solve(F, beta) / chi2)
    lam_star = np.linalg.solve(F, beta)
    g_star = growth_of(lam_star, qpdf)
    g_sup, l_sup = sup_over_family(lam_star, qpdf)
    gaps[dfrac] = rel_gap = (g_sup - g_star) / g_sup
    devs[dfrac] = dev = abs(2.0 * g_sup / chi2 - kappa)
    print(f"\n  d={dfrac}σ: kappa={kappa:.4f}, chi2={chi2:.4e}, KL={kl:.4e}, "
          f"2KL/chi2={2*kl/chi2:.3f}; ріст(clip l*)={g_star:.4e}, "
          f"sup_C={g_sup:.4e}, 2 sup_C/chi2={2*g_sup/chi2:.4f} "
          f"@ l=({l_sup[0]:.4f},{l_sup[1]:.4f}) [l*=({lam_star[0]:.4f},{lam_star[1]:.4f})]")
    check(f"d={dfrac}σ A1: sup_C >= ріст(clip l*)", g_sup >= g_star - 1e-12)
    if dfrac == 0.05:
        check("d=0.05σ A2: clip-l* 99%-оптимальний у C",
              rel_gap <= 0.01, f"rel gap = {rel_gap:.2e}")
        check("d=0.05σ A3: |2 sup_C/chi2 - kappa| <= 0.03 (χ²-валюта)",
              dev <= 0.03, f"dev = {dev:.4f}")
    else:
        check("d=0.1σ A2: rel gap <= 0.04 (O(d)-твердження)",
              rel_gap <= 0.04, f"rel gap = {rel_gap:.2e}")
        check("d=0.1σ A3: dev <= 0.10 (репорт асимптотики)",
              dev <= 0.10, f"dev = {dev:.4f}")
    check(f"d={dfrac}σ A4: 2KL/chi2 у [1.2, 1.8] (QMD-межа Gamma(2) видима)",
          1.2 <= 2 * kl / chi2 <= 1.8, f"{2*kl/chi2:.3f} (для QMD-закону було б ~1)")

check("A2-збіжність: gap(0.05σ) < 0.6*gap(0.1σ)",
      gaps[0.05] < 0.6 * gaps[0.10], f"{gaps[0.05]:.4f} < 0.6*{gaps[0.10]:.4f}")
check("A3-збіжність: dev(0.05σ) < 0.6*dev(0.1σ)",
      devs[0.05] < 0.6 * devs[0.10], f"{devs[0.05]:.4f} < 0.6*{devs[0.10]:.4f}")

print()
print("=" * 74)
print("B. Проп. W3 (plug-in дрейф): оракульний l2, предиктивний mu2-hat")
print("=" * 74)
N, R, N0 = 3000, 2000, 100
L = float(np.log(40.0))                       # log(2/alpha), alpha=0.05
w = float(np.sqrt(2.0 * MU2 * 0.6 * L / N))   # two-метод, g2=0.6
lam2 = float(np.linalg.solve(F, np.array([w, w * w]))[1])
X = rng.gamma(2.0, 1.0, (R, N)) - 2.0
S2 = np.cumsum(X ** 2, axis=1)
idx = np.arange(1, N + 1, dtype=float)
mu2_hat = np.empty_like(X)
mu2_hat[:, 0] = MU2                            # до 1-го кроку — прайор
mu2_hat[:, 1:] = S2[:, :-1] / idx[:-1]         # предиктивний running mean
D = lam2 * np.sum(MU2 - mu2_hat[:, N0:], axis=1)
sd_pred = abs(lam2) * np.sqrt(2.0 * NU * N)
sd_emp = float(np.std(D, ddof=1))
mean_emp = float(np.mean(D))
print(f"  n={N}, R={R}, warmup={N0}: lam2={lam2:.5f}, "
      f"sd_pred=|l2|sqrt(2 nu n)={sd_pred:.3f}")
print(f"  D_n: mean={mean_emp:.3f}, sd={sd_emp:.3f}; поріг L={L:.3f} "
      f"(дрейф O(sqrt(L)) проти сигналу O(L))")
check("B1: sd(D_n)/sd_pred у [0.75, 1.35]",
      0.75 <= sd_emp / sd_pred <= 1.35, f"ratio={sd_emp/sd_pred:.3f}")
check("B2: |mean(D_n)|/sd(D_n) <= 0.15 (леак — флуктуація, не зсув)",
      abs(mean_emp) / sd_emp <= 0.15, f"{abs(mean_emp)/sd_emp:.3f}")

print("=" * 74)
if FAILURES:
    print(f"ПІДСУМОК: {len(FAILURES)} FAIL: {FAILURES}")
    raise SystemExit(1)
print("ПІДСУМОК: ALL PASS (Теорема CS + Проп. W3 — числово підтверджені)")
