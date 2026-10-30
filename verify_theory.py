#!/usr/bin/env python3
"""verify_theory.py — Ku_Betting_CS: числовий sanity-check теорії (P0).

НЕ pre-registered гейт (ті — BET-1..5 зі Spec, пороги фіксуються перед
повним прогоном). Це перевірка ТОТОЖНОСТЕЙ math-spine.md:

  A. Проп. 5.1: J2 = J1/g2 точно, g2 = 1 - g3^2/(2+g4)   (аналітичні моменти)
  B. Теор. 3.1 + Лема 2.1: емпіричний log-ріст оптимальної квадратичної
     ставки узгоджений із GROW-проксі в межах кубічної похибки (MC)
  C. Теор. 4.2: дефіцит проти Kelly-оракула -> (1-kappa) при delta -> 0
     (гауссова суміш: повний носій, щільність аналітична, KL/chi2 квадратурою)

Seed: 2026. Залежності: numpy.
"""

import math

import numpy as np

trapezoid = getattr(np, "trapezoid", None) or np.trapz
rng = np.random.default_rng(2026)
FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------- Part A
print("=" * 72)
print("A. Тотожність J2 = J1/g2 (Проп. 5.1) на аналітичних центральних моментах")
print("=" * 72)


def j_ratio_from_moments(mu2, mu3, mu4):
    """[F^-1]_11 * mu2  ==  J2/J1 у головному порядку (див. math-spine §5)."""
    F = np.array([[mu2, mu3], [mu3, mu4 - mu2 ** 2]])
    Finv11 = np.linalg.inv(F)[0, 0]
    return Finv11 * mu2, F


def g2_from_moments(mu2, mu3, mu4):
    g3 = mu3 / mu2 ** 1.5
    g4 = mu4 / mu2 ** 2 - 3.0
    return 1.0 - g3 ** 2 / (2.0 + g4)


def gaussian_raw_moments(a, s, r_max=4):
    """E[(a + s Z)^r], Z~N(0,1), r=0..r_max."""
    z_mom = [1.0, 0.0, 1.0, 0.0, 3.0]  # E Z^r
    out = []
    for r in range(r_max + 1):
        m = sum(
            math.comb(r, j) * a ** (r - j) * s ** j * z_mom[j]
            for j in range(r + 1)
        )
        out.append(m)
    return out


def mixture_central_moments(w, a, s):
    """Центральні моменти 2..4 суміші sum w_i N(a_i, s_i^2) із нульовим середнім."""
    raw = np.zeros(5)
    for wi, ai, si in zip(w, a, s):
        raw += wi * np.array(gaussian_raw_moments(ai, si))
    m1 = raw[1]
    assert abs(m1) < 1e-12, "суміш має бути центрованою"
    mu2 = raw[2]
    mu3 = raw[3]
    mu4 = raw[4]
    return mu2, mu3, mu4


laws = {}
for k in (1, 2, 3, 5):
    laws[f"Gamma(k={k})"] = (k, 2 * k, 3 * k ** 2 + 6 * k)
sig = 0.4
m = [np.exp(r * r * sig * sig / 2.0) for r in range(5)]  # raw lognormal, mu=0
mean = m[1]
mu2 = m[2] - mean ** 2
mu3 = m[3] - 3 * mean * m[2] + 2 * mean ** 3
mu4 = m[4] - 4 * mean * m[3] + 6 * mean ** 2 * m[2] - 3 * mean ** 4
laws[f"Lognormal(s={sig})"] = (mu2, mu3, mu4)
MIX_W, MIX_A, MIX_S = (0.8, 0.2), (-0.6, 2.4), (0.8, 1.2)
laws["GaussMix(.8,.2)"] = mixture_central_moments(MIX_W, MIX_A, MIX_S)

for name, (u2, u3, u4) in laws.items():
    ratio, _ = j_ratio_from_moments(u2, u3, u4)
    g2 = g2_from_moments(u2, u3, u4)
    err = abs(ratio - 1.0 / g2)
    check(
        f"{name}: J2/J1 == 1/g2",
        err < 1e-10,
        f"J2/J1={ratio:.6f}, 1/g2={1/g2:.6f}, ширина CS x sqrt(g2)={np.sqrt(g2):.4f}",
    )

# ---------------------------------------------------------------- Part B/C
print("=" * 72)
print("B/C. Growth sandwich (Лема 2.1) і дефіцит -> (1-kappa) (Теор. 4.2)")
print("     P0 = скошена гауссова суміш (g3~1.1); Q_d = P0 зсунута на d.")
print("     Ріст/KL/chi2 — квадратурою (детерм.); MC — лише крос-чек і O1-демо")
print("=" * 72)

u2, u3, u4 = laws["GaussMix(.8,.2)"]
sd = np.sqrt(u2)
W, A, S = np.array(MIX_W), np.array(MIX_A), np.array(MIX_S)


def p0_density(x):
    d = np.zeros_like(x, dtype=float)
    for wi, ai, si in zip(W, A, S):
        d += wi * np.exp(-0.5 * ((x - ai) / si) ** 2) / (si * np.sqrt(2 * np.pi))
    return d


def sample_p0(n):
    comp = rng.choice(len(W), size=n, p=W)
    return A[comp] + S[comp] * rng.standard_normal(n)


def bet_terms(d):
    """(lam, grow_pred, kappa, kl, b(xg)) для зсуву d; math-spine §3, §5."""
    beta = np.array([d, d * d])
    F = np.array([[u2, u3], [u3, u4 - u2 ** 2]])
    EQ_x2 = u2 + d * d
    EQ_x3 = u3 + 3 * d * u2 + d ** 3
    EQ_x4 = u4 + 4 * d * u3 + 6 * d * d * u2 + d ** 4
    M = np.array(
        [
            [EQ_x2, EQ_x3 - u2 * d],
            [EQ_x3 - u2 * d, EQ_x4 - 2 * u2 * EQ_x2 + u2 ** 2],
        ]
    )
    lam = np.linalg.solve(M, beta)  # точний максимайзер M_Q-форми (Зауваж. 3.2)
    grow_pred = beta @ lam - 0.5 * lam @ M @ lam
    qg = p0_density(xg - d)
    chi2 = trapezoid((qg - p0g) ** 2 / p0g, xg)
    kl = trapezoid(qg * np.log(qg / p0g), xg)
    J = beta @ np.linalg.solve(F, beta)
    return lam, float(grow_pred), float(J / chi2), float(kl), qg


def bet_min_on_R(lam):
    """min по всій прямій квадратичної ставки b = lam1*x + lam2*(x^2-u2)."""
    if lam[1] <= 0:
        return -np.inf
    xstar = -lam[0] / (2.0 * lam[1])
    return lam[0] * xstar + lam[1] * (xstar ** 2 - u2)


# квадратурна сітка під носій суміші (комп. 2 на 2.4 +- 1.2)
xg = np.linspace(-12.0, 18.0, 600001)
p0g = p0_density(xg)

# --- O1 наживо (math-spine §6): для g3 > 0 оптимальний lam2 < 0 (парабола
# вниз) => СИРА квадратична ставка нелегальна на R при будь-якому d.
# Тому далі скрізь маршрут T1: кліп ставки в [-CLIP, CLIP] + ТОЧНЕ
# перецентрування під відомим P0 (простий нуль) — валідність точна.
CLIP = 0.45

lam_demo, *_ = bet_terms(0.05 * sd)
check(
    "O1-демо: lam2* < 0 при g3 > 0 => min_R b = -inf уже при d=0.05sd",
    lam_demo[1] < 0 and bet_min_on_R(lam_demo) == -np.inf,
    f"lam* = ({lam_demo[0]:.4f}, {lam_demo[1]:.4f}) — тому T1/T2 у §6 "
    f"хребта є вхідною умовою, не крайовим випадком",
)


def t1_bet(lam, qg):
    """T1: кліп + точне P0-перецентрування. Повертає (b_c на сітці, eta,
    q-масу кліпу, проксі росту й точний ріст/куб. конверт для b_c)."""
    braw = lam[0] * xg + lam[1] * (xg ** 2 - u2)
    bc = np.clip(braw, -CLIP, CLIP)
    eta = float(trapezoid(p0g * bc, xg))  # E0[clip] != 0
    bc = bc - eta                          # точно fair під P0
    clip_mass = float(trapezoid(qg * (np.abs(braw) > CLIP), xg))
    eq_b = float(trapezoid(qg * bc, xg))
    eq_b2 = float(trapezoid(qg * bc ** 2, xg))
    proxy = eq_b - 0.5 * eq_b2
    growth = float(trapezoid(qg * np.log1p(bc), xg))
    env = float(trapezoid(qg * np.abs(bc) ** 3 / (1.0 - np.abs(bc)), xg))
    return bc, eta, clip_mass, proxy, growth, env


print(
    f"  {'delta/sd':>9} {'kappa':>7} {'ріст(b_c)':>11} {'проксі(b_c)':>11} "
    f"{'|розрив|':>9} {'конверт':>9} {'Q(кліп)':>8} {'дефіцит/KL':>11} {'1-kappa':>8}"
)

rows = []
for dfrac in (0.02, 0.05, 0.10):
    d = dfrac * sd
    lam, grow_pred, kappa, kl, qg = bet_terms(d)
    bc, eta, clip_mass, proxy, growth, env = t1_bet(lam, qg)
    bmin = float(np.min(bc))
    gap = abs(growth - proxy)
    deficit_ratio = (kl - growth) / kl
    rows.append((dfrac, kappa, deficit_ratio))
    print(
        f"  {dfrac:>9.2f} {kappa:>7.4f} {growth:>11.4e} {proxy:>11.4e} "
        f"{gap:>9.2e} {env:>9.2e} {clip_mass:>8.1e} {deficit_ratio:>11.4f} "
        f"{1 - kappa:>8.4f}"
    )
    check(
        f"d={dfrac}sd: T1-ставка легальна і в small-bet зоні",
        (bmin > -1.0) and (float(np.max(np.abs(bc))) < 0.5),
        f"b_c у [{bmin:.3f}, {float(np.max(bc)):.3f}], eta={eta:.1e}",
    )
    check(
        f"d={dfrac}sd: growth sandwich (Лема 2.1, строгий конверт)",
        gap <= env + 1e-12,
        f"|розрив|={gap:.2e} <= E_Q|b|^3/(1-|b|)={env:.2e}; Q(кліп)={clip_mass:.1e}",
    )
    check(f"d={dfrac}sd: kappa у [0,1]", 0.0 <= kappa <= 1.0)

# MC-крос-чек квадратури при d = 0.05*sd (та сама T1-ставка)
d = 0.05 * sd
lam, grow_pred, kappa, kl, qg = bet_terms(d)
bc_g, eta, _, _, growth_quad, _ = t1_bet(lam, qg)
N_MC = 2_000_000
Xq = sample_p0(N_MC) + d
b_mc = np.clip(lam[0] * Xq + lam[1] * (Xq ** 2 - u2), -CLIP, CLIP) - eta
growth_mc = float(np.mean(np.log1p(b_mc)))
se = float(np.std(np.log1p(b_mc), ddof=1) / np.sqrt(N_MC))
check(
    f"MC-крос-чек (n={N_MC:,}): ріст_MC ~= ріст_quad",
    abs(growth_mc - growth_quad) <= 4 * se,
    f"{growth_mc:.4e} vs {growth_quad:.4e} (4SE={4*se:.1e})",
)

# збіжність дефіциту до 1-kappa при d -> 0 (Теор. 4.2 — локальний шар, O(d))
errs = [abs(r[2] - (1.0 - r[1])) for r in rows]
check(
    "дефіцит/KL -> (1-kappa): похибка строго спадає при d -> 0",
    errs[0] < errs[1] < errs[2],
    f"|розбіжність| при d/sd=0.02/0.05/0.10: {errs[0]:.4f}/{errs[1]:.4f}/{errs[2]:.4f}",
)
check(
    "дефіцит/KL ~= (1-kappa) при найменшому d (толеранс 0.03 абс.)",
    errs[0] < 0.03,
    f"{rows[0][2]:.4f} vs {1-rows[0][1]:.4f}",
)
ratio_lin = errs[0] / errs[1] if errs[1] > 0 else np.nan
check(
    "порядок збіжності узгоджений з O(d) (err(0.02)/err(0.05) ~ 0.4)",
    0.15 <= ratio_lin <= 0.85,
    f"вимірено {ratio_lin:.2f}",
)

print("=" * 72)
if FAILURES:
    print(f"ПІДСУМОК: {len(FAILURES)} FAIL: {FAILURES}")
    raise SystemExit(1)
print("ПІДСУМОК: ALL PASS (sanity-check теорії; гейти BET-1..5 — окремо, за Spec)")
