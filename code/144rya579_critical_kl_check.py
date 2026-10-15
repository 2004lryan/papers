#!/usr/bin/env python3
r"""
144 — RYA-579：临界档下界改走「精确高斯 KL」之后，核心不等式的数值核验。

背景（取证 F001，推导取证，HP-DERIVATION-INVALID）
-------------------------------------------------------------------
旧证明对可恢复档与临界档合并取 δ_n=c'(nλmin)^{-1/2}，并以「在 LAN 邻域内合法」+
lem:laqf 余项作为 KL 有界的依据。临界档 nλmin→γ'∈(0,∞) 使 δ_n=Θ(1)，是固定分离；
且 lem:laqf 是对数似然比【过程】的依概率陈述，而 KL 是期望，两者之间缺一致可积性。

新证明改走恒等式：A1 下模型是已知尺度的高斯位移模型，故
    KL(P_θ1^{⊗n} ‖ P_θ2^{⊗n}) = (n/2σ²)·E_{Q_n}[(β(·;θ1)−β(·;θ2))²]      ……(恒等式)
再用 log k 对 θ 仿射这一结构事实：∂_vdeg u 与 θ 无关、|∂_u β|=|g|≤e^{-1} 在
Θ×supp(Q_n) 上一致有界，故 sup_Θ sup_supp |∂_vdeg β| = O(√v_n)，于是
    KL = O(n δ² v_n) = O(δ² Ψ_n)                                        ……(eq:klexact)
该式对固定分离与收缩分离一视同仁，三档一并裁定。

本脚本核验四件（全部只用 numpy，不依赖任何既有产物）：
  A ∂_vdeg u 逐点与 θ 无关（仿射性），且 sup|∂_vdeg β| / √v_n 随 ΔT→0 保持有界。
  B (eq:klexact)：对【固定】δ=Θ(1)，KL/(δ²Ψ_n) 随 ΔT→0 保持有界（不发散）。
  C 临界标度 ΔT_n ∝ n^{-1/2} 上，Ψ_n→γ 且固定 δ 的 KL 保持 O(1)，
    即 Le Cam 双点在临界档给出 Θ(1) 下界所需的正是这一条。
  D 反面对照，结论与预期相反，如实记录：同一固定 δ 下，旧路线所用的 Fisher 二次近似
    ½(2δ)²·n·⟨vdeg,I₁vdeg⟩ 相对精确 KL 的误差【极小且随 n 减小】（δ 扫到 4.0 仍 ≤6e-5）。
    也就是说 finding 给的理由——「Taylor 余项不足以支撑固定非局部位移下的 Fisher/KL 近似」
    ——在本模型里【不成立】：log k 的仿射性把退化方向的二阶/三阶导压到 O(v_n)/O(v_n^{3/2})，
    近似在 Θ(1) 位移下照样准。finding 的【结论】仍然成立（那一步确实没兑现临界档下界），
    但成立的理由是另一个：lem:laqf 是对【过程】的依概率陈述，拿它去认证一个【期望】(KL)
    需要一致可积性，稿面从未建立；且把 Θ(1) 位移称作「局部」本身会误导读者。
    新路线绕开的正是这两点，而不是去修一个其实很准的近似。

输出 04outputs/144rya579_critical_kl_check.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R_GAS = 8.314
T0_C, TREF_C = 2.0, 6.0
T0, TREF = T0_C + 273.15, TREF_C + 273.15
SIGMA = 0.02
T_MIN, T_MAX = 12.0, 96.0
NQ = 4001            # 设计测度求积节点（u×t 各自均匀）


def design(dT):
    """Q_n：u=(T−T0)/ΔT 与 t 上各自均匀，张量积求积节点。"""
    m = int(np.sqrt(NQ))
    u = (np.arange(m) + 0.5) / m
    tt = T_MIN + (T_MAX - T_MIN) * (np.arange(m) + 0.5) / m
    T = T0 + dT * u
    TT, T_ = np.meshgrid(tt, T, indexing="ij")
    return T_.ravel(), TT.ravel()


def beta(T, t, phi, logA):
    """β = 1 − exp(−k t)，log k = log A − φ r，r = T_ref/T。"""
    r = TREF / T
    k = np.exp(logA - phi * r)
    return 1.0 - np.exp(-k * t)


def g_of(T, t, phi, logA):
    r = TREF / T
    k = np.exp(logA - phi * r)
    return k * t * np.exp(-k * t)


def fisher_and_vdeg(T, t, phi, logA):
    r = TREF / T
    w = g_of(T, t, phi, logA) ** 2 / SIGMA ** 2
    Sw = w.mean()
    mw = (w * r).mean() / Sw
    qw = (w * r * r).mean() / Sw
    vn = qw - mw * mw
    I1 = np.array([[(w * r * r).mean(), -(w * r).mean()], [-(w * r).mean(), w.mean()]]) / 1.0
    evals, evecs = np.linalg.eigh(I1)
    return I1, evals[0], evecs[:, 0], Sw, mw, vn


def exact_kl(n, T, t, th1, th2):
    d = beta(T, t, *th1) - beta(T, t, *th2)
    return n / (2 * SIGMA ** 2) * float((d ** 2).mean())


def main() -> int:
    phi0, logA0 = 34.0, 31.0
    rec = {"任务": "RYA-579", "用途": "支撑定理 thm:c5f 下界改走精确高斯 KL 后的核心不等式 eq:klexact"}

    # ---- A 仿射性与结构性压低 -------------------------------------------------
    A = []
    for dT in (8.0, 4.0, 2.0, 1.0, 0.5, 0.25):
        T, t = design(dT)
        I1, lmin, vdeg, Sw, mw, vn = fisher_and_vdeg(T, t, phi0, logA0)
        r = TREF / T
        du = vdeg[0] * (-r) + vdeg[1] * np.ones_like(r)   # 解析式：∂_vdeg u = vdeg·(−r,1)
        # ∂_vdeg u 与 θ 无关：不是拿 du 跟自己比（那是恒真），而是在若干个不同的 θ 上
        # 用中心差分重算 (u(θ+h·vdeg) − u(θ−h·vdeg))/(2h)，再与解析式 du 逐点比。
        h = 1e-4
        du_err = 0.0
        max_dbeta = 0.0
        for dphi, dlogA in ((0.0, 0.0), (3.0, 3.0), (-3.0, -3.0), (5.0, -5.0)):
            ph, la = phi0 + dphi, logA0 + dlogA
            u_p = (la + h * vdeg[1]) - (ph + h * vdeg[0]) * r
            u_m = (la - h * vdeg[1]) - (ph - h * vdeg[0]) * r
            du_fd = (u_p - u_m) / (2 * h)
            du_err = max(du_err, float(np.max(np.abs(du_fd - du))))
            gg = g_of(T, t, ph, la)
            max_dbeta = max(max_dbeta, float(np.max(np.abs(gg * du))))
        A.append({
            "deltaT_K": dT, "v_n": vn, "lambda_min(I1)": float(lmin),
            "sup|d_vdeg u|/sqrt(v_n)": float(np.max(np.abs(du)) / np.sqrt(vn)),
            "sup_Theta sup_supp |d_vdeg beta|/sqrt(v_n)": float(max_dbeta / np.sqrt(vn)),
            "d_vdeg_u_中心差分_vs_解析_最大绝对误差(4个θ)": du_err,
            "d_vdeg_u_theta_independent": bool(du_err < 1e-9),
        })
    ratios = [a["sup_Theta sup_supp |d_vdeg beta|/sqrt(v_n)"] for a in A]
    rec["A_结构性压低"] = {
        "逐档": A,
        "比值展布倍数": max(ratios) / min(ratios),
        "d_vdeg_u_theta无关_全档通过": all(a["d_vdeg_u_theta_independent"] for a in A),
        "AGREES": bool(max(ratios) / min(ratios) < 3.0
                       and all(a["d_vdeg_u_theta_independent"] for a in A)),
        "判据": "sup|∂_vdeg β|/√v_n 在 ΔT 跨 32 倍时保持同一量级 -> O(√v_n) 成立；"
                "且 ∂_vdeg u 的中心差分在 4 个不同 θ 上与解析式一致 -> 与 θ 无关",
    }

    # ---- B (eq:klexact) 对固定 δ ---------------------------------------------
    B = []
    n = 20000
    delta = 0.30                      # Θ(1) 固定分离，不随 ΔT 收缩
    for dT in (8.0, 4.0, 2.0, 1.0, 0.5, 0.25):
        T, t = design(dT)
        I1, lmin, vdeg, Sw, mw, vn = fisher_and_vdeg(T, t, phi0, logA0)
        Psi = n * lmin
        th1 = (phi0 + delta * vdeg[0], logA0 + delta * vdeg[1])
        th2 = (phi0 - delta * vdeg[0], logA0 - delta * vdeg[1])
        kl = exact_kl(n, T, t, th1, th2)
        B.append({"deltaT_K": dT, "Psi_n": float(Psi), "delta": delta,
                  "KL_exact": kl, "KL/(delta^2 Psi_n)": kl / (delta ** 2 * Psi)})
    q = [b["KL/(delta^2 Psi_n)"] for b in B]
    rec["B_eq_klexact_固定分离"] = {
        "逐档": B, "比值展布倍数": max(q) / min(q),
        "AGREES": bool(max(q) / min(q) < 3.0),
        "判据": "KL/(δ²Ψ_n) 在固定 δ 下不随 ΔT→0 发散 -> KL=O(δ²Ψ_n) 对非收缩分离成立",
    }

    # ---- C 临界标度上 Ψ_n→γ 且固定 δ 的 KL 保持 O(1) ---------------------------
    C = []
    dT0, n0 = 8.0, 500
    for n in (500, 2000, 8000, 32000, 128000):
        dT = dT0 * (n / n0) ** -0.5          # ρ=1/2 临界标度
        T, t = design(dT)
        I1, lmin, vdeg, Sw, mw, vn = fisher_and_vdeg(T, t, phi0, logA0)
        Psi = n * lmin
        d = min(0.30, 0.5)                    # 固定分离，封顶于 r_0
        th1 = (phi0 + d * vdeg[0], logA0 + d * vdeg[1])
        th2 = (phi0 - d * vdeg[0], logA0 - d * vdeg[1])
        kl = exact_kl(n, T, t, th1, th2)
        quad = 0.5 * (2 * d) ** 2 * n * float(vdeg @ I1 @ vdeg)
        C.append({"n": n, "deltaT_K": dT, "Psi_n": float(Psi), "delta": d,
                  "KL_exact": kl, "Fisher_quadratic": quad,
                  "quad/exact": quad / kl})
    psis = [c["Psi_n"] for c in C]
    kls = [c["KL_exact"] for c in C]
    rec["C_临界标度"] = {
        "逐档": C,
        "Psi_n_展布倍数": max(psis) / min(psis),
        "KL_展布倍数": max(kls) / min(kls),
        "AGREES": bool(max(kls) / min(kls) < 3.0),
        "判据": "ρ=1/2 上 Ψ_n→γ 有界，固定 δ 的精确 KL 亦有界 -> Le Cam 给出临界档 Θ(1) 下界",
    }

    # ---- D 反面对照：finding 给的理由经实测不成立，如实记录 -------------------
    rr = [c["quad/exact"] for c in C]
    sweep = []
    for d in (0.05, 0.30, 1.0, 2.0, 4.0):
        for nn in (2000, 128000):
            dT = dT0 * (nn / n0) ** -0.5
            T, t = design(dT)
            I1, lmin, vdeg, Sw, mw, vn = fisher_and_vdeg(T, t, phi0, logA0)
            th1 = (phi0 + d * vdeg[0], logA0 + d * vdeg[1])
            th2 = (phi0 - d * vdeg[0], logA0 - d * vdeg[1])
            kl = exact_kl(nn, T, t, th1, th2)
            quad = 0.5 * (2 * d) ** 2 * nn * float(vdeg @ I1 @ vdeg)
            sweep.append({"delta": d, "n": nn, "Psi_n": float(nn * lmin),
                          "KL_exact": kl, "quad": quad, "rel_err": quad / kl - 1.0})
    worst = max(abs(x["rel_err"]) for x in sweep)
    rec["D_finding给的理由经实测不成立"] = {
        "quad/exact_临界标度逐档": rr,
        "delta_扫描": sweep,
        "最大相对误差": worst,
        "finding原话": ("The displayed LAQ result is only uniform for bounded rescaled coordinates, "
                        "and its Taylor remainder does not by itself establish the stated Fisher/KL "
                        "approximation for a fixed, nonlocal parameter displacement."),
        "实测": ("Fisher 二次近似在固定 δ 直至 4.0 时相对精确 KL 的误差 <= %.1e，且随 n 增大而减小；"
                 "根源是 log k 对 θ 仿射，使退化方向的二阶/三阶导被压到 O(v_n)/O(v_n^{3/2})。"
                 "故 finding 所指的『近似不成立』在本模型里不成立。" % worst),
        "结论": ("finding 的【结论】成立——所示步骤确实没有兑现临界档下界；但【理由】不成立。"
                 "真正的缺口是：lem:laqf 是对数似然比【过程】的依概率陈述，用它认证一个【期望】(KL) "
                 "需要余项的一致可积性，稿面未建立；另外把 Θ(1) 位移写成『局部』会误导。"
                 "新路线走恒等式，两点一并绕开，且不依赖任何近似。"),
        "AGREES_近似其实很准": bool(worst < 1e-3),
    }

    rec["总判定"] = all(rec[k]["AGREES"] for k in ("A_结构性压低", "B_eq_klexact_固定分离", "C_临界标度"))
    dst = ROOT / "04outputs" / "144rya579_critical_kl_check.json"
    dst.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    for k in ("A_结构性压低", "B_eq_klexact_固定分离", "C_临界标度"):
        print("%-26s AGREES=%s  展布=%.3f" % (
            k, rec[k]["AGREES"],
            rec[k].get("比值展布倍数") or rec[k].get("KL_展布倍数")))
    print("D finding 给的理由经实测不成立：二次近似最大相对误差 %.2e（δ 扫到 4.0）"
          % rec["D_finding给的理由经实测不成立"]["最大相对误差"])
    print("总判定:", rec["总判定"], "->", dst.relative_to(ROOT))
    return 0 if rec["总判定"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
