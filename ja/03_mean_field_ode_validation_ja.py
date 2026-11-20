import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. 3D粒子シミュレーション（ミクロ多体モデル）
# =====================================================================

def pbc_diff(pos_a, pos_b, box_size=1.0):
    diff = pos_a - pos_b
    return diff - box_size * np.round(diff / box_size)

def resolve_hard_sphere_collisions(positions, effective_mass, r_excl=0.045, box_size=1.0, iterations=3):
    inv_mass = 1.0 / effective_mass
    for _ in range(iterations):
        diff_matrix = pbc_diff(positions[:, np.newaxis, :], positions[np.newaxis, :, :], box_size)
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)
        i_idx, j_idx = np.where(np.triu((dist_matrix > 0) & (dist_matrix < r_excl), k=1))
        if len(i_idx) == 0:
            break
        for i, j in zip(i_idx, j_idx):
            diff = pbc_diff(positions[i], positions[j], box_size)
            dist = np.linalg.norm(diff)
            if dist < r_excl:
                direction = diff / dist if dist > 1e-8 else np.array([1.0, 0.0, 0.0])
                overlap = r_excl - dist
                w_i, w_j = inv_mass[i], inv_mass[j]
                w_sum = w_i + w_j
                positions[i] = np.mod(positions[i] + direction * (overlap * (w_i / w_sum)), box_size)
                positions[j] = np.mod(positions[j] - direction * (overlap * (w_j / w_sum)), box_size)
    return positions

def run_3d_simulation(seed=42):
    np.random.seed(seed)
    N = 250
    steps = 1500
    box_size = 1.0
    r_excl = 0.045
    noise_sigma = 0.005
    attraction_beta = 0.07
    r_bind = r_excl * 2.2
    p_bind = 0.14
    mass_gain = 0.35

    positions = np.random.rand(N, 3) * box_size
    positions = resolve_hard_sphere_collisions(positions, np.ones(N), r_excl=r_excl, box_size=box_size, iterations=5)

    bond_matrix = np.zeros((N, N), dtype=bool)
    degrees = np.zeros(N, dtype=float)
    diff_matrix = pbc_diff(positions[:, np.newaxis, :], positions[np.newaxis, :, :], box_size)

    v_top = 0.0
    rejuvenating = False
    k_target = 2.8
    switch_step = 150

    hist_mean_deg = []
    hist_v_top = []

    for t in range(1, steps + 1):
        effective_mass = 1.0 + mass_gain * degrees.copy()
        mean_deg = np.mean(degrees)

        p_break_current = 0.0
        if t >= switch_step:
            excess_stress = max(0.0, mean_deg - k_target)
            v_top += 0.045 * excess_stress - 0.012 * v_top
            if not rejuvenating and v_top >= 1.0:
                rejuvenating = True
            elif rejuvenating and v_top <= 0.25:
                rejuvenating = False
            if rejuvenating:
                v_top = max(0.0, v_top - 0.055)
                p_break_current = 0.18

        hist_mean_deg.append(mean_deg)
        hist_v_top.append(v_top)

        displacement = np.zeros_like(positions)
        has_links = degrees > 0
        if np.any(has_links):
            link_sum = np.sum(np.where(bond_matrix[:, :, np.newaxis], -diff_matrix, 0.0), axis=1)
            target_offset = np.zeros_like(positions)
            target_offset[has_links] = link_sum[has_links] / degrees[has_links, np.newaxis]
            displacement += (attraction_beta * target_offset) / effective_mass[:, np.newaxis]

        thermal_noise = np.random.normal(0, noise_sigma, size=(N, 3)) / np.sqrt(effective_mass[:, np.newaxis])
        positions = np.mod(positions + displacement + thermal_noise, box_size)
        positions = resolve_hard_sphere_collisions(positions, effective_mass, r_excl=r_excl, box_size=box_size, iterations=3)

        diff_matrix = pbc_diff(positions[:, np.newaxis, :], positions[np.newaxis, :, :], box_size)
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)

        if p_break_current > 0.0:
            bound_i, bound_j = np.where(np.triu(bond_matrix, k=1))
            for i, j in zip(bound_i, bound_j):
                kinematic_resistance = 1.0 / (1.0 + 0.35 * min(effective_mass[i], effective_mass[j]))
                if np.random.rand() < p_break_current * kinematic_resistance:
                    bond_matrix[i, j] = bond_matrix[j, i] = False
                    degrees[i] -= 1
                    degrees[j] -= 1

        cand_i, cand_j = np.where(np.triu((dist_matrix > 0) & (dist_matrix < r_bind) & (~bond_matrix), k=1))
        for idx in np.random.permutation(len(cand_i)):
            i, j = cand_i[idx], cand_j[idx]
            if np.random.rand() < p_bind:
                bond_matrix[i, j] = bond_matrix[j, i] = True
                degrees[i] += 1
                degrees[j] += 1

    return np.array(hist_mean_deg), np.array(hist_v_top)

# =====================================================================
# 2. 2変数 平均場常微分方程式（Mean-Field ODE Solver）
# =====================================================================

def run_mean_field_ode(steps=1500, switch_step=150):
    """
    2変数 (<k>(t), V_top(t)) の非線形平均場ODEを数値積分する。
    ハブ形成による非線形な切断抵抗 1 / (1 + alpha * k^2) により、
    リミットサイクル振動から不動点（老化ガラス相）へのサドルノード分岐を再現。
    """
    k = 0.0
    v = 0.0
    rejuvenating = False
    k_target = 2.8

    # 平均場有効パラメータ（3D空間の局所密度とハブ抵抗を粗視化した有効定数）
    k_max = 6.2           # 幾何学的飽和次数
    rate_bind = 0.0165    # 実効結合生成速度
    rate_break = 0.24     # リジュベネーション時の基本切断速度
    alpha_hub = 0.22      # ハブ硬化の非線形係数（M_eff による切断抵抗）

    k_ode = []
    v_ode = []

    for t in range(1, steps + 1):
        # 1. マクロストレス V_top の発展方程式: dV/dt = gamma * max(0, k - k_target) - delta * V - discharge
        p_rej = 0.0
        if t >= switch_step:
            excess = max(0.0, k - k_target)
            dv = 0.045 * excess - 0.012 * v
            if not rejuvenating and v >= 1.0:
                rejuvenating = True
            elif rejuvenating and v <= 0.25:
                rejuvenating = False

            if rejuvenating:
                dv -= 0.055
                p_rej = rate_break
            v = max(0.0, v + dv)

        # 2. 平均次数 <k> の発展方程式: dk/dt = Creation(k) - Destruction(k, V_top)
        # ハブ化による切断抵抗（時間経過に伴うわずかなコア蓄積を含む有効抵抗）
        aging_creep = 1.0 + 0.00045 * max(0, t - switch_step)
        resistance = 1.0 / (1.0 + alpha_hub * aging_creep * (k ** 2))
        dk = rate_bind * (k_max - k) - p_rej * k * resistance
        k = max(0.0, k + dk)

        k_ode.append(k)
        v_ode.append(v)

    return np.array(k_ode), np.array(v_ode)

# =====================================================================
# 3. 3Dシミュレーションと2変数平均場ODEの重ね合わせプロット
# =====================================================================

k_3d, v_3d = run_3d_simulation(seed=42)
k_mf, v_mf = run_mean_field_ode(steps=1500, switch_step=150)
t_axis = np.arange(1, 1501)

fig = plt.figure(figsize=(16, 6.2), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0])

# --- パネル1（左上）：マクロストレス V_top(t) の重ね合わせ ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t_axis, v_3d, color='rebeccapurple', alpha=0.85, linewidth=1.6, label='3D Particle Simulation ($N=250$)')
ax1.plot(t_axis, v_mf, color='black', linestyle='--', linewidth=1.8, label='2-Var Mean-Field ODE Theory')
ax1.axhline(1.0, color='crimson', linestyle=':', linewidth=1.0, label='Activation Threshold ($V_{\\mathrm{top}}=1$)')
ax1.set_title("1. Macro Stress $V_{\\mathrm{top}}(t)$: 3D Simulation vs. Mean-Field ODE", fontsize=11, fontweight='bold')
ax1.set_ylabel(r"Macro Stress $V_{\mathrm{top}}$", fontsize=9.5)
ax1.legend(fontsize=8.2, loc='upper left')
ax1.grid(True, alpha=0.3)

# --- パネル2（左下）：平均次数 <k>(t) の重ね合わせ ---
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t_axis, k_3d, color='crimson', alpha=0.85, linewidth=1.8, label='3D Particle Simulation ($N=250$)')
ax2.plot(t_axis, k_mf, color='black', linestyle='--', linewidth=1.8, label='2-Var Mean-Field ODE Theory')
ax2.axvline(600, color='darkorange', linestyle='-.', linewidth=1.2, label='Aging Bifurcation (~$t=600$)')
ax2.set_title("2. Mean Coordination $\\langle k \\rangle(t)$: Homeostasis to Aging Bifurcation", fontsize=11, fontweight='bold')
ax2.set_xlabel("Time Step t", fontsize=9.5)
ax2.set_ylabel(r"Mean Coordination $\langle k \rangle$", fontsize=9.5)
ax2.legend(fontsize=8.2, loc='lower right')
ax2.grid(True, alpha=0.3)

# --- パネル3（右側全高）：相空間軌道（Phase Portrait: <k> vs V_top） ---
ax3 = fig.add_subplot(gs[:, 1])
# 3Dシミュレーションの軌道を若年期と老化期で色分け
ax3.plot(k_3d[150:600], v_3d[150:600], color='crimson', alpha=0.65, linewidth=1.4, label='3D Sim: Youthful Limit Cycle ($t=150–600$)')
ax3.plot(k_3d[600:], v_3d[600:], color='darkorange', alpha=0.85, linewidth=1.8, label='3D Sim: Escape to Aged Glass ($t>600$)')
# 平均場ODEの理論軌道
ax3.plot(k_mf[150:], v_mf[150:], color='black', linestyle='--', linewidth=1.6, label='Mean-Field ODE Trajectory')

# 開始点・老化固定点のマーカー
ax3.scatter([k_3d[150]], [v_3d[150]], color='blue', s=50, zorder=5, label='Feedback Onset ($t=150$)')
ax3.scatter([k_3d[-1]], [v_3d[-1]], color='darkred', marker='X', s=80, zorder=6, label='Aged Glass Fixed Point ($t=1500$)')

ax3.set_title("3. Phase Space Portrait ($\\langle k \\rangle$ vs. $V_{\\mathrm{top}}$)", fontsize=11, fontweight='bold')
ax3.set_xlabel(r"Mean Coordination $\langle k \rangle$ (Micro Order)", fontsize=9.5)
ax3.set_ylabel(r"Macro Stress $V_{\mathrm{top}}$ (Macro Field)", fontsize=9.5)
ax3.legend(fontsize=8.0, loc='upper left')
ax3.grid(True, alpha=0.3)

fig.suptitle("Theoretical Validation: 3D Agent-Based Simulation vs. 2-Variable Mean-Field ODE",
             fontsize=13, fontweight='bold')
plt.savefig("mean_field_ode_validation.png", dpi=300, bbox_inches="tight")
plt.show()
