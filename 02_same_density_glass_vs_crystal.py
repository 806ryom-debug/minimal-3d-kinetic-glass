import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 0. Common Physics & Topological Analysis Functions
# =====================================================================

def pbc_diff(pos_a, pos_b, box_size=1.0):
    diff = pos_a - pos_b
    return diff - box_size * np.round(diff / box_size)

def resolve_hard_sphere_collisions(positions, unwrapped_pos, effective_mass, r_excl, box_size=1.0, iterations=6):
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
                if dist < 1e-8:
                    direction = np.random.randn(3)
                    direction /= np.linalg.norm(direction)
                else:
                    direction = diff / dist

                overlap = r_excl - dist
                w_i, w_j = inv_mass[i], inv_mass[j]
                w_sum = w_i + w_j

                shift_i = direction * (overlap * (w_i / w_sum))
                shift_j = -direction * (overlap * (w_j / w_sum))

                positions[i] = np.mod(positions[i] + shift_i, box_size)
                positions[j] = np.mod(positions[j] + shift_j, box_size)
                unwrapped_pos[i] += shift_i
                unwrapped_pos[j] += shift_j

    return positions, unwrapped_pos

def calculate_rdf(positions, num_bins=55, r_max=0.5, box_size=1.0):
    N = len(positions)
    diff = pbc_diff(positions[:, np.newaxis, :], positions[np.newaxis, :, :], box_size)
    dists = np.linalg.norm(diff, axis=2)
    pair_distances = dists[np.triu_indices(N, k=1)]

    counts, bin_edges = np.histogram(pair_distances, bins=num_bins, range=(1e-5, r_max))
    dr = bin_edges[1] - bin_edges[0]
    radii = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    rho = N / (box_size ** 3)

    shell_volumes = 4.0 * np.pi * (radii ** 2) * dr
    expected_counts = shell_volumes * rho * N / 2.0
    rdf = np.divide(counts, expected_counts, out=np.zeros_like(radii), where=expected_counts > 0)
    return radii, rdf

def analyze_cna(bond_matrix):
    cna_signatures = {}
    rows, cols = np.where(np.triu(bond_matrix, k=1))
    for i, j in zip(rows, cols):
        common_nbs = np.where(bond_matrix[i] & bond_matrix[j])[0]
        num_common = len(common_nbs)
        num_bonds = 0
        if num_common > 1:
            sub_adj = bond_matrix[np.ix_(common_nbs, common_nbs)]
            num_bonds = int(np.sum(np.triu(sub_adj, k=1)))
        signature = (num_common, num_bonds)
        cna_signatures[signature] = cna_signatures.get(signature, 0) + 1
    return cna_signatures

# =====================================================================
# 1. Generate Common Initial State (Identical Density & Configuration)
# =====================================================================

def create_initial_dense_fluid(N=256, r_target=0.158, box_size=1.0, seed=42):
    """Prepare an identical initial dense fluid (packing fraction ≈ 48%) for both glass and crystal."""
    np.random.seed(seed)
    positions = np.random.rand(N, 3) * box_size
    unwrapped = positions.copy()
    for r_step in np.linspace(0.080, r_target, 30):
        step_noise = np.random.normal(0, 0.005, size=(N, 3))
        positions = np.mod(positions + step_noise, box_size)
        unwrapped += step_noise
        positions, unwrapped = resolve_hard_sphere_collisions(
            positions, unwrapped, np.ones(N), r_excl=r_step, box_size=box_size, iterations=12
        )
    return positions, unwrapped

# =====================================================================
# 2. Identical Rules & Density Simulation (Differing Only in Cooling Rate)
# =====================================================================

def run_thermal_history_simulation(init_pos, init_unwrapped, cooling_mode="quench", seed=42):
    np.random.seed(seed)
    N = len(init_pos)
    steps = 400
    box_size = 1.0
    r_excl = 0.158            # Identical density (packing fraction ≈ 48%)

    # --- Base parameters completely shared between glass and crystal ---
    noise_init = 0.022        # Initial high-temperature noise
    noise_final = 0.0005      # Final low-temperature noise (solidification)
    attraction_beta = 0.13    # Short-range bond attraction
    r_bind = r_excl * 1.25    # Capture radius of the first coordination shell
    r_break = r_excl * 1.32   # Bond-breaking threshold for off-lattice links
    p_bind = 0.35
    p_break = 0.35
    mass_gain = 0.05          # Degree-dependent kinematic constraint coefficient

    positions = init_pos.copy()
    unwrapped_pos = init_unwrapped.copy()
    base_unwrapped = unwrapped_pos.copy()
    initial_rdf = calculate_rdf(positions, box_size=box_size)

    bond_matrix = np.zeros((N, N), dtype=bool)
    degrees = np.zeros(N, dtype=float)
    diff_matrix = pbc_diff(positions[:, np.newaxis, :], positions[np.newaxis, :, :], box_size)

    sq_disp_history = []

    for t in range(1, steps + 1):
        # [Sole control variable: Thermal history (cooling schedule)]
        if cooling_mode == "quench":
            # Quench: Rapidly drop to ultra-low temperature in the initial steps to arrest structure
            progress = min(1.0, t / 10.0)
            current_noise = noise_init * (1.0 - progress) + noise_final * progress
        else:
            # Anneal: Hold at high temperature for the first 200 steps to grow crystal nuclei, then slowly cool over the remaining 200 steps
            if t <= 200:
                current_noise = noise_init
            else:
                progress = (t - 200) / 200.0
                current_noise = noise_init * (1.0 - progress) + noise_final * progress

        effective_mass = 1.0 + mass_gain * degrees.copy()
        displacement = np.zeros_like(positions)

        # 1. Short-range bond attraction
        has_links = degrees > 0
        if np.any(has_links):
            link_sum = np.sum(np.where(bond_matrix[:, :, np.newaxis], -diff_matrix, 0.0), axis=1)
            target_offset = np.zeros_like(positions)
            target_offset[has_links] = link_sum[has_links] / degrees[has_links, np.newaxis]
            displacement += (attraction_beta * target_offset) / effective_mass[:, np.newaxis]

        # 2. Thermal fluctuations (following cooling schedule)
        thermal_noise = np.random.normal(0, current_noise, size=(N, 3)) / np.sqrt(effective_mass[:, np.newaxis])
        step_delta = displacement + thermal_noise

        positions = np.mod(positions + step_delta, box_size)
        unwrapped_pos += step_delta

        # 3. Hard-sphere collision projection
        positions, unwrapped_pos = resolve_hard_sphere_collisions(
            positions, unwrapped_pos, effective_mass, r_excl=r_excl, box_size=box_size, iterations=8
        )

        # 4. Record Mean Squared Displacement (MSD) corrected for center-of-mass drift
        rel_disp = (unwrapped_pos - base_unwrapped) - np.mean(unwrapped_pos - base_unwrapped, axis=0)
        sq_disp_history.append(np.sum(rel_disp ** 2, axis=1))

        # 5. Update pairwise distance matrix
        diff_matrix = pbc_diff(positions[:, np.newaxis, :], positions[np.newaxis, :, :], box_size)
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)

        # 6. Reversible bond breaking (thermal dissociation of overstretched bonds; frequency naturally drops at low T)
        thermal_break_prob = p_break * (current_noise / noise_init)
        rows, cols = np.where(np.triu(bond_matrix, k=1))
        for i, j in zip(rows, cols):
            if dist_matrix[i, j] > r_break and np.random.rand() < thermal_break_prob:
                bond_matrix[i, j] = False
                bond_matrix[j, i] = False
                degrees[i] -= 1
                degrees[j] -= 1

        # 7. Bond formation between proximate pairs
        cand_i, cand_j = np.where(np.triu((dist_matrix > 0) & (dist_matrix < r_bind) & (~bond_matrix), k=1))
        for idx in np.random.permutation(len(cand_i)):
            i, j = cand_i[idx], cand_j[idx]
            if np.random.rand() < p_bind:
                bond_matrix[i, j] = True
                bond_matrix[j, i] = True
                degrees[i] += 1
                degrees[j] += 1

    sq_disp_arr = np.array(sq_disp_history)
    msd_total = np.mean(sq_disp_arr, axis=1)
    core_mask = degrees >= 6
    msd_core = np.mean(sq_disp_arr[:, core_mask], axis=1) if np.any(core_mask) else np.full(steps, np.nan)

    return {
        "positions": positions,
        "bond_matrix": bond_matrix,
        "degrees": degrees,
        "initial_rdf": initial_rdf,
        "final_rdf": calculate_rdf(positions, box_size=box_size),
        "cna_res": analyze_cna(bond_matrix),
        "msd_total": msd_total,
        "msd_core": msd_core,
        "r_excl": r_excl,
        "steps": steps
    }

# =====================================================================
# 3. Run Simulations & Render Comparison Plot (2 Rows x 4 Columns)
# =====================================================================

init_pos, init_unwrapped = create_initial_dense_fluid(N=256, r_target=0.158, seed=42)

configs = [
    {
        "mode": "quench",
        "title": "A. Quenched Glass (Fast Cooling, Same Density)\n[Kinetically Trapped Amorphous Solid]",
        "color": "crimson",
        "cmap": "plasma"
    },
    {
        "mode": "anneal",
        "title": "B. Annealed Crystal (Slow Cooling, Same Density)\n[Thermodynamically Relaxed Crystal]",
        "color": "forestgreen",
        "cmap": "viridis"
    }
]

results = [run_thermal_history_simulation(init_pos, init_unwrapped, cfg["mode"], seed=42) for cfg in configs]
max_msd = max(np.nanmax(r["msd_total"]) for r in results) * 1.15

fig = plt.figure(figsize=(19, 9.5), constrained_layout=True)
gs = fig.add_gridspec(2, 4, width_ratios=[1.1, 1, 1, 1.05])

for row_idx, (cfg, res) in enumerate(zip(configs, results)):
    pos = res["positions"]
    gf = res["bond_matrix"]
    deg = res["degrees"]
    r_excl = res["r_excl"]
    t_axis = np.arange(1, res["steps"] + 1)

    # --- Column 1: 3D Structure ---
    ax1 = fig.add_subplot(gs[row_idx, 0], projection='3d')
    sc = ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=deg, cmap=cfg["cmap"],
                     vmin=0, vmax=12, s=65, alpha=0.88, edgecolors='k', linewidths=0.4)
    rows, cols = np.where(np.triu(gf, k=1))
    for i, j in zip(rows, cols):
        if np.linalg.norm(pos[i] - pos[j]) < 0.45:
            ax1.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], [pos[i, 2], pos[j, 2]],
                     color='gray', alpha=0.28, linewidth=0.8)
    ax1.set_title(cfg["title"], fontsize=10.5, fontweight='bold')
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_zlim(0, 1)
    cbar = plt.colorbar(sc, ax=ax1, shrink=0.65, pad=0.08)
    cbar.set_label("Coordination (Degree)", fontsize=8)

    # --- Column 2: Normalized RDF g(r / r_excl) ---
    ax2 = fig.add_subplot(gs[row_idx, 1])
    r_init, rdf_init = res["initial_rdf"]
    r_final, rdf_final = res["final_rdf"]
    ax2.plot(r_init / r_excl, rdf_init, 'o--', color='lightgray', markersize=3, label='Initial Dense Fluid')
    ax2.plot(r_final / r_excl, rdf_final, 's-', color=cfg["color"], markersize=3.5, linewidth=1.6, label='Final Solid')
    ax2.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    for mult, lbl in [(1.0, '1'), (np.sqrt(2), '√2'), (np.sqrt(3), '√3'), (2.0, '2')]:
        ax2.axvline(mult, color='black', linestyle='--', linewidth=0.7, alpha=0.4)
    ax2.set_title(f"Normalized RDF g(r / d) — Mean Deg: {np.mean(deg):.1f}", fontsize=10, fontweight='bold')
    ax2.set_xlabel("Normalized Distance r / r_excl", fontsize=8.5)
    ax2.set_ylabel("g(r)", fontsize=8.5)
    ax2.set_xlim(0.7, 3.2)
    ax2.legend(fontsize=7.5)
    ax2.grid(True, alpha=0.3)

    # --- Column 3: Common Neighbor Analysis (CNA Top 8) ---
    ax3 = fig.add_subplot(gs[row_idx, 2])
    cna_res = res["cna_res"]
    if cna_res:
        sorted_cna = sorted(cna_res.items(), key=lambda x: x[1], reverse=True)[:8]
        labels = [str(k) for k, _ in sorted_cna]
        total_bonds = sum(cna_res.values())
        freqs = [(v / total_bonds) * 100 for _, v in sorted_cna]
        bars = ax3.bar(labels, freqs, color=cfg["color"], edgecolor='black', alpha=0.75)
        ax3.tick_params(axis='x', rotation=40, labelsize=8)
        for bar, f in zip(bars, freqs):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{f:.1f}%", ha='center', va='bottom', fontsize=7)
        ax3.set_ylim(0, max(freqs) * 1.2)
    ax3.set_title("CNA Topology (Top 8 Motifs)", fontsize=10, fontweight='bold')
    ax3.set_xlabel("Signature (CNs, Bonds)", fontsize=8.5)
    ax3.set_ylabel("Frequency (%)", fontsize=8.5)

    # --- Column 4: Mean Squared Displacement MSD(t) ---
    ax4 = fig.add_subplot(gs[row_idx, 3])
    ax4.plot(t_axis, res["msd_total"], color=cfg["color"], linewidth=2.2, label='System Mean MSD')
    ax4.plot(t_axis, res["msd_core"], color='gold', linestyle='--', linewidth=1.8, label='High-Deg Nodes (deg ≥ 6)')
    ax4.set_title(f"MSD(t) Arrest Plateau — Final: {res['msd_total'][-1]:.4f}", fontsize=10, fontweight='bold')
    ax4.set_xlabel("Time Step t", fontsize=8.5)
    ax4.set_ylabel(r"MSD $\langle |\Delta \mathbf{r}(t)|^2 \rangle$", fontsize=8.5)
    ax4.set_ylim(0, max_msd)
    ax4.legend(fontsize=7.5, loc='lower right')
    ax4.grid(True, alpha=0.3)

fig.suptitle("Same Density & Same Rules: Cooling Rate Governs Glass vs. Crystal Arrest",
             fontsize=14, fontweight='bold', y=1.02)
plt.savefig("same_density_glass_vs_crystal.png", dpi=300, bbox_inches="tight")
plt.show()
