import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 0. Common Physics & Topological Analysis Functions
# =====================================================================

def pbc_diff(pos_a, pos_b, box_size=1.0):
    """Minimum image convention: shortest displacement vector across periodic boundaries."""
    diff = pos_a - pos_b
    return diff - box_size * np.round(diff / box_size)


def resolve_hard_sphere_collisions(positions, unwrapped_pos, effective_mass, r_excl=0.045, box_size=1.0, iterations=3):
    """Hard-sphere collision projection (updates both PBC coordinates and continuous unwrapped coordinates)."""
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

                # Apply the exact same displacement to continuous coordinates for MSD calculation
                unwrapped_pos[i] += shift_i
                unwrapped_pos[j] += shift_j

    return positions, unwrapped_pos


def calculate_rdf(positions, num_bins=25, r_max=0.5, box_size=1.0):
    """Calculate the radial distribution function g(r)."""
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
    """Perform Common Neighbor Analysis (CNA) on the bond network."""
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
# 1. Simulation Engine (with Time-Series MSD Tracking)
# =====================================================================

def run_simulation(enable_mass=True, enable_attraction=True, seed=42):
    N = 250
    steps = 150
    box_size = 1.0
    r_excl = 0.045
    noise_sigma = 0.005
    attraction_beta = 0.07 if enable_attraction else 0.0
    r_bind = r_excl * 2.2
    p_bind = 0.15

    np.random.seed(seed)
    positions = np.random.rand(N, 3) * box_size
    unwrapped_pos = positions.copy()

    positions, unwrapped_pos = resolve_hard_sphere_collisions(
        positions, unwrapped_pos, np.ones(N), r_excl=r_excl, box_size=box_size, iterations=5
    )

    # Store reference positions (continuous coordinates) at t = 0
    initial_unwrapped = unwrapped_pos.copy()
    initial_rdf = calculate_rdf(positions, box_size=box_size)

    bond_matrix = np.zeros((N, N), dtype=bool)
    degrees = np.zeros(N, dtype=float)
    diff_matrix = pbc_diff(positions[:, np.newaxis, :], positions[np.newaxis, :, :], box_size)

    msd_total_history = []
    msd_core_history = []
    msd_iso_history = []

    for _ in range(1, steps + 1):
        effective_mass = (1.0 + degrees.copy()) if enable_mass else np.ones(N, dtype=float)
        displacement = np.zeros_like(positions)

        # 1. Short-range bond attraction
        has_links = degrees > 0
        if np.any(has_links) and attraction_beta > 0.0:
            link_sum = np.sum(np.where(bond_matrix[:, :, np.newaxis], -diff_matrix, 0.0), axis=1)
            target_offset = np.zeros_like(positions)
            target_offset[has_links] = link_sum[has_links] / degrees[has_links, np.newaxis]
            displacement += (attraction_beta * target_offset) / effective_mass[:, np.newaxis]

        # 2. Thermal fluctuations
        thermal_noise = np.random.normal(0, noise_sigma, size=(N, 3)) / np.sqrt(effective_mass[:, np.newaxis])
        step_delta = displacement + thermal_noise

        positions = np.mod(positions + step_delta, box_size)
        unwrapped_pos += step_delta

        # 3. Hard-sphere collision projection
        positions, unwrapped_pos = resolve_hard_sphere_collisions(
            positions, unwrapped_pos, effective_mass, r_excl=r_excl, box_size=box_size, iterations=3
        )

        # 4. Compute Mean Squared Displacement (MSD): <|r(t) - r(0)|^2>
        sq_disp = np.sum((unwrapped_pos - initial_unwrapped) ** 2, axis=1)
        msd_total_history.append(np.mean(sq_disp))

        # Track dynamic arrest (clustered hub nodes vs. isolated particles)
        if np.any(degrees >= 3):
            msd_core_history.append(np.mean(sq_disp[degrees >= 3]))
        else:
            msd_core_history.append(np.nan)

        if np.any(degrees == 0):
            msd_iso_history.append(np.mean(sq_disp[degrees == 0]))
        else:
            msd_iso_history.append(np.nan)

        # 5. Update pairwise distance matrix
        diff_matrix = pbc_diff(positions[:, np.newaxis, :], positions[np.newaxis, :, :], box_size)
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)

        # 6. Irreversible bond formation between proximate pairs
        cand_i, cand_j = np.where(np.triu((dist_matrix > 0) & (dist_matrix < r_bind) & (~bond_matrix), k=1))
        for idx in np.random.permutation(len(cand_i)):
            i, j = cand_i[idx], cand_j[idx]
            if np.random.rand() < p_bind:
                bond_matrix[i, j] = True
                bond_matrix[j, i] = True
                degrees[i] += 1
                degrees[j] += 1

    final_mass = (1.0 + degrees) if enable_mass else np.ones(N)
    final_rdf = calculate_rdf(positions, box_size=box_size)
    cna_res = analyze_cna(bond_matrix)

    return {
        "positions": positions,
        "bond_matrix": bond_matrix,
        "final_mass": final_mass,
        "initial_rdf": initial_rdf,
        "final_rdf": final_rdf,
        "cna_res": cna_res,
        "msd_total": np.array(msd_total_history),
        "msd_core": np.array(msd_core_history),
        "msd_iso": np.array(msd_iso_history),
        "r_excl": r_excl,
        "steps": steps
    }


# =====================================================================
# 2. Run 3 Conditions and Render Comparison Plot (3 Rows x 4 Columns)
# =====================================================================

experiments = [
    {
        "title": "1. Minimal Glass (All 4 Factors)\n[Amorphous Network Freeze]",
        "enable_mass": True,
        "enable_attraction": True,
        "color": "crimson"
    },
    {
        "title": "2. Ablation: Fixed Mass (M = 1)\n[Collapse into Liquid Droplet]",
        "enable_mass": False,
        "enable_attraction": True,
        "color": "darkorange"
    },
    {
        "title": "3. Ablation: Zero Attraction (β = 0)\n[Remains Unstructured Gas]",
        "enable_mass": True,
        "enable_attraction": False,
        "color": "royalblue"
    }
]

results = [run_simulation(exp["enable_mass"], exp["enable_attraction"]) for exp in experiments]

# Unify Y-axis scale across all conditions using global max MSD for direct comparison
global_max_msd = max(
    np.nanmax(res["msd_total"]) for res in results
) * 1.15
fig = plt.figure(figsize=(19, 13.5), constrained_layout=True)
gs = fig.add_gridspec(3, 4, width_ratios=[1.1, 1, 1, 1.05])

for row_idx, (exp, res) in enumerate(zip(experiments, results)):
    positions = res["positions"]
    bond_matrix = res["bond_matrix"]
    final_mass = res["final_mass"]
    r_excl = res["r_excl"]
    t_axis = np.arange(1, res["steps"] + 1)

    # --- Column 1: 3D Spatial Plot ---
    ax1 = fig.add_subplot(gs[row_idx, 0], projection='3d')
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                c=final_mass, cmap='plasma', vmin=1.0, vmax=10.0,
                s=22, alpha=0.9, edgecolors='k', linewidths=0.3)

    rows, cols = np.where(np.triu(bond_matrix, k=1))
    for i, j in zip(rows, cols):
        if np.linalg.norm(positions[i] - positions[j]) < 0.45:
            ax1.plot([positions[i, 0], positions[j, 0]],
                     [positions[i, 1], positions[j, 1]],
                     [positions[i, 2], positions[j, 2]],
                     color='gray', alpha=0.25, linewidth=0.6)

    ax1.set_title(exp["title"], fontsize=10.5, fontweight='bold', pad=8)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1); ax1.set_zlim(0, 1)
    ax1.tick_params(labelsize=7.5)

    # --- Column 2: Radial Distribution Function g(r) ---
    ax2 = fig.add_subplot(gs[row_idx, 1])
    r_init, rdf_init = res["initial_rdf"]
    r_final, rdf_final = res["final_rdf"]

    ax2.plot(r_init, rdf_init, 'o--', color='lightgray', markersize=3.5, label='Initial Gas')
    ax2.plot(r_final, rdf_final, 's-', color=exp["color"], markersize=4, linewidth=1.6, label='Final State')
    ax2.axhline(1.0, color='gray', linestyle=':', linewidth=1)
    ax2.axvline(r_excl, color='black', linestyle='--', linewidth=0.8, alpha=0.5, label=f'Hard Core ({r_excl})')

    ax2.set_title(f"Pair Distribution g(r) — Peak: {np.max(rdf_final):.1f}", fontsize=10, fontweight='bold')
    ax2.set_xlabel("Distance r", fontsize=8.5)
    ax2.set_ylabel("g(r)", fontsize=8.5)
    ax2.legend(fontsize=7.5, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=8)

    # --- Column 3: Common Neighbor Analysis (CNA Top 6) ---
    ax3 = fig.add_subplot(gs[row_idx, 2])
    cna_res = res["cna_res"]
    if cna_res:
        sorted_cna = sorted(cna_res.items(), key=lambda x: x[1], reverse=True)[:6]
        labels = [str(k) for k, _ in sorted_cna]
        total_bonds = sum(cna_res.values())
        freqs = [(v / total_bonds) * 100 for _, v in sorted_cna]
        bars = ax3.bar(labels, freqs, color=exp["color"], edgecolor='black', alpha=0.75)
        ax3.tick_params(axis='x', rotation=35, labelsize=8)
        for bar, f in zip(bars, freqs):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                     f"{f:.1f}%", ha='center', va='bottom', fontsize=7.5)
        ax3.set_ylim(0, max(freqs) * 1.22)
    ax3.set_title("CNA Topology (Top 6 Motifs)", fontsize=10, fontweight='bold')
    ax3.set_xlabel("Signature (CNs, Bonds)", fontsize=8.5)
    ax3.set_ylabel("Frequency (%)", fontsize=8.5)
    ax3.tick_params(axis='y', labelsize=8)

    # --- Column 4: Mean Squared Displacement MSD(t) (Quantitative Metric for Dynamic Arrest) ---
    ax4 = fig.add_subplot(gs[row_idx, 3])
    ax4.plot(t_axis, res["msd_total"], color=exp["color"], linewidth=2.2, label='System Mean MSD')
    ax4.plot(t_axis, res["msd_core"], color='gold', linestyle='--', linewidth=1.4, label='Clustered (deg ≥ 3)')
    ax4.plot(t_axis, res["msd_iso"], color='dimgray', linestyle=':', linewidth=1.2, label='Isolated (deg = 0)')
    final_msd = res["msd_total"][-1]
    ax4.set_title(f"Mean Squared Displacement — Final: {final_msd:.4f}", fontsize=10, fontweight='bold')
    ax4.set_xlabel("Time Step t", fontsize=8.5)
    ax4.set_ylabel(r"MSD $\langle |\Delta \mathbf{r}(t)|^2 \rangle$", fontsize=8.5)
    ax4.set_ylim(0, global_max_msd)
    ax4.legend(fontsize=7.5, loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=8)

fig.suptitle("Ablation Study with Dynamic Arrest Metric (MSD): Minimal Kinetic 3D Glassification",
             fontsize=14, fontweight='bold', y=1.02)
plt.savefig("ablation_study_with_msd.png", dpi=300, bbox_inches="tight")
plt.show()
