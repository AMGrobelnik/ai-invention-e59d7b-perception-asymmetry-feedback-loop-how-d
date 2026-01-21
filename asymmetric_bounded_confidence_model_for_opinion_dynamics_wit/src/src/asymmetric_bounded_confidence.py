"""
Asymmetric Bounded-Confidence Agent-Based Model for Opinion Dynamics.

This module implements a modified Hegselmann-Krause bounded-confidence model
where the confidence bound is modulated by perception asymmetry - agents
perceive their ingroup as more diverse than their outgroup.

Key hypothesis: Higher perception asymmetry (alpha) leads to more stable
opinion bubbles by discounting signals from outgroups perceived as homogeneous.
"""

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass
class Agent:
    """Agent with opinion and perception asymmetry parameters."""

    opinion: float  # continuous value in [-1, +1]
    ingroup_id: int = -1  # cluster assignment
    alpha: float = 0.5  # perception asymmetry

    def __post_init__(self):
        self.opinion = np.clip(self.opinion, -1.0, 1.0)


@dataclass
class ClusterMetrics:
    """Metrics for a single timestep."""

    timestep: int
    cluster_count: int
    smooth_metric: float  # c = 1 / sum(r_i^2)
    modularity_Q: float
    variance: float
    opinion_max_change: float = 0.0


@dataclass
class Simulation:
    """Simulation state and history."""

    agents: list
    network: nx.Graph
    base_confidence_bound: float = 0.2
    timestep: int = 0
    cluster_history: list = field(default_factory=list)


# ============================================================
# INITIALIZATION
# ============================================================


def load_opinion_distribution(data_path: Path) -> list:
    """Load opinion scores from political bias dataset."""
    with open(data_path, "r") as f:
        data = json.load(f)

    opinions = []
    for example in data["examples"]:
        score = example["context"]["opinion_score"]
        opinions.append(score)

    return opinions


def initialize_agents(
    n_agents: int,
    alpha_value: float,
    opinion_distribution: list,
    seed: int,
    noise_std: float = 0.15
) -> list:
    """
    Initialize agents with sampled opinions and fixed asymmetry.

    Adds Gaussian noise to sampled opinions to create more continuous
    distribution and more realistic opinion dynamics.
    """
    rng = np.random.default_rng(seed)

    agents = []
    for _ in range(n_agents):
        # Sample from real distribution with replacement
        base_opinion = rng.choice(opinion_distribution)
        # Add noise to create continuous distribution
        noise = rng.normal(0, noise_std)
        opinion = np.clip(base_opinion + noise, -1.0, 1.0)
        agent = Agent(
            opinion=opinion,
            alpha=alpha_value
        )
        agents.append(agent)

    return agents


def create_network(
    n_agents: int,
    network_type: str,
    seed: int
) -> nx.Graph:
    """Create network topology for agent interactions."""
    if network_type == "small_world":
        # Watts-Strogatz: k=6 neighbors, p=0.15 rewiring (per validation protocol)
        network = nx.watts_strogatz_graph(n=n_agents, k=6, p=0.15, seed=seed)
    elif network_type == "random":
        # Erdos-Renyi: p=0.01 edge probability
        network = nx.erdos_renyi_graph(n=n_agents, p=0.01, seed=seed)
        # Ensure connected
        if not nx.is_connected(network):
            # Add edges to connect components
            components = list(nx.connected_components(network))
            for i in range(len(components) - 1):
                node1 = list(components[i])[0]
                node2 = list(components[i + 1])[0]
                network.add_edge(node1, node2)
    else:
        raise ValueError(f"Unknown network type: {network_type}")

    return network


def initialize_simulation(
    n_agents: int,
    alpha_value: float,
    network_type: str,
    seed: int,
    opinion_distribution: list,
    base_epsilon: float = 0.2
) -> Simulation:
    """Initialize complete simulation state."""
    agents = initialize_agents(n_agents, alpha_value, opinion_distribution, seed)
    network = create_network(n_agents, network_type, seed)

    return Simulation(
        agents=agents,
        network=network,
        base_confidence_bound=base_epsilon,
        timestep=0,
        cluster_history=[]
    )


# ============================================================
# CLUSTER DETECTION (using finding 2 protocol)
# ============================================================


def detect_clusters(agents: list, min_cluster_size: int = 3, cluster_threshold: float = 0.05) -> tuple:
    """
    Detect opinion clusters using distance-based connectivity.

    Uses gap-based clustering: agents are in same cluster if no gap > threshold
    between consecutive sorted opinions.
    Returns cluster centers and smooth clustering metric c = 1/Σ(r_i²).
    """
    n = len(agents)
    opinions = np.array([a.opinion for a in agents])

    # Sort indices by opinion for efficient clustering
    sorted_indices = np.argsort(opinions)
    sorted_opinions = opinions[sorted_indices]

    # Distance-based clustering: agents within threshold are in same cluster
    cluster_labels = np.zeros(n, dtype=int)
    current_cluster = 0

    for i in range(n):
        if i == 0:
            cluster_labels[sorted_indices[i]] = current_cluster
        else:
            # Check gap to previous agent - larger gaps create new clusters
            gap = sorted_opinions[i] - sorted_opinions[i - 1]
            if gap > cluster_threshold:
                current_cluster += 1
            cluster_labels[sorted_indices[i]] = current_cluster

    # Update agent cluster assignments
    for i, agent in enumerate(agents):
        agent.ingroup_id = cluster_labels[i]

    # Compute cluster centers (only for clusters meeting min size)
    cluster_ids = set(cluster_labels)
    cluster_centers = []
    for cid in sorted(cluster_ids):
        mask = cluster_labels == cid
        if np.sum(mask) >= min_cluster_size:
            center = np.mean(opinions[mask])
            cluster_centers.append(center)

    # Count actual meaningful clusters
    meaningful_clusters = len(cluster_centers)

    # Smooth clustering metric: c = 1 / Σ(r_i²)
    # r_i = fraction of agents in cluster i
    cluster_sizes = Counter(cluster_labels)
    r_i_squared = sum((count / n) ** 2 for count in cluster_sizes.values())
    smooth_metric = 1.0 / r_i_squared if r_i_squared > 0 else 1.0

    return cluster_centers, smooth_metric, meaningful_clusters


# ============================================================
# OPINION UPDATE RULE (core hypothesis mechanism)
# ============================================================


def compute_effective_confidence_bound(
    agent: Agent,
    neighbor: Agent,
    base_epsilon: float
) -> float:
    """
    Modulate confidence bound based on perception asymmetry.

    Same group: use full confidence bound.
    Different group: discount by factor (1 - alpha).
    Higher alpha = more asymmetry = stronger discount = less outgroup influence.
    """
    if agent.ingroup_id == neighbor.ingroup_id:
        # Same group: full confidence bound
        return base_epsilon
    else:
        # Outgroup: discount based on asymmetry
        # Floor at 0.05 to avoid complete isolation
        discount_factor = max(0.05, 1.0 - agent.alpha)
        return base_epsilon * discount_factor


def update_opinions(simulation: Simulation) -> float:
    """
    Modified bounded-confidence update with asymmetry modulation.

    Returns maximum opinion change for convergence checking.
    """
    agents = simulation.agents
    network = simulation.network
    base_epsilon = simulation.base_confidence_bound

    new_opinions = []

    for idx, agent in enumerate(agents):
        # Get neighbors from network
        neighbor_indices = list(network.neighbors(idx))

        # Collect opinions within effective confidence bound
        influences = []
        for neighbor_idx in neighbor_indices:
            neighbor = agents[neighbor_idx]

            # Compute effective bound based on asymmetry
            eff_bound = compute_effective_confidence_bound(
                agent, neighbor, base_epsilon
            )

            # Check if within bound
            if abs(agent.opinion - neighbor.opinion) <= eff_bound:
                influences.append(neighbor.opinion)

        # Update opinion: weighted average with self (mu=0.5 standard)
        if influences:
            new_opinion = 0.5 * agent.opinion + 0.5 * np.mean(influences)
        else:
            new_opinion = agent.opinion

        new_opinions.append(new_opinion)

    # Calculate max change before applying updates
    max_change = max(
        abs(new_opinions[i] - agents[i].opinion)
        for i in range(len(agents))
    )

    # Apply updates
    for agent, new_opinion in zip(agents, new_opinions):
        agent.opinion = np.clip(new_opinion, -1.0, 1.0)

    return max_change


# ============================================================
# METRICS COMPUTATION
# ============================================================


def compute_modularity(network: nx.Graph, cluster_labels: list) -> float:
    """Compute Newman-Girvan modularity for current clustering."""
    # Group nodes by cluster
    communities = {}
    for idx, label in enumerate(cluster_labels):
        if label not in communities:
            communities[label] = []
        communities[label].append(idx)

    community_list = [frozenset(nodes) for nodes in communities.values()]

    try:
        Q = nx.community.modularity(network, community_list)
    except ZeroDivisionError:
        Q = 0.0

    return Q


def compute_cluster_metrics(simulation: Simulation) -> ClusterMetrics:
    """Compute all metrics for current timestep."""
    agents = simulation.agents
    network = simulation.network

    # Detect clusters and get metrics
    cluster_centers, smooth_metric, cluster_count = detect_clusters(agents)

    # Get cluster labels for modularity
    cluster_labels = [agent.ingroup_id for agent in agents]
    modularity_Q = compute_modularity(network, cluster_labels)

    # Opinion variance
    opinions = [agent.opinion for agent in agents]
    variance = np.var(opinions)

    return ClusterMetrics(
        timestep=simulation.timestep,
        cluster_count=cluster_count,
        smooth_metric=smooth_metric,
        modularity_Q=modularity_Q,
        variance=variance
    )


# ============================================================
# CONVERGENCE DETECTION
# ============================================================


def check_convergence(
    history: list,
    window: int = 50,
    threshold: float = 0.0005
) -> bool:
    """
    Check if max opinion change < threshold for window consecutive steps.

    Uses validation protocol criterion: |m| <= 0.001 for 100 steps.
    Made more sensitive (0.0005) to allow more dynamics to play out.
    """
    if len(history) < window:
        return False

    recent_changes = [m.opinion_max_change for m in history[-window:]]

    # All changes must be below threshold
    return all(c < threshold for c in recent_changes)


# ============================================================
# CLUSTER LIFETIME COMPUTATION
# ============================================================


def compute_cluster_lifetime(history: list) -> float:
    """
    Compute mean cluster lifetime (persistence).

    Tracks how long cluster configurations persist before changes.
    """
    if len(history) < 2:
        return 0.0

    cluster_counts = [m.cluster_count for m in history]

    # Count stable periods
    lifetimes = []
    current_lifetime = 1

    for i in range(1, len(cluster_counts)):
        if cluster_counts[i] == cluster_counts[i - 1]:
            current_lifetime += 1
        else:
            lifetimes.append(current_lifetime)
            current_lifetime = 1
    lifetimes.append(current_lifetime)

    return np.mean(lifetimes) if lifetimes else 0.0


# ============================================================
# EARLY WARNING SIGNALS (for phase transition)
# ============================================================


def compute_autocorrelation(series: list, lag: int = 10) -> float:
    """Compute lag-k autocorrelation."""
    n = len(series)
    if n <= lag:
        return 0.0

    series = np.array(series)
    mean = np.mean(series)
    c0 = np.sum((series - mean) ** 2) / n

    if c0 == 0:
        return 0.0

    ck = np.sum((series[:-lag] - mean) * (series[lag:] - mean)) / n
    return ck / c0


def detect_changepoint(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """
    Detect change point using piecewise regression.

    Returns alpha value where qualitative change occurs.
    """
    from sklearn.linear_model import LinearRegression

    if len(x) < 5:
        return None

    x = np.array(x)
    y = np.array(y)

    best_breakpoint = None
    best_error = float('inf')

    for i in range(2, len(x) - 2):
        # Fit two linear models
        model1 = LinearRegression().fit(x[:i].reshape(-1, 1), y[:i])
        model2 = LinearRegression().fit(x[i:].reshape(-1, 1), y[i:])

        # Compute total error
        error1 = np.sum((y[:i] - model1.predict(x[:i].reshape(-1, 1))) ** 2)
        error2 = np.sum((y[i:] - model2.predict(x[i:].reshape(-1, 1))) ** 2)
        total_error = error1 + error2

        if total_error < best_error:
            best_error = total_error
            best_breakpoint = x[i]

    return best_breakpoint


# ============================================================
# RUN SINGLE SIMULATION
# ============================================================


def run_simulation(
    n_agents: int,
    alpha_value: float,
    network_type: str,
    seed: int,
    opinion_distribution: list,
    max_timesteps: int = 1500,
    base_epsilon: float = 0.2
) -> Simulation:
    """Run a single simulation to convergence or max timesteps."""
    sim = initialize_simulation(
        n_agents=n_agents,
        alpha_value=alpha_value,
        network_type=network_type,
        seed=seed,
        opinion_distribution=opinion_distribution,
        base_epsilon=base_epsilon
    )

    for t in range(max_timesteps):
        sim.timestep = t

        # Detect clusters and update ingroup assignments
        detect_clusters(sim.agents)

        # Compute metrics before update
        metrics = compute_cluster_metrics(sim)

        # Update opinions
        max_change = update_opinions(sim)
        metrics.opinion_max_change = max_change

        sim.cluster_history.append(metrics)

        # Check convergence
        if check_convergence(sim.cluster_history):
            break

    return sim


# ============================================================
# EXPERIMENT 1: Stability vs Asymmetry (Prediction 1)
# ============================================================


def experiment_1_stability_correlation(
    opinion_distribution: list,
    output_dir: Path,
    alpha_values: np.ndarray = None,
    n_seeds: int = 20,
    n_agents: int = 300,
    max_timesteps: int = 1000,
    network_type: str = "small_world"
) -> tuple:
    """
    Test: bubble stability correlates with asymmetry magnitude alpha.
    """
    if alpha_values is None:
        alpha_values = np.arange(0.0, 1.55, 0.1)

    results = []
    total_runs = len(alpha_values) * n_seeds
    run_count = 0

    print(f"Experiment 1: {len(alpha_values)} alpha values x {n_seeds} seeds = {total_runs} runs")

    for alpha in alpha_values:
        for seed in range(n_seeds):
            run_count += 1
            if run_count % 20 == 0:
                print(f"  Progress: {run_count}/{total_runs}")

            sim = run_simulation(
                n_agents=n_agents,
                alpha_value=alpha,
                network_type=network_type,
                seed=seed,
                opinion_distribution=opinion_distribution,
                max_timesteps=max_timesteps
            )

            # Compute stability metrics
            cluster_lifetime_mean = compute_cluster_lifetime(sim.cluster_history)
            final_metrics = sim.cluster_history[-1]

            results.append({
                "alpha": alpha,
                "seed": seed,
                "cluster_count": final_metrics.cluster_count,
                "cluster_lifetime_mean": cluster_lifetime_mean,
                "modularity_Q": final_metrics.modularity_Q,
                "final_variance": final_metrics.variance,
                "convergence_time": len(sim.cluster_history)
            })

    # Save results
    df = pd.DataFrame(results)
    csv_path = output_dir / "stability_vs_asymmetry.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Statistical analysis: Pearson correlation
    r, p_value = stats.pearsonr(df["alpha"], df["cluster_lifetime_mean"])
    r_count, p_count = stats.pearsonr(df["alpha"], df["cluster_count"])

    statistics = {
        "pearson_r_lifetime": float(r),
        "p_value_lifetime": float(p_value),
        "pearson_r_cluster_count": float(r_count),
        "p_value_cluster_count": float(p_count),
        "n_samples": len(df)
    }

    print(f"  Correlation (alpha vs lifetime): r={r:.3f}, p={p_value:.4f}")
    print(f"  Correlation (alpha vs cluster_count): r={r_count:.3f}, p={p_count:.4f}")

    return df, statistics


# ============================================================
# EXPERIMENT 2: Asymmetry Reduction Intervention (Prediction 2)
# ============================================================


def experiment_2_asymmetry_reduction(
    opinion_distribution: list,
    output_dir: Path,
    n_seeds: int = 10,
    n_agents: int = 300,
    intervention_timestep: int = 400,
    total_timesteps: int = 1200,
    network_type: str = "small_world"
) -> tuple:
    """
    Test: reducing asymmetry specifically enables bubble merging.
    """
    results = []

    print(f"Experiment 2: 2 conditions x {n_seeds} seeds")

    for seed in range(n_seeds):
        for condition in ["treatment", "control"]:
            # Initialize with high asymmetry
            sim = initialize_simulation(
                n_agents=n_agents,
                alpha_value=0.8,
                network_type=network_type,
                seed=seed,
                opinion_distribution=opinion_distribution
            )

            prev_cluster_count = None

            for t in range(total_timesteps):
                sim.timestep = t

                # Intervention at specified timestep
                if t == intervention_timestep and condition == "treatment":
                    for agent in sim.agents:
                        agent.alpha = 0.2

                detect_clusters(sim.agents)
                metrics = compute_cluster_metrics(sim)
                max_change = update_opinions(sim)
                metrics.opinion_max_change = max_change

                # Track merger events
                merger_event = 0
                if prev_cluster_count is not None:
                    if metrics.cluster_count < prev_cluster_count:
                        merger_event = 1

                results.append({
                    "timestep": t,
                    "condition": condition,
                    "seed": seed,
                    "cluster_count": metrics.cluster_count,
                    "modularity_Q": metrics.modularity_Q,
                    "variance": metrics.variance,
                    "merger_event": merger_event
                })

                prev_cluster_count = metrics.cluster_count

        print(f"  Completed seed {seed + 1}/{n_seeds}")

    # Save results
    df = pd.DataFrame(results)
    csv_path = output_dir / "merger_events.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Statistical analysis
    # Compare merger rates post-intervention
    post_intervention = df[df["timestep"] >= intervention_timestep]
    treatment_mergers = post_intervention[post_intervention["condition"] == "treatment"]["merger_event"].sum()
    control_mergers = post_intervention[post_intervention["condition"] == "control"]["merger_event"].sum()

    # T-test on final cluster counts
    final_ts = total_timesteps - 1
    treatment_final = df[(df["condition"] == "treatment") & (df["timestep"] == final_ts)]["cluster_count"]
    control_final = df[(df["condition"] == "control") & (df["timestep"] == final_ts)]["cluster_count"]

    t_stat, p_value = stats.ttest_ind(treatment_final, control_final)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((treatment_final.var() + control_final.var()) / 2)
    cohens_d = (control_final.mean() - treatment_final.mean()) / pooled_std if pooled_std > 0 else 0

    statistics = {
        "treatment_mergers": int(treatment_mergers),
        "control_mergers": int(control_mergers),
        "treatment_final_clusters_mean": float(treatment_final.mean()),
        "control_final_clusters_mean": float(control_final.mean()),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d)
    }

    print(f"  Treatment mergers: {treatment_mergers}, Control mergers: {control_mergers}")
    print(f"  T-test: t={t_stat:.3f}, p={p_value:.4f}, Cohen's d={cohens_d:.3f}")

    return df, statistics


# ============================================================
# EXPERIMENT 3: Critical Threshold Detection (Prediction 3)
# ============================================================


def experiment_3_critical_threshold(
    opinion_distribution: list,
    output_dir: Path,
    alpha_values: np.ndarray = None,
    n_seeds: int = 15,
    n_agents: int = 500,
    max_timesteps: int = 1000,
    network_type: str = "random"
) -> tuple:
    """
    Test: identify critical alpha_c where phase transition occurs.
    """
    if alpha_values is None:
        alpha_values = np.arange(0.3, 0.95, 0.05)

    results = []
    total_runs = len(alpha_values) * n_seeds
    run_count = 0

    print(f"Experiment 3: {len(alpha_values)} alpha values x {n_seeds} seeds = {total_runs} runs")

    for alpha in alpha_values:
        for seed in range(n_seeds):
            run_count += 1
            if run_count % 20 == 0:
                print(f"  Progress: {run_count}/{total_runs}")

            sim = run_simulation(
                n_agents=n_agents,
                alpha_value=alpha,
                network_type=network_type,
                seed=seed,
                opinion_distribution=opinion_distribution,
                max_timesteps=max_timesteps
            )

            # Compute early warning signals
            cluster_counts = [m.cluster_count for m in sim.cluster_history]
            variances = [m.variance for m in sim.cluster_history]

            # Variance of cluster count (increases near transition)
            variance_of_counts = np.var(cluster_counts) if cluster_counts else 0

            # Autocorrelation (increases near transition)
            autocorr = compute_autocorrelation(cluster_counts, lag=10)

            results.append({
                "alpha": alpha,
                "seed": seed,
                "final_cluster_count": cluster_counts[-1] if cluster_counts else 0,
                "variance_of_counts": variance_of_counts,
                "autocorrelation": autocorr,
                "final_variance": variances[-1] if variances else 0,
                "convergence_time": len(sim.cluster_history)
            })

    # Save results
    df = pd.DataFrame(results)
    csv_path = output_dir / "phase_transition_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Detect change point
    grouped = df.groupby("alpha")["final_cluster_count"].mean()
    alpha_c = detect_changepoint(grouped.index.values, grouped.values)

    statistics = {
        "critical_alpha": float(alpha_c) if alpha_c is not None else None,
        "alpha_range": [float(alpha_values.min()), float(alpha_values.max())],
        "n_samples": len(df)
    }

    print(f"  Critical threshold alpha_c ~ {alpha_c:.2f}" if alpha_c else "  No clear threshold detected")

    return df, statistics


# ============================================================
# VISUALIZATION
# ============================================================


def create_visualizations(output_dir: Path):
    """Generate all visualization plots."""
    import seaborn as sns

    # Load results
    df_exp1 = pd.read_csv(output_dir / "stability_vs_asymmetry.csv")
    df_exp2 = pd.read_csv(output_dir / "merger_events.csv")
    df_exp3 = pd.read_csv(output_dir / "phase_transition_analysis.csv")

    # Plot 1: Alpha vs Stability (scatter with regression)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Lifetime vs alpha
    ax1 = axes[0]
    sns.regplot(data=df_exp1, x="alpha", y="cluster_lifetime_mean",
                scatter_kws={"alpha": 0.3, "s": 20}, ax=ax1)
    r, p = stats.pearsonr(df_exp1["alpha"], df_exp1["cluster_lifetime_mean"])
    ax1.set_xlabel("Perception Asymmetry (alpha)")
    ax1.set_ylabel("Mean Cluster Lifetime")
    ax1.set_title(f"Prediction 1: Stability vs Asymmetry\nr={r:.3f}, p={p:.4f}")

    # Cluster count vs alpha
    ax2 = axes[1]
    sns.regplot(data=df_exp1, x="alpha", y="cluster_count",
                scatter_kws={"alpha": 0.3, "s": 20}, ax=ax2)
    r, p = stats.pearsonr(df_exp1["alpha"], df_exp1["cluster_count"])
    ax2.set_xlabel("Perception Asymmetry (alpha)")
    ax2.set_ylabel("Final Cluster Count")
    ax2.set_title(f"Cluster Count vs Asymmetry\nr={r:.3f}, p={p:.4f}")

    plt.tight_layout()
    plt.savefig(output_dir / "plot_stability_vs_alpha.png", dpi=150)
    plt.close()

    # Plot 2: Merger Intervention Trajectories
    fig, ax = plt.subplots(figsize=(10, 6))

    for condition in ["treatment", "control"]:
        subset = df_exp2[df_exp2["condition"] == condition]
        grouped = subset.groupby("timestep")["cluster_count"].agg(["mean", "std"])

        color = "blue" if condition == "treatment" else "orange"
        label = f"{condition.capitalize()}"

        ax.plot(grouped.index, grouped["mean"], label=label, linewidth=2, color=color)
        ax.fill_between(
            grouped.index,
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.2,
            color=color
        )

    # Mark intervention point
    intervention_t = 400
    ax.axvline(x=intervention_t, color="red", linestyle="--", linewidth=2, label="Intervention")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean Cluster Count")
    ax.set_title("Prediction 2: Asymmetry Reduction Intervention\n(alpha: 0.8 -> 0.2 for treatment)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "plot_merger_intervention.png", dpi=150)
    plt.close()

    # Plot 3: Bifurcation Diagram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cluster count vs alpha
    ax1 = axes[0]
    grouped = df_exp3.groupby("alpha")["final_cluster_count"].agg(["mean", "std"])
    ax1.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"],
                 fmt="o-", capsize=3, linewidth=2, markersize=6)
    ax1.set_xlabel("Perception Asymmetry (alpha)")
    ax1.set_ylabel("Final Cluster Count")
    ax1.set_title("Prediction 3: Phase Transition Detection")

    # Early warning signals
    ax2 = axes[1]
    grouped_var = df_exp3.groupby("alpha")["variance_of_counts"].mean()
    grouped_ac = df_exp3.groupby("alpha")["autocorrelation"].mean()

    ax2_twin = ax2.twinx()
    l1 = ax2.plot(grouped_var.index, grouped_var.values, "b-o",
                  label="Variance of counts", linewidth=2, markersize=5)
    l2 = ax2_twin.plot(grouped_ac.index, grouped_ac.values, "r-s",
                       label="Autocorrelation", linewidth=2, markersize=5)

    ax2.set_xlabel("Perception Asymmetry (alpha)")
    ax2.set_ylabel("Variance of Cluster Counts", color="blue")
    ax2_twin.set_ylabel("Autocorrelation (lag=10)", color="red")
    ax2.set_title("Early Warning Signals")

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_dir / "plot_phase_transition.png", dpi=150)
    plt.close()

    print(f"Visualizations saved to {output_dir}")


# ============================================================
# MAIN EXECUTION
# ============================================================


def main():
    """Run all three experiments."""
    from loguru import logger
    import sys

    # Setup logging
    BLUE, GREEN, YELLOW, CYAN, END = "\033[94m", "\033[92m", "\033[93m", "\033[96m", "\033[0m"

    logger.remove()
    logger.add(
        sys.stdout,
        format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level: <7}}|{CYAN}{{name: >12.12}}{END}.{CYAN}{{function: <22.22}}{END}:{CYAN}{{line: <4}}{END}| {{message}}",
        level="INFO",
        colorize=False
    )

    # Paths
    workspace = Path(__file__).parent.parent
    output_dir = workspace / "output"
    output_dir.mkdir(exist_ok=True)

    # Log file
    log_path = workspace / "logs" / "experiment.log"
    log_path.parent.mkdir(exist_ok=True)
    logger.add(str(log_path), rotation="30 MB")

    # Load opinion distribution from dependency
    data_path = workspace / "dependencies" / "Political_Bias_Opinion_Distribution_Dataset_for_Ag" / "full_data_out.json"

    logger.info(f"{BLUE}Loading opinion distribution from: {data_path}{END}")
    opinion_distribution = load_opinion_distribution(data_path)
    logger.info(f"Loaded {len(opinion_distribution)} opinion scores")

    # Distribution statistics
    logger.info(f"Opinion distribution: mean={np.mean(opinion_distribution):.3f}, "
                f"std={np.std(opinion_distribution):.3f}, "
                f"min={np.min(opinion_distribution):.3f}, "
                f"max={np.max(opinion_distribution):.3f}")

    print("\n" + "=" * 60)
    print("ASYMMETRIC BOUNDED-CONFIDENCE EXPERIMENTS")
    print("=" * 60)

    # Run experiments
    all_stats = {}

    print(f"\n{BLUE}=== Experiment 1: Stability vs Asymmetry ==={END}")
    df1, stats1 = experiment_1_stability_correlation(
        opinion_distribution=opinion_distribution,
        output_dir=output_dir,
        n_seeds=12,  # Reduced for runtime
        n_agents=250,
        max_timesteps=600
    )
    all_stats["experiment_1"] = stats1

    print(f"\n{BLUE}=== Experiment 2: Asymmetry Reduction Intervention ==={END}")
    df2, stats2 = experiment_2_asymmetry_reduction(
        opinion_distribution=opinion_distribution,
        output_dir=output_dir,
        n_seeds=6,  # Reduced for runtime
        n_agents=250,
        intervention_timestep=300,
        total_timesteps=700
    )
    all_stats["experiment_2"] = stats2

    print(f"\n{BLUE}=== Experiment 3: Critical Threshold Detection ==={END}")
    df3, stats3 = experiment_3_critical_threshold(
        opinion_distribution=opinion_distribution,
        output_dir=output_dir,
        n_seeds=10,  # Reduced for runtime
        n_agents=350,
        max_timesteps=600
    )
    all_stats["experiment_3"] = stats3

    # Create visualizations
    print(f"\n{BLUE}Generating visualizations...{END}")
    create_visualizations(output_dir)

    # Save summary
    summary_path = output_dir / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n{GREEN}Experiments complete!{END}")
    print(f"Results saved to: {output_dir}")
    print(f"\nSummary:")
    print(json.dumps(all_stats, indent=2))

    return all_stats


if __name__ == "__main__":
    main()
