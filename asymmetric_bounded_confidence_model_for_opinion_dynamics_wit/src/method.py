"""
method.py - Asymmetric Bounded-Confidence Agent-Based Model

This module implements:
1. Our METHOD: Asymmetric Bounded-Confidence Model - modulates confidence bounds
   based on perception asymmetry (alpha) to test opinion bubble stability.
2. BASELINE: Standard Bounded-Confidence Model (alpha=0, no asymmetry)

Output format follows exp_gen_sol_out.json schema.
"""

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# LOGGING SETUP
# ============================================================

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("method.log", mode="w")
    ]
)
logger = logging.getLogger("method")

# Truncate long strings in logs
def log_truncated(msg: str, max_len: int = 500) -> str:
    """Truncate message if too long for logging."""
    if len(msg) > max_len:
        return msg[:max_len] + f"... [TRUNCATED, total {len(msg)} chars]"
    return msg


def validate_output_schema(output: dict) -> list:
    """
    Validate output against exp_gen_sol_out.json schema.
    Returns list of error messages (empty if valid).
    """
    errors = []

    # Check root is object with examples
    if not isinstance(output, dict):
        errors.append('Root must be an object')
        return errors

    if 'examples' not in output:
        errors.append('Missing required "examples" field')
        return errors

    if not isinstance(output['examples'], list):
        errors.append('"examples" must be an array')
        return errors

    if len(output['examples']) < 1:
        errors.append('"examples" must have at least 1 item')
        return errors

    # Check each example
    required_fields = ['input', 'output', 'context', 'dataset', 'split', 'predict_baseline', 'predict_method', 'method']
    valid_splits = ['train', 'val', 'test', 'validation']

    for i, example in enumerate(output['examples']):
        if not isinstance(example, dict):
            errors.append(f'Example {i} is not an object')
            continue

        for field in required_fields:
            if field not in example:
                errors.append(f'Example {i} missing required field "{field}"')

        # Check types
        if 'input' in example and not isinstance(example['input'], str):
            errors.append(f'Example {i} "input" must be string')
        if 'output' in example and not isinstance(example['output'], str):
            errors.append(f'Example {i} "output" must be string')
        if 'context' in example and not isinstance(example['context'], dict):
            errors.append(f'Example {i} "context" must be object')
        if 'dataset' in example and not isinstance(example['dataset'], str):
            errors.append(f'Example {i} "dataset" must be string')
        if 'split' in example:
            if not isinstance(example['split'], str):
                errors.append(f'Example {i} "split" must be string')
            elif example['split'] not in valid_splits:
                errors.append(f'Example {i} "split" must be one of {valid_splits}')
        if 'predict_baseline' in example and not isinstance(example['predict_baseline'], str):
            errors.append(f'Example {i} "predict_baseline" must be string')
        if 'predict_method' in example and not isinstance(example['predict_method'], str):
            errors.append(f'Example {i} "predict_method" must be string')
        if 'method' in example and not isinstance(example['method'], str):
            errors.append(f'Example {i} "method" must be string')

    return errors


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Agent:
    """Agent with opinion and perception asymmetry parameters."""
    opinion: float  # continuous value in [-1, +1]
    ingroup_id: int = -1  # cluster assignment
    alpha: float = 0.0  # perception asymmetry (0 = baseline, >0 = our method)

    def __post_init__(self):
        self.opinion = float(np.clip(self.opinion, -1.0, 1.0))


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
class SimulationResult:
    """Results from a simulation run."""
    final_cluster_count: int
    cluster_lifetime_mean: float
    final_variance: float
    final_modularity_Q: float
    convergence_time: int
    alpha: float
    seed: int
    network_type: str
    n_agents: int


# ============================================================
# DATA LOADING
# ============================================================

def load_opinion_distribution(data_path: Path) -> list:
    """Load opinion scores from political bias dataset."""
    logger.info(f"Loading opinion distribution from: {data_path}")

    # Sanity check: verify path exists
    if not data_path.exists():
        logger.error(f"Data file does not exist: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if not data_path.is_file():
        logger.error(f"Path is not a file: {data_path}")
        raise ValueError(f"Path is not a file: {data_path}")

    try:
        with open(data_path, "r") as f:
            data = json.load(f)

        # Validate data structure
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict at root level, got {type(data).__name__}")

        if "examples" not in data:
            raise KeyError("Missing 'examples' key in data")

        if not isinstance(data["examples"], list):
            raise ValueError(f"Expected list for 'examples', got {type(data['examples']).__name__}")

        if len(data["examples"]) == 0:
            raise ValueError("Empty examples list in data")

        opinions = []
        for idx, example in enumerate(data["examples"]):
            # Validate each example
            if not isinstance(example, dict):
                logger.warning(f"Example {idx} is not a dict, skipping")
                continue

            if "context" not in example:
                logger.warning(f"Example {idx} missing 'context', skipping")
                continue

            if "opinion_score" not in example["context"]:
                logger.warning(f"Example {idx} missing 'opinion_score', skipping")
                continue

            score = example["context"]["opinion_score"]

            # Validate score is numeric
            try:
                score_float = float(score)
            except (TypeError, ValueError) as e:
                logger.warning(f"Example {idx} has invalid opinion_score '{score}', skipping")
                continue

            # Validate score is in valid range
            if not (-1.0 <= score_float <= 1.0):
                logger.warning(f"Example {idx} has out-of-range score {score_float}, clipping to [-1, 1]")
                score_float = float(np.clip(score_float, -1.0, 1.0))

            opinions.append(score_float)

        if len(opinions) == 0:
            raise ValueError("No valid opinion scores found in data")

        logger.info(f"Loaded {len(opinions)} opinion scores from {len(data['examples'])} examples")
        logger.debug(f"Opinion range: [{min(opinions):.2f}, {max(opinions):.2f}]")
        logger.debug(f"Opinion mean: {np.mean(opinions):.3f}, std: {np.std(opinions):.3f}")

        # Sanity check: warn if low diversity
        if np.std(opinions) < 0.01:
            logger.warning(f"Very low opinion diversity (std={np.std(opinions):.4f}). Results may not be meaningful.")

        return opinions

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {data_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in data file: {e}")
        raise
    except KeyError as e:
        logger.error(f"Missing key in data structure: {e}")
        raise
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        raise


# ============================================================
# INITIALIZATION
# ============================================================

def initialize_agents(
    n_agents: int,
    alpha_value: float,
    opinion_distribution: list,
    seed: int,
    noise_std: float = 0.15
) -> list:
    """
    Initialize agents with sampled opinions and fixed asymmetry.

    Args:
        n_agents: Number of agents
        alpha_value: Perception asymmetry (0 = baseline, >0 = our method)
        opinion_distribution: List of opinion scores to sample from
        seed: Random seed
        noise_std: Standard deviation of Gaussian noise added to opinions
    """
    # Input validation
    if not isinstance(n_agents, int) or n_agents <= 0:
        raise ValueError(f"n_agents must be a positive integer, got {n_agents}")

    if not isinstance(alpha_value, (int, float)):
        raise ValueError(f"alpha_value must be numeric, got {type(alpha_value).__name__}")

    if alpha_value < 0:
        raise ValueError(f"alpha_value must be non-negative, got {alpha_value}")

    if not opinion_distribution or len(opinion_distribution) == 0:
        raise ValueError("opinion_distribution cannot be empty")

    if not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {type(seed).__name__}")

    if noise_std < 0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")

    logger.debug(f"Initializing {n_agents} agents with alpha={alpha_value}, seed={seed}")

    try:
        rng = np.random.default_rng(seed)

        agents = []
        for i in range(n_agents):
            # Sample from real distribution with replacement
            base_opinion = rng.choice(opinion_distribution)
            # Add noise to create continuous distribution
            noise = rng.normal(0, noise_std)
            opinion = float(np.clip(base_opinion + noise, -1.0, 1.0))

            agent = Agent(
                opinion=opinion,
                alpha=alpha_value
            )
            agents.append(agent)

        opinions = [a.opinion for a in agents]
        logger.debug(f"Agent opinions: mean={np.mean(opinions):.3f}, std={np.std(opinions):.3f}")

        # Sanity check: ensure all agents were created
        if len(agents) != n_agents:
            raise RuntimeError(f"Expected {n_agents} agents, but created {len(agents)}")

        return agents

    except Exception as e:
        logger.error(f"Error initializing agents: {e}")
        raise


def create_network(
    n_agents: int,
    network_type: str,
    seed: int
) -> nx.Graph:
    """Create network topology for agent interactions."""
    # Input validation
    if not isinstance(n_agents, int) or n_agents <= 0:
        raise ValueError(f"n_agents must be a positive integer, got {n_agents}")

    if not isinstance(network_type, str):
        raise ValueError(f"network_type must be a string, got {type(network_type).__name__}")

    valid_types = ["small_world", "random"]
    if network_type not in valid_types:
        raise ValueError(f"network_type must be one of {valid_types}, got '{network_type}'")

    if not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {type(seed).__name__}")

    # Minimum agents for small-world (need at least k+1 = 7)
    if network_type == "small_world" and n_agents < 7:
        raise ValueError(f"small_world network requires at least 7 agents, got {n_agents}")

    logger.debug(f"Creating {network_type} network with {n_agents} nodes, seed={seed}")

    try:
        if network_type == "small_world":
            # Watts-Strogatz: k=6 neighbors, p=0.15 rewiring
            network = nx.watts_strogatz_graph(n=n_agents, k=6, p=0.15, seed=seed)
        elif network_type == "random":
            # Erdos-Renyi: p=0.01 edge probability
            network = nx.erdos_renyi_graph(n=n_agents, p=0.015, seed=seed)
            # Ensure connected
            if not nx.is_connected(network):
                components = list(nx.connected_components(network))
                logger.debug(f"Network has {len(components)} components, connecting...")
                for i in range(len(components) - 1):
                    node1 = list(components[i])[0]
                    node2 = list(components[i + 1])[0]
                    network.add_edge(node1, node2)
        else:
            raise ValueError(f"Unknown network type: {network_type}")

        # Sanity checks
        if network.number_of_nodes() != n_agents:
            raise RuntimeError(f"Network has {network.number_of_nodes()} nodes, expected {n_agents}")

        if network.number_of_edges() == 0:
            raise RuntimeError("Network has no edges - agents cannot interact")

        avg_degree = sum(d for n, d in network.degree()) / n_agents
        logger.debug(f"Network created: {network.number_of_edges()} edges, avg_degree={avg_degree:.2f}")

        # Warning for very sparse networks
        if avg_degree < 2:
            logger.warning(f"Network is very sparse (avg_degree={avg_degree:.2f}). Dynamics may be slow.")

        return network

    except Exception as e:
        logger.error(f"Error creating network: {e}")
        raise


# ============================================================
# CLUSTER DETECTION
# ============================================================

def detect_clusters(agents: list, min_cluster_size: int = 3, cluster_threshold: float = 0.05) -> tuple:
    """
    Detect opinion clusters using distance-based connectivity.

    Uses gap-based clustering: agents are in same cluster if no gap > threshold
    between consecutive sorted opinions.

    Returns: (cluster_centers, smooth_metric, meaningful_cluster_count)
    """
    n = len(agents)
    opinions = np.array([a.opinion for a in agents])

    # Sort indices by opinion
    sorted_indices = np.argsort(opinions)
    sorted_opinions = opinions[sorted_indices]

    # Distance-based clustering
    cluster_labels = np.zeros(n, dtype=int)
    current_cluster = 0

    for i in range(n):
        if i == 0:
            cluster_labels[sorted_indices[i]] = current_cluster
        else:
            gap = sorted_opinions[i] - sorted_opinions[i - 1]
            if gap > cluster_threshold:
                current_cluster += 1
            cluster_labels[sorted_indices[i]] = current_cluster

    # Update agent cluster assignments
    for i, agent in enumerate(agents):
        agent.ingroup_id = int(cluster_labels[i])

    # Compute cluster centers (only for clusters meeting min size)
    cluster_ids = set(cluster_labels)
    cluster_centers = []
    for cid in sorted(cluster_ids):
        mask = cluster_labels == cid
        if np.sum(mask) >= min_cluster_size:
            center = float(np.mean(opinions[mask]))
            cluster_centers.append(center)

    meaningful_clusters = len(cluster_centers)

    # Smooth clustering metric: c = 1 / Σ(r_i²)
    cluster_sizes = Counter(cluster_labels)
    r_i_squared = sum((count / n) ** 2 for count in cluster_sizes.values())
    smooth_metric = 1.0 / r_i_squared if r_i_squared > 0 else 1.0

    return cluster_centers, smooth_metric, meaningful_clusters


# ============================================================
# OPINION UPDATE RULE
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

    This is the KEY DIFFERENCE between METHOD (alpha > 0) and BASELINE (alpha = 0).
    """
    if agent.ingroup_id == neighbor.ingroup_id:
        return base_epsilon
    else:
        # Outgroup: discount based on asymmetry
        discount_factor = max(0.05, 1.0 - agent.alpha)
        return base_epsilon * discount_factor


def update_opinions(agents: list, network: nx.Graph, base_epsilon: float = 0.2) -> float:
    """
    Modified bounded-confidence update with asymmetry modulation.

    Returns maximum opinion change for convergence checking.
    """
    new_opinions = []

    for idx, agent in enumerate(agents):
        neighbor_indices = list(network.neighbors(idx))

        # Collect opinions within effective confidence bound
        influences = []
        for neighbor_idx in neighbor_indices:
            neighbor = agents[neighbor_idx]

            # Compute effective bound based on asymmetry
            eff_bound = compute_effective_confidence_bound(agent, neighbor, base_epsilon)

            # Check if within bound
            if abs(agent.opinion - neighbor.opinion) <= eff_bound:
                influences.append(neighbor.opinion)

        # Update opinion: weighted average with self (mu=0.5)
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
        agent.opinion = float(np.clip(new_opinion, -1.0, 1.0))

    return max_change


# ============================================================
# METRICS COMPUTATION
# ============================================================

def compute_modularity(network: nx.Graph, cluster_labels: list) -> float:
    """Compute Newman-Girvan modularity for current clustering."""
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

    return float(Q)


def compute_cluster_lifetime(history: list) -> float:
    """Compute mean cluster lifetime (persistence)."""
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

    return float(np.mean(lifetimes)) if lifetimes else 0.0


def check_convergence(history: list, window: int = 50, threshold: float = 0.0005) -> bool:
    """Check if max opinion change < threshold for window consecutive steps."""
    if len(history) < window:
        return False

    recent_changes = [m.opinion_max_change for m in history[-window:]]
    return all(c < threshold for c in recent_changes)


# ============================================================
# RUN SINGLE SIMULATION
# ============================================================

def run_simulation(
    n_agents: int,
    alpha_value: float,
    network_type: str,
    seed: int,
    opinion_distribution: list,
    max_timesteps: int = 500,
    base_epsilon: float = 0.2
) -> SimulationResult:
    """
    Run a single simulation to convergence or max timesteps.

    Args:
        alpha_value: 0 for BASELINE, >0 for METHOD
    """
    logger.info(f"Running simulation: alpha={alpha_value}, n={n_agents}, seed={seed}, network={network_type}")

    try:
        # Initialize
        agents = initialize_agents(n_agents, alpha_value, opinion_distribution, seed)
        network = create_network(n_agents, network_type, seed)

        history = []

        for t in range(max_timesteps):
            # Detect clusters
            cluster_centers, smooth_metric, cluster_count = detect_clusters(agents)

            # Compute modularity
            cluster_labels = [a.ingroup_id for a in agents]
            modularity_Q = compute_modularity(network, cluster_labels)

            # Compute variance
            opinions = [a.opinion for a in agents]
            variance = float(np.var(opinions))

            # Update opinions
            max_change = update_opinions(agents, network, base_epsilon)

            # Record metrics
            metrics = ClusterMetrics(
                timestep=t,
                cluster_count=cluster_count,
                smooth_metric=smooth_metric,
                modularity_Q=modularity_Q,
                variance=variance,
                opinion_max_change=max_change
            )
            history.append(metrics)

            # Log every 100 timesteps
            if t % 100 == 0:
                logger.debug(f"  t={t}: clusters={cluster_count}, variance={variance:.4f}, max_change={max_change:.6f}")

            # Check convergence
            if check_convergence(history):
                logger.debug(f"  Converged at t={t}")
                break

        # Compute final metrics
        final_metrics = history[-1]
        cluster_lifetime = compute_cluster_lifetime(history)

        result = SimulationResult(
            final_cluster_count=final_metrics.cluster_count,
            cluster_lifetime_mean=cluster_lifetime,
            final_variance=final_metrics.variance,
            final_modularity_Q=final_metrics.modularity_Q,
            convergence_time=len(history),
            alpha=alpha_value,
            seed=seed,
            network_type=network_type,
            n_agents=n_agents
        )

        logger.info(f"  Result: clusters={result.final_cluster_count}, lifetime={result.cluster_lifetime_mean:.2f}")

        return result

    except Exception as e:
        logger.error(f"Error in simulation: {e}")
        raise


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment(
    opinion_distribution: list,
    alpha_values: list,
    n_seeds: int,
    n_agents: int,
    max_timesteps: int,
    network_type: str
) -> pd.DataFrame:
    """Run parameter sweep experiment."""
    logger.info(f"Running experiment: {len(alpha_values)} alphas x {n_seeds} seeds")

    results = []
    total_runs = len(alpha_values) * n_seeds
    run_count = 0

    for alpha in alpha_values:
        for seed in range(n_seeds):
            run_count += 1
            logger.info(f"Run {run_count}/{total_runs}: alpha={alpha}, seed={seed}")

            try:
                result = run_simulation(
                    n_agents=n_agents,
                    alpha_value=alpha,
                    network_type=network_type,
                    seed=seed,
                    opinion_distribution=opinion_distribution,
                    max_timesteps=max_timesteps
                )
                results.append(asdict(result))

            except Exception as e:
                logger.error(f"Failed run alpha={alpha}, seed={seed}: {e}")
                # Add failed result with NaN
                results.append({
                    "final_cluster_count": np.nan,
                    "cluster_lifetime_mean": np.nan,
                    "final_variance": np.nan,
                    "final_modularity_Q": np.nan,
                    "convergence_time": np.nan,
                    "alpha": alpha,
                    "seed": seed,
                    "network_type": network_type,
                    "n_agents": n_agents
                })

    return pd.DataFrame(results)


# ============================================================
# MAIN: METHOD VS BASELINE COMPARISON
# ============================================================

def main():
    """
    Main entry point - runs both METHOD and BASELINE and produces
    output in exp_gen_sol_out.json format.
    """
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("ASYMMETRIC BOUNDED-CONFIDENCE MODEL - METHOD VS BASELINE")
    logger.info("=" * 60)

    # Paths
    workspace = Path(__file__).parent

    # Use MINI data for testing (as per TODO instructions)
    data_path = workspace / "dependencies" / "Political_Bias_Opinion_Distribution_Dataset_for_Ag" / "full_data_out.json"
    output_path = workspace / "output" / "method_out.json"
    output_path.parent.mkdir(exist_ok=True)

    # Load opinion distribution
    try:
        opinion_distribution = load_opinion_distribution(data_path)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    data_load_time = time.time() - start_time
    logger.info(f"Data loading took {data_load_time:.2f} seconds")

    # Experiment parameters (full scale for production run)
    n_agents = 100  # Agents per simulation
    n_seeds = 10    # Seeds per alpha value
    max_timesteps = 300  # Max timesteps
    network_type = "small_world"

    # Alpha values: 0 = BASELINE, others = METHOD
    # Generate 20 alpha values from 0.1 to 1.5 (matching empirical ranges from dependency)
    baseline_alpha = 0.0
    method_alphas = [round(0.1 + i * 0.07, 2) for i in range(20)]  # 20 values: 0.1 to 1.43
    # This gives us: 10 seeds × 20 method_alphas = 200 examples

    logger.info(f"Parameters: n_agents={n_agents}, n_seeds={n_seeds}, max_t={max_timesteps}")
    logger.info(f"Baseline alpha: {baseline_alpha}")
    logger.info(f"Method alphas: {method_alphas}")

    # Run BASELINE (alpha=0)
    logger.info("\n" + "=" * 40)
    logger.info("RUNNING BASELINE (alpha=0, no asymmetry)")
    logger.info("=" * 40)

    baseline_start = time.time()
    baseline_results = run_experiment(
        opinion_distribution=opinion_distribution,
        alpha_values=[baseline_alpha],
        n_seeds=n_seeds,
        n_agents=n_agents,
        max_timesteps=max_timesteps,
        network_type=network_type
    )
    baseline_time = time.time() - baseline_start
    logger.info(f"BASELINE experiment took {baseline_time:.2f} seconds ({n_seeds} simulations)")

    # Run METHOD (various alpha values)
    logger.info("\n" + "=" * 40)
    logger.info("RUNNING METHOD (alpha>0, with asymmetry)")
    logger.info("=" * 40)

    method_start = time.time()
    method_results = run_experiment(
        opinion_distribution=opinion_distribution,
        alpha_values=method_alphas,
        n_seeds=n_seeds,
        n_agents=n_agents,
        max_timesteps=max_timesteps,
        network_type=network_type
    )
    method_time = time.time() - method_start
    logger.info(f"METHOD experiment took {method_time:.2f} seconds ({len(method_alphas) * n_seeds} simulations)")

    # Aggregate results for comparison
    logger.info("\n" + "=" * 40)
    logger.info("AGGREGATING RESULTS")
    logger.info("=" * 40)

    baseline_lifetime = baseline_results["cluster_lifetime_mean"].mean()
    baseline_clusters = baseline_results["final_cluster_count"].mean()

    method_lifetime = method_results["cluster_lifetime_mean"].mean()
    method_clusters = method_results["final_cluster_count"].mean()

    logger.info(f"BASELINE: avg_lifetime={baseline_lifetime:.2f}, avg_clusters={baseline_clusters:.1f}")
    logger.info(f"METHOD:   avg_lifetime={method_lifetime:.2f}, avg_clusters={method_clusters:.1f}")

    # Statistical comparison
    all_baseline_lifetimes = baseline_results["cluster_lifetime_mean"].dropna().values
    all_method_lifetimes = method_results["cluster_lifetime_mean"].dropna().values

    if len(all_baseline_lifetimes) > 1 and len(all_method_lifetimes) > 1:
        t_stat, p_value = stats.ttest_ind(all_baseline_lifetimes, all_method_lifetimes)
        logger.info(f"T-test (lifetime): t={t_stat:.3f}, p={p_value:.4f}")
    else:
        t_stat, p_value = np.nan, np.nan
        logger.warning("Not enough samples for t-test")

    # Compute correlation between alpha and lifetime (across method results)
    combined = pd.concat([baseline_results, method_results])
    r, r_pval = stats.pearsonr(combined["alpha"], combined["cluster_lifetime_mean"])
    logger.info(f"Correlation (alpha vs lifetime): r={r:.3f}, p={r_pval:.4f}")

    # Format output according to exp_gen_sol_out.json schema
    logger.info("\n" + "=" * 40)
    logger.info("GENERATING OUTPUT")
    logger.info("=" * 40)

    examples = []

    # Create output for each seed/alpha combination
    for idx, (_, baseline_row) in enumerate(baseline_results.iterrows()):
        # Find corresponding method row with similar seed
        seed = baseline_row["seed"]

        for method_alpha in method_alphas:
            method_row = method_results[
                (method_results["seed"] == seed) &
                (method_results["alpha"] == method_alpha)
            ]

            if len(method_row) == 0:
                continue

            method_row = method_row.iloc[0]

            # Create example entry
            example = {
                "input": f"Simulate opinion dynamics with {n_agents} agents on {network_type} network. "
                         f"Measure cluster stability (lifetime) with perception asymmetry alpha={method_alpha}.",
                "output": f"Hypothesis: Higher perception asymmetry (alpha) leads to more stable opinion bubbles "
                          f"because agents discount outgroup signals, preventing cluster merging.",
                "context": {
                    "experiment_type": "parameter_sweep",
                    "n_agents": n_agents,
                    "network_type": network_type,
                    "max_timesteps": max_timesteps,
                    "seed": int(seed),
                    "baseline_alpha": baseline_alpha,
                    "method_alpha": method_alpha,
                    "correlation_r": float(r),
                    "correlation_p": float(r_pval),
                    "t_statistic": float(t_stat) if not np.isnan(t_stat) else None,
                    "t_pvalue": float(p_value) if not np.isnan(p_value) else None
                },
                "dataset": "cajcodes/political-bias",
                "split": "train",
                "predict_baseline": json.dumps({
                    "alpha": baseline_alpha,
                    "final_cluster_count": int(baseline_row["final_cluster_count"]),
                    "cluster_lifetime_mean": float(baseline_row["cluster_lifetime_mean"]),
                    "final_variance": float(baseline_row["final_variance"]),
                    "convergence_time": int(baseline_row["convergence_time"]),
                    "interpretation": "Standard bounded-confidence without perception asymmetry"
                }),
                "predict_method": json.dumps({
                    "alpha": method_alpha,
                    "final_cluster_count": int(method_row["final_cluster_count"]),
                    "cluster_lifetime_mean": float(method_row["cluster_lifetime_mean"]),
                    "final_variance": float(method_row["final_variance"]),
                    "convergence_time": int(method_row["convergence_time"]),
                    "interpretation": f"Asymmetric bounded-confidence with alpha={method_alpha}"
                }),
                "method": "Asymmetric Bounded-Confidence Agent-Based Model: Modulates confidence bounds "
                          "based on perception asymmetry - agents perceive their ingroup as more diverse "
                          "than outgroups, discounting outgroup signals by factor (1-alpha). "
                          "Tested via parameter sweep varying alpha from 0 (baseline) to 1.2 (extreme)."
            }
            examples.append(example)

    # Create output object
    output = {"examples": examples}

    # Validate output before saving
    logger.info("Validating output structure before saving...")
    validation_errors = validate_output_schema(output)
    if validation_errors:
        for err in validation_errors:
            logger.error(f"Validation error: {err}")
        raise ValueError(f"Output validation failed with {len(validation_errors)} errors")
    logger.info("Output validation PASSED!")

    # Save output
    logger.info(f"Saving output to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Generated {len(examples)} examples")
    logger.info(log_truncated(f"Output preview: {json.dumps(examples[0], indent=2)[:500]}..."))

    # Also save raw results as CSV for further analysis
    combined.to_csv(workspace / "output" / "raw_results.csv", index=False)
    logger.info(f"Saved raw results to: {workspace / 'output' / 'raw_results.csv'}")

    total_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)

    # Runtime statistics
    total_simulations = n_seeds + len(method_alphas) * n_seeds
    time_per_simulation = (baseline_time + method_time) / total_simulations
    logger.info(f"Total runtime: {total_time:.2f} seconds")
    logger.info(f"Total simulations: {total_simulations}")
    logger.info(f"Average time per simulation: {time_per_simulation:.2f} seconds")

    # Estimate for full dataset (same simulation count regardless of data size)
    # The simulation parameters determine runtime, not the number of opinion examples
    logger.info("\n" + "=" * 40)
    logger.info("RUNTIME ESTIMATION FOR FULL DATASET")
    logger.info("=" * 40)
    logger.info(f"Current runtime ({len(opinion_distribution)} opinions): {total_time:.2f} seconds")
    logger.info("Note: Simulation count is independent of dataset size (opinions are sampled)")
    logger.info("Estimated full runtime: same as current (~{:.1f} minutes)".format(total_time / 60))
    logger.info("Full dataset will complete well within 1 hour limit.")

    return output


if __name__ == "__main__":
    main()
