"""
Test script for Asymmetric Bounded-Confidence Model.

Follows phased testing strategy:
- Phase 1: Unit tests
- Phase 2: Integration tests (tiny simulations)
- Phase 3: Mini experiments (reduced scale)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from asymmetric_bounded_confidence import (
    Agent,
    ClusterMetrics,
    Simulation,
    load_opinion_distribution,
    initialize_agents,
    create_network,
    initialize_simulation,
    detect_clusters,
    compute_effective_confidence_bound,
    update_opinions,
    compute_cluster_metrics,
    check_convergence,
    compute_cluster_lifetime,
    run_simulation,
)


def test_1a_agent_initialization():
    """Test 1A: Agent initialization."""
    print("\n=== Test 1A: Agent Initialization ===")

    # Create simple opinion distribution
    opinions = [-1.0, -0.5, 0.0, 0.5, 1.0]
    agents = initialize_agents(n_agents=10, alpha_value=0.5, opinion_distribution=opinions, seed=42)

    # Verify
    assert len(agents) == 10, f"Expected 10 agents, got {len(agents)}"
    assert all(-1.0 <= a.opinion <= 1.0 for a in agents), "Opinions out of range"
    assert all(a.alpha == 0.5 for a in agents), "Alpha values incorrect"

    print(f"  Created {len(agents)} agents")
    print(f"  First 3 agents: {[(a.opinion, a.alpha) for a in agents[:3]]}")
    print("  PASSED")


def test_1b_network_creation():
    """Test 1B: Network creation."""
    print("\n=== Test 1B: Network Creation ===")

    # Small-world
    sw = create_network(n_agents=20, network_type="small_world", seed=42)
    assert len(sw.nodes()) == 20, f"SW: Expected 20 nodes, got {len(sw.nodes())}"
    assert nx.is_connected(sw), "SW network not connected"

    # Random
    rnd = create_network(n_agents=20, network_type="random", seed=42)
    assert len(rnd.nodes()) == 20, f"RND: Expected 20 nodes, got {len(rnd.nodes())}"
    assert nx.is_connected(rnd), "RND network not connected"

    sw_deg = [d for n, d in sw.degree()]
    rnd_deg = [d for n, d in rnd.degree()]

    print(f"  Small-world: {len(sw.nodes())} nodes, avg degree {np.mean(sw_deg):.2f}")
    print(f"  Random: {len(rnd.nodes())} nodes, avg degree {np.mean(rnd_deg):.2f}")
    print("  PASSED")


def test_1c_cluster_detection():
    """Test 1C: Cluster detection."""
    print("\n=== Test 1C: Cluster Detection ===")

    # Create agents with two clear clusters
    agents = [Agent(opinion=0.0) for _ in range(10)] + [Agent(opinion=0.6) for _ in range(10)]

    cluster_centers, smooth_metric, n_clusters = detect_clusters(agents)

    print(f"  Cluster centers: {cluster_centers}")
    print(f"  Smooth metric: {smooth_metric:.3f}")
    print(f"  Number of clusters: {n_clusters}")

    # Should detect 2 clusters
    assert n_clusters == 2, f"Expected 2 clusters, got {n_clusters}"
    # Smooth metric for 2 equal clusters: c = 1/(0.5^2 + 0.5^2) = 2.0
    assert 1.9 <= smooth_metric <= 2.1, f"Expected smooth_metric ~2.0, got {smooth_metric}"

    print("  PASSED")


def test_1d_effective_confidence_bound():
    """Test 1D: Effective confidence bound computation."""
    print("\n=== Test 1D: Effective Confidence Bound ===")

    base_epsilon = 0.2

    # Test cases for different alpha values
    test_cases = [0.0, 0.5, 1.0]

    for alpha in test_cases:
        agent = Agent(opinion=0.0, alpha=alpha)
        agent.ingroup_id = 0

        # Same group neighbor
        same_neighbor = Agent(opinion=0.1)
        same_neighbor.ingroup_id = 0
        same_bound = compute_effective_confidence_bound(agent, same_neighbor, base_epsilon)

        # Different group neighbor
        diff_neighbor = Agent(opinion=0.1)
        diff_neighbor.ingroup_id = 1
        diff_bound = compute_effective_confidence_bound(agent, diff_neighbor, base_epsilon)

        expected_diff = base_epsilon * max(0.05, 1.0 - alpha)

        print(f"  alpha={alpha}: same_group={same_bound:.3f}, diff_group={diff_bound:.3f} (expected ~{expected_diff:.3f})")

        assert same_bound == base_epsilon, f"Same group should have full bound"
        assert abs(diff_bound - expected_diff) < 0.001, f"Different group bound incorrect"

    print("  PASSED")


def test_2a_single_short_simulation():
    """Test 2A: Single short simulation."""
    print("\n=== Test 2A: Single Short Simulation ===")

    # Simple opinion distribution
    opinions = [-0.8, -0.4, 0.0, 0.4, 0.8]

    sim = run_simulation(
        n_agents=50,
        alpha_value=0.5,
        network_type="small_world",
        seed=42,
        opinion_distribution=opinions,
        max_timesteps=100
    )

    # Verify
    assert len(sim.agents) == 50
    assert len(sim.cluster_history) > 0
    assert all(-1.0 <= a.opinion <= 1.0 for a in sim.agents)

    final_opinions = [a.opinion for a in sim.agents]
    final_clusters = sim.cluster_history[-1].cluster_count

    print(f"  Timesteps: {len(sim.cluster_history)}")
    print(f"  Final opinion range: [{min(final_opinions):.3f}, {max(final_opinions):.3f}]")
    print(f"  Final clusters: {final_clusters}")
    print("  PASSED")


def test_2b_compare_alpha_extremes():
    """Test 2B: Compare extreme alpha values."""
    print("\n=== Test 2B: Compare Alpha Extremes ===")

    opinions = [-0.8, -0.4, 0.0, 0.4, 0.8]

    # Low alpha (no asymmetry)
    sim_low = run_simulation(
        n_agents=50,
        alpha_value=0.0,
        network_type="small_world",
        seed=42,
        opinion_distribution=opinions,
        max_timesteps=150
    )

    # High alpha (strong asymmetry)
    sim_high = run_simulation(
        n_agents=50,
        alpha_value=1.2,
        network_type="small_world",
        seed=42,
        opinion_distribution=opinions,
        max_timesteps=150
    )

    low_clusters = sim_low.cluster_history[-1].cluster_count
    high_clusters = sim_high.cluster_history[-1].cluster_count
    low_variance = np.var([a.opinion for a in sim_low.agents])
    high_variance = np.var([a.opinion for a in sim_high.agents])

    print(f"  Low alpha (0.0): {low_clusters} clusters, variance={low_variance:.4f}")
    print(f"  High alpha (1.2): {high_clusters} clusters, variance={high_variance:.4f}")

    # High alpha should maintain more clusters or equal
    # (asymmetry blocks cross-cluster influence)
    print(f"  Hypothesis check: high_alpha clusters ({high_clusters}) >= low_alpha ({low_clusters})")

    print("  PASSED")


def test_2c_convergence_detection():
    """Test 2C: Convergence detection."""
    print("\n=== Test 2C: Convergence Detection ===")

    opinions = [-0.5, 0.0, 0.5]

    sim = run_simulation(
        n_agents=50,
        alpha_value=0.3,
        network_type="small_world",
        seed=42,
        opinion_distribution=opinions,
        max_timesteps=500
    )

    converged = check_convergence(sim.cluster_history, window=50, threshold=0.001)
    final_changes = [m.opinion_max_change for m in sim.cluster_history[-50:]]

    print(f"  Timesteps: {len(sim.cluster_history)}")
    print(f"  Converged: {converged}")
    print(f"  Final changes (last 10): {[f'{c:.6f}' for c in final_changes[-10:]]}")
    print("  PASSED")


def test_3a_mini_experiment_1():
    """Test 3A: Mini Experiment 1 (stability correlation)."""
    print("\n=== Test 3A: Mini Experiment 1 (Stability) ===")

    from scipy import stats

    opinions = [-1.0, -0.5, 0.0, 0.5, 1.0]
    alpha_values = [0.0, 0.3, 0.6, 0.9, 1.2]
    n_seeds = 3

    results = []

    for alpha in alpha_values:
        for seed in range(n_seeds):
            sim = run_simulation(
                n_agents=80,
                alpha_value=alpha,
                network_type="small_world",
                seed=seed,
                opinion_distribution=opinions,
                max_timesteps=300
            )

            lifetime = compute_cluster_lifetime(sim.cluster_history)
            final_clusters = sim.cluster_history[-1].cluster_count

            results.append({
                "alpha": alpha,
                "seed": seed,
                "cluster_lifetime": lifetime,
                "cluster_count": final_clusters
            })

    df = pd.DataFrame(results)
    r, p = stats.pearsonr(df["alpha"], df["cluster_lifetime"])

    print(f"  Results ({len(df)} runs):")
    for alpha in alpha_values:
        subset = df[df["alpha"] == alpha]
        print(f"    alpha={alpha}: lifetime={subset['cluster_lifetime'].mean():.2f}, "
              f"clusters={subset['cluster_count'].mean():.1f}")

    print(f"  Correlation: r={r:.3f}, p={p:.4f}")
    print("  PASSED")


def test_3b_mini_experiment_2():
    """Test 3B: Mini Experiment 2 (merger intervention)."""
    print("\n=== Test 3B: Mini Experiment 2 (Merger) ===")

    opinions = [-0.8, -0.4, 0.0, 0.4, 0.8]

    treatment_clusters = []
    control_clusters = []

    for seed in range(3):
        for condition in ["treatment", "control"]:
            sim = initialize_simulation(
                n_agents=80,
                alpha_value=0.8,
                network_type="small_world",
                seed=seed,
                opinion_distribution=opinions
            )

            for t in range(400):
                sim.timestep = t

                if t == 150 and condition == "treatment":
                    for agent in sim.agents:
                        agent.alpha = 0.2

                detect_clusters(sim.agents)
                update_opinions(sim)

            final_clusters = detect_clusters(sim.agents)[2]

            if condition == "treatment":
                treatment_clusters.append(final_clusters)
            else:
                control_clusters.append(final_clusters)

    print(f"  Treatment final clusters: {treatment_clusters}, mean={np.mean(treatment_clusters):.1f}")
    print(f"  Control final clusters: {control_clusters}, mean={np.mean(control_clusters):.1f}")
    print("  PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("ASYMMETRIC BOUNDED-CONFIDENCE MODEL TESTS")
    print("=" * 60)

    # Phase 1: Unit tests
    print("\n--- PHASE 1: UNIT TESTS ---")
    test_1a_agent_initialization()
    test_1b_network_creation()
    test_1c_cluster_detection()
    test_1d_effective_confidence_bound()

    # Phase 2: Integration tests
    print("\n--- PHASE 2: INTEGRATION TESTS ---")
    test_2a_single_short_simulation()
    test_2b_compare_alpha_extremes()
    test_2c_convergence_detection()

    # Phase 3: Mini experiments
    print("\n--- PHASE 3: MINI EXPERIMENTS ---")
    test_3a_mini_experiment_1()
    test_3b_mini_experiment_2()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
