# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Data processing script for Opinion Dynamics dataset.

Selected dataset: cajcodes/political-bias
Reason: Best continuous 5-point scale (-1.0 to +1.0) with multimodal clustering,
ideal for agent-based opinion dynamics simulations.

Output format follows exp_sel_data_out.json schema:
- examples: array of {input, context, output, dataset, split}
"""

import json
from pathlib import Path
from typing import Any

# Configuration
WORKSPACE = Path("/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260120_210012/invention_loop/iter1_dataset_workspace_idx0")
DATASETS_DIR = WORKSPACE / "temp" / "datasets"
EXAMPLES_COUNT = 200


def load_json(filepath: Path) -> list[dict[str, Any]]:
    """Load JSON file and return list of records."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def truncate_strings(obj: Any, max_len: int = 200) -> Any:
    """Recursively truncate all strings in an object to max_len characters."""
    if isinstance(obj, str):
        return obj[:max_len] + "..." if len(obj) > max_len else obj
    elif isinstance(obj, dict):
        return {k: truncate_strings(v, max_len) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [truncate_strings(item, max_len) for item in obj]
    return obj


def process_cajcodes_political_bias(data: list[dict]) -> list[dict]:
    """
    Process cajcodes/political-bias dataset.

    Structure: {"text": str, "label": int (0-4)}
    - 0: Far Right, 1: Right, 2: Center, 3: Left, 4: Far Left

    This is ideal for opinion dynamics as it has a continuous 5-point scale!
    Maps to -1.0 to +1.0 for continuous opinion modeling.
    """
    examples = []
    label_map = {0: "far_right", 1: "right", 2: "center", 3: "left", 4: "far_left"}
    # Convert 0-4 to -1 to +1 scale for opinion dynamics
    opinion_map = {0: -1.0, 1: -0.5, 2: 0.0, 3: 0.5, 4: 1.0}

    for i, item in enumerate(data[:EXAMPLES_COUNT]):
        label = item.get("label", 2)
        examples.append({
            "input": f"Classify the political bias of this statement: \"{item['text']}\"",
            "context": {
                "opinion_score": opinion_map.get(label, 0.0),
                "bias_category": label_map.get(label, "center"),
                "raw_label": label,
                "scale": "5-point (0=far_right to 4=far_left)",
                "normalized_scale": "-1.0 to +1.0 (right to left)"
            },
            "output": f"{label_map.get(label, 'center')} (opinion_score: {opinion_map.get(label, 0.0):.1f})",
            "dataset": "cajcodes/political-bias",
            "split": "train"
        })

    return examples


def main():
    """Main processing function."""
    print("=" * 60)
    print("Opinion Dynamics Dataset Processing")
    print("Selected: cajcodes/political-bias")
    print("=" * 60)

    # Load cajcodes/political-bias dataset
    print("\nProcessing cajcodes/political-bias...")
    filepath = DATASETS_DIR / "full_cajcodes_political-bias_train.json"

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    data = load_json(filepath)
    examples = process_cajcodes_political_bias(data)

    print(f"   Processed {len(examples)} examples (continuous 5-point scale)")
    print(f"   Opinion scores range: -1.0 (far_right) to +1.0 (far_left)")

    # Create output in schema format: {"examples": [...]}
    output = {"examples": examples}

    # Generate three versions:
    # 1. full_data_out.json - All 200 examples
    full_output_file = WORKSPACE / "full_data_out.json"
    with open(full_output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nGenerated: {full_output_file} ({len(examples)} examples)")

    # 2. mini_data_out.json - First 3 examples
    mini_output = {"examples": examples[:3]}
    mini_output_file = WORKSPACE / "mini_data_out.json"
    with open(mini_output_file, "w", encoding="utf-8") as f:
        json.dump(mini_output, f, indent=2, ensure_ascii=False)
    print(f"Generated: {mini_output_file} (3 examples)")

    # 3. preview_data_out.json - First 3 examples with truncated strings
    preview_output = {"examples": truncate_strings(examples[:3])}
    preview_output_file = WORKSPACE / "preview_data_out.json"
    with open(preview_output_file, "w", encoding="utf-8") as f:
        json.dump(preview_output, f, indent=2, ensure_ascii=False)
    print(f"Generated: {preview_output_file} (3 examples, truncated)")

    print(f"\n{'=' * 60}")
    print(f"Dataset Summary:")
    print(f"  - Total examples: {len(examples)}")
    print(f"  - Dataset: cajcodes/political-bias")
    print(f"  - Full output size: {full_output_file.stat().st_size / 1024:.1f} KB")

    # Show label distribution
    label_counts = {}
    for ex in examples:
        label = ex["context"]["bias_category"]
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {label}: {count} examples")

    print("\nDone!")


if __name__ == "__main__":
    main()
