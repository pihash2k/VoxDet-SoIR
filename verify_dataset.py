# verify_dataset.py
"""
Verification script for VoxDet Instance Retrieval Dataset
Checks dataset integrity, structure, and provides detailed statistics
"""

import torch
import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm


def verify_paths_exist(anns, base_path):
    """Verify that all annotated image paths actually exist."""
    missing_files = []
    existing_files = 0

    print("\n📁 Verifying file paths...")
    for img_path in tqdm(anns.keys(), desc="Checking paths"):
        # Handle both absolute and relative paths
        if Path(img_path).is_absolute():
            full_path = Path(img_path)
        else:
            full_path = base_path / img_path

        if full_path.exists():
            existing_files += 1
        else:
            missing_files.append(str(img_path))

    return existing_files, missing_files


def verify_annotation_structure(anns):
    """Verify that all annotations have required fields."""
    required_fields = {"bbox", "is_query", "is_val", "ins", "set"}
    optional_fields = {"mask", "num_ins"}

    print("\n🔍 Verifying annotation structure...")

    issues = defaultdict(list)
    field_stats = defaultdict(int)

    for img_path, ann in tqdm(anns.items(), desc="Checking annotations"):
        # Check required fields
        for field in required_fields:
            if field not in ann:
                issues["missing_required"].append(
                    f"{img_path}: missing '{field}'"
                )
            else:
                field_stats[field] += 1

        # Check field types and values
        if "is_query" in ann:
            if not isinstance(ann["is_query"], bool):
                issues["type_error"].append(f"{img_path}: 'is_query' not bool")

        if "is_val" in ann:
            if not isinstance(ann["is_val"], bool):
                issues["type_error"].append(f"{img_path}: 'is_val' not bool")

        if "set" in ann:
            if ann["set"] not in ["OWID", "RoboTools"]:
                issues["invalid_value"].append(
                    f"{img_path}: invalid set '{ann['set']}'"
                )

        # Verify bbox format
        if "bbox" in ann:
            if ann["is_query"]:
                # Query should have single bbox
                if not (
                    isinstance(ann["bbox"], (list, tuple))
                    and len(ann["bbox"]) == 4
                ):
                    issues["bbox_format"].append(
                        f"{img_path}: query bbox format error"
                    )
            else:
                # Gallery should have list of bboxes
                if not isinstance(ann["bbox"], list):
                    issues["bbox_format"].append(
                        f"{img_path}: gallery bbox should be list"
                    )

        # Track optional fields
        for field in optional_fields:
            if field in ann:
                field_stats[field] += 1

    return issues, field_stats


def analyze_retrieval_statistics(anns):
    """Analyze instance retrieval specific statistics."""
    print("\n📊 Analyzing retrieval statistics...")

    # Separate queries and gallery
    queries = {k: v for k, v in anns.items() if v.get("is_query", False)}
    gallery = {k: v for k, v in anns.items() if not v.get("is_query", False)}

    # Instance frequency analysis
    instance_frequency = defaultdict(int)
    instances_per_image = []

    for ann in gallery.values():
        if "ins" in ann:
            ins_list = (
                ann["ins"] if isinstance(ann["ins"], list) else [ann["ins"]]
            )
            instances_per_image.append(len(ins_list))
            for ins_id in ins_list:
                instance_frequency[ins_id] += 1

    # Query-Gallery matching
    query_matches = defaultdict(list)
    for q_path, q_ann in queries.items():
        if "ins" in q_ann:
            q_ins = q_ann["ins"]
            for g_path, g_ann in gallery.items():
                if "ins" in g_ann:
                    g_ins_list = (
                        g_ann["ins"]
                        if isinstance(g_ann["ins"], list)
                        else [g_ann["ins"]]
                    )
                    if q_ins in g_ins_list:
                        query_matches[q_ins].append(g_path)

    # Calculate retrieval difficulty
    retrieval_difficulty = {
        "easy": sum(1 for freq in instance_frequency.values() if freq > 10),
        "medium": sum(
            1 for freq in instance_frequency.values() if 5 <= freq <= 10
        ),
        "hard": sum(1 for freq in instance_frequency.values() if 1 <= freq < 5),
        "singleton": sum(
            1 for freq in instance_frequency.values() if freq == 1
        ),
    }

    # Dataset split statistics
    dataset_splits = {
        "OWID": {"query": 0, "gallery": 0, "train": 0, "val": 0},
        "RoboTools": {"query": 0, "gallery": 0, "test": 0},
    }

    for ann in anns.values():
        if "set" in ann:
            ds = ann["set"]
            if ann.get("is_query", False):
                dataset_splits[ds]["query"] += 1
            else:
                dataset_splits[ds]["gallery"] += 1

                if ds == "OWID":
                    if ann.get("is_val", False):
                        dataset_splits[ds]["val"] += 1
                    else:
                        dataset_splits[ds]["train"] += 1
                elif ds == "RoboTools":
                    dataset_splits[ds]["test"] += 1

    return {
        "n_queries": len(queries),
        "n_gallery": len(gallery),
        "n_unique_instances": len(instance_frequency),
        "instance_frequency": instance_frequency,
        "instances_per_image": instances_per_image,
        "query_matches": query_matches,
        "retrieval_difficulty": retrieval_difficulty,
        "dataset_splits": dataset_splits,
    }


def verify_masks(anns, sample_size=10):
    """Verify mask encoding and format."""
    print(f"\n🎭 Verifying masks (sampling {sample_size} entries)...")

    mask_stats = {
        "with_mask": 0,
        "without_mask": 0,
        "valid_rle": 0,
        "invalid_rle": 0,
    }

    mask_entries = [
        (k, v) for k, v in anns.items() if "mask" in v and v["mask"]
    ]

    # Sample verification
    sample = mask_entries[: min(sample_size, len(mask_entries))]

    for img_path, ann in sample:
        if "mask" in ann and ann["mask"]:
            mask_stats["with_mask"] += 1

            # Check RLE encoding structure
            mask = ann["mask"]
            if ann.get("is_query", False):
                # Query should have single mask
                if (
                    isinstance(mask, dict)
                    and "counts" in mask
                    and "size" in mask
                ):
                    mask_stats["valid_rle"] += 1
                else:
                    mask_stats["invalid_rle"] += 1
            else:
                # Gallery might have list of masks
                if isinstance(mask, list):
                    for m in mask:
                        if (
                            isinstance(m, dict)
                            and "counts" in m
                            and "size" in m
                        ):
                            mask_stats["valid_rle"] += 1
                        else:
                            mask_stats["invalid_rle"] += 1
        else:
            mask_stats["without_mask"] += 1

    return mask_stats


def check_data_consistency(anns):
    """Check consistency between related fields."""
    print("\n⚖️ Checking data consistency...")

    consistency_issues = []

    for img_path, ann in anns.items():
        # Check bbox and ins list consistency
        if not ann.get("is_query", False):  # Gallery images
            if "bbox" in ann and "ins" in ann:
                bbox_list = (
                    ann["bbox"]
                    if isinstance(ann["bbox"], list)
                    else [ann["bbox"]]
                )
                ins_list = (
                    ann["ins"] if isinstance(ann["ins"], list) else [ann["ins"]]
                )

                if len(bbox_list) != len(ins_list):
                    consistency_issues.append(
                        f"{img_path}: bbox count ({len(bbox_list)}) != ins count ({len(ins_list)})"
                    )

        # Check mask and bbox consistency
        if "mask" in ann and ann["mask"] and "bbox" in ann:
            if ann.get("is_query", False):
                # Query: should have 1 mask and 1 bbox
                if isinstance(ann["mask"], list) or isinstance(
                    ann["bbox"][0], list
                ):
                    consistency_issues.append(
                        f"{img_path}: query has multiple masks/bboxes"
                    )
            else:
                # Gallery: mask count should match bbox count
                mask_list = (
                    ann["mask"]
                    if isinstance(ann["mask"], list)
                    else [ann["mask"]]
                )
                bbox_list = (
                    ann["bbox"]
                    if isinstance(ann["bbox"], list)
                    else [ann["bbox"]]
                )

                if len(mask_list) != len(bbox_list):
                    consistency_issues.append(
                        f"{img_path}: mask count ({len(mask_list)}) != bbox count ({len(bbox_list)})"
                    )

    return consistency_issues


def generate_report(results):
    """Generate a detailed verification report."""
    print("\n" + "=" * 60)
    print("📋 DATASET VERIFICATION REPORT")
    print("=" * 60)

    # File verification
    print(f"\n✅ Files found: {results['files']['existing']}")
    if results["files"]["missing"]:
        print(f"❌ Missing files: {len(results['files']['missing'])}")
        print(f"   First 5 missing: {results['files']['missing'][:5]}")

    # Structure verification
    print(f"\n📐 Annotation Structure:")
    for field, count in results["structure"]["field_stats"].items():
        print(f"   {field}: {count} entries")

    if results["structure"]["issues"]:
        print(f"\n⚠️  Structure Issues:")
        for issue_type, issues in results["structure"]["issues"].items():
            if issues:
                print(f"   {issue_type}: {len(issues)} issues")
                if len(issues) > 0:
                    print(f"      Example: {issues[0]}")

    # Retrieval statistics
    stats = results["retrieval_stats"]
    print(f"\n🔍 Instance Retrieval Statistics:")
    print(f"   Query instances: {stats['n_queries']}")
    print(f"   Gallery images: {stats['n_gallery']}")
    print(f"   Unique instances: {stats['n_unique_instances']}")

    if stats["instances_per_image"]:
        print(
            f"   Avg instances/image: {np.mean(stats['instances_per_image']):.2f}"
        )
        print(f"   Max instances/image: {max(stats['instances_per_image'])}")

    print(f"\n📊 Retrieval Difficulty Distribution:")
    for difficulty, count in stats["retrieval_difficulty"].items():
        print(f"   {difficulty}: {count} instances")

    print(f"\n🗂️  Dataset Splits:")
    for dataset, splits in stats["dataset_splits"].items():
        print(f"   {dataset}:")
        for split, count in splits.items():
            if count > 0:
                print(f"      {split}: {count}")

    # Query coverage
    queries_with_matches = sum(
        1 for matches in stats["query_matches"].values() if len(matches) > 0
    )
    queries_without_matches = stats["n_queries"] - queries_with_matches
    print(f"\n🎯 Query Coverage:")
    print(f"   Queries with gallery matches: {queries_with_matches}")
    print(f"   Queries without matches: {queries_without_matches}")

    # Mask verification
    if "masks" in results:
        print(f"\n🎭 Mask Verification:")
        for key, value in results["masks"].items():
            print(f"   {key}: {value}")

    # Consistency check
    if results["consistency"]:
        print(f"\n⚠️  Consistency Issues: {len(results['consistency'])}")
        for issue in results["consistency"][:3]:
            print(f"   - {issue}")
    else:
        print(f"\n✅ Data Consistency: No issues found")

    # Overall status
    print("\n" + "=" * 60)
    total_issues = (
        len(results["files"]["missing"])
        + sum(len(v) for v in results["structure"]["issues"].values())
        + len(results["consistency"])
    )

    if total_issues == 0:
        print("✅ DATASET VERIFICATION: PASSED")
    else:
        print(f"⚠️  DATASET VERIFICATION: {total_issues} issues found")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Verify VoxDet Instance Retrieval Dataset"
    )
    parser.add_argument(
        "--ann-file",
        type=str,
        default="anns.pt",
        help="Path to annotation file",
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=".",
        help="Base path for relative image paths",
    )
    parser.add_argument(
        "--skip-paths", action="store_true", help="Skip path verification"
    )
    parser.add_argument(
        "--sample-masks",
        type=int,
        default=10,
        help="Number of masks to sample for verification",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print(f"🔄 Loading annotations from {args.ann_file}...")
    try:
        anns = torch.load(args.ann_file)
        print(f"✅ Loaded {len(anns)} annotations")
    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        return

    results = {}

    # Verify paths
    if not args.skip_paths:
        existing, missing = verify_paths_exist(anns, Path(args.base_path))
        results["files"] = {"existing": existing, "missing": missing}
    else:
        results["files"] = {"existing": len(anns), "missing": []}

    # Verify structure
    issues, field_stats = verify_annotation_structure(anns)
    results["structure"] = {"issues": issues, "field_stats": field_stats}

    # Analyze retrieval statistics
    results["retrieval_stats"] = analyze_retrieval_statistics(anns)

    # Verify masks
    results["masks"] = verify_masks(anns, args.sample_masks)

    # Check consistency
    results["consistency"] = check_data_consistency(anns)

    # Generate report
    generate_report(results)

    # Save detailed results if verbose
    if args.verbose:
        output_file = args.ann_file.replace(".pt", "_verification.json")
        with open(output_file, "w") as f:
            # Convert defaultdict to dict for JSON serialization
            json_results = {
                "files": results["files"],
                "structure": {
                    "issues": {
                        k: v[:10]
                        for k, v in results["structure"]["issues"].items()
                    },
                    "field_stats": dict(results["structure"]["field_stats"]),
                },
                "retrieval_stats": {
                    "n_queries": results["retrieval_stats"]["n_queries"],
                    "n_gallery": results["retrieval_stats"]["n_gallery"],
                    "n_unique_instances": results["retrieval_stats"][
                        "n_unique_instances"
                    ],
                    "retrieval_difficulty": results["retrieval_stats"][
                        "retrieval_difficulty"
                    ],
                    "dataset_splits": results["retrieval_stats"][
                        "dataset_splits"
                    ],
                },
                "masks": results["masks"],
                "consistency": results["consistency"][:10],
            }
            json.dump(json_results, f, indent=2)
        print(f"\n💾 Detailed results saved to {output_file}")


if __name__ == "__main__":
    main()
