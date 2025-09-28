# VoxDet Instance Retrieval Dataset

A comprehensive instance retrieval dataset built upon the VoxDet foundation, combining synthetic (OWID) and real-world (RoboTools) data for visual instance search and retrieval tasks.

## Overview

This dataset extends VoxDet for **instance retrieval** applications, where the primary task is to retrieve all occurrences of a query object instance from a large gallery of images. Unlike traditional object detection that focuses on category-level detection, this dataset emphasizes **instance-level retrieval** - finding the exact same object or highly similar instances across different viewpoints, lighting conditions, and contexts.

### Key Features
- **Instance-level annotations** for precise object retrieval
- **Query-gallery paradigm** optimized for retrieval benchmarking  
- **Cross-domain evaluation** with synthetic-to-real transfer capabilities
- **Multi-instance support** for complex retrieval scenarios

## Instance Retrieval Task

In instance retrieval, given a query image containing a specific object instance, the goal is to:
1. Search through a large gallery of images
2. Retrieve all images containing that specific instance
3. Localize the instance with bounding boxes and masks
4. Rank results by relevance/confidence

This differs from category-level detection by requiring fine-grained instance matching rather than category classification.

## Dataset Components

### 1. OWID (Open World Instance Detection)
- **Purpose**: Training and validation for instance retrieval models
- **Type**: Synthetic dataset with controlled variations
- **Content**: Computer-generated images with precise instance annotations
- **Retrieval Setup**: Query-gallery pairs with ground truth matches
- **Split**: Training and validation sets for retrieval evaluation

### 2. RoboTools  
- **Purpose**: Real-world instance retrieval evaluation
- **Type**: Real-world dataset from robotic/industrial scenarios
- **Content**: Challenging real-world instances with occlusions and viewpoint changes
- **Retrieval Setup**: Video frame queries to retrieve from test scenes
- **Split**: Test set only for zero-shot retrieval evaluation

## Directory Structure

```
VoxDet-InstanceRetrieval/
├── OWID/
│   ├── P1/                    # Query instances directory
│   │   ├── <category_id>/     # Instance category folders
│   │   │   ├── rgb/           # Query instance images
│   │   │   └── mask/          # Instance segmentation masks
│   │   └── ...
│   └── P2/                    # Gallery for retrieval
│       ├── images/            # Gallery images to search
│       ├── train_annotations.json  # Training instance annotations
│       └── val_annotations.json    # Validation instance annotations
│
├── RoboTools/
│   ├── test/                  # Gallery for retrieval evaluation
│   │   ├── scene_gt_coco_all.json  # Ground truth instance locations
│   │   └── *.jpg/png          # Gallery images
│   └── test_video/            # Query instances from videos
│       ├── <category_name>_<category_id>/
│       │   ├── rgb/           # Query instance frames
│       │   └── mask/          # Instance segmentation masks
│       └── ...
│
└── anns.pt                    # Processed retrieval annotations
```

## Retrieval Annotation Format

### PyTorch Dictionary Structure (`anns.pt`)

The annotations are optimized for instance retrieval tasks:

```python
{
    "<image_path>": {
        "bbox": [...],        # Instance bounding boxes
        "mask": [...],        # Instance segmentation masks (RLE)
        "is_query": bool,     # Query instance flag
        "is_val": bool,       # Validation set flag
        "ins": [...],         # Instance IDs for retrieval matching
        "set": str,           # Dataset source ("OWID" or "RoboTools")
        "num_ins": [...]      # Instance count (OWID only)
    },
    ...
}
```

### Instance Retrieval Fields

#### Query Instances
- **bbox**: Precise instance location `(x1, y1, x2, y2)`
- **mask**: Instance segmentation for accurate matching
- **is_query**: `True` - marks retrieval query
- **ins**: Unique instance identifier for matching
- **set**: Dataset source for cross-domain evaluation

#### Gallery Images (Retrieval Targets)
- **bbox**: List of all instance locations for retrieval evaluation
- **mask**: Instance masks for IoU-based retrieval metrics
- **is_query**: `False` - marks gallery/database image
- **ins**: List of instance IDs present (for ground truth matching)
- **num_ins**: Instance counts for retrieval difficulty assessment

## Instance Retrieval Usage

### Processing for Retrieval

```bash
# Process full retrieval dataset
python create_dataset.py --voxdet-root /path/to/VoxDet

# Process for specific retrieval scenarios
python create_dataset.py --voxdet-root /path/to/VoxDet --output-file retrieval_anns.pt

# Synthetic-only retrieval (OWID)
python create_dataset.py --skip-robotools

# Real-world retrieval evaluation (RoboTools)
python create_dataset.py --skip-owid
```

### Instance Retrieval in Python

```python
import torch
from collections import defaultdict

# Load retrieval annotations
anns = torch.load('anns.pt')

# Separate query and gallery for retrieval
queries = {k: v for k, v in anns.items() if v['is_query']}
gallery = {k: v for k, v in anns.items() if not v['is_query']}

# Instance retrieval setup
def setup_retrieval_pairs():
    """Create query-gallery pairs for instance retrieval evaluation"""
    retrieval_pairs = defaultdict(list)
    
    for query_path, query_ann in queries.items():
        instance_id = query_ann['ins']
        
        # Find all gallery images containing this instance
        for gallery_path, gallery_ann in gallery.items():
            if instance_id in gallery_ann['ins']:
                retrieval_pairs[query_path].append({
                    'gallery_path': gallery_path,
                    'instance_bbox': gallery_ann['bbox'][
                        gallery_ann['ins'].index(instance_id)
                    ],
                    'dataset': gallery_ann['set']
                })
    
    return retrieval_pairs

# Evaluate retrieval performance
def evaluate_retrieval(retrieved_images, ground_truth):
    """Calculate retrieval metrics (mAP, Recall@K, etc.)"""
    # Implementation of retrieval metrics
    pass

# Cross-domain retrieval analysis
synthetic_queries = {k: v for k, v in queries.items() if v['set'] == 'OWID'}
real_queries = {k: v for k, v in queries.items() if v['set'] == 'RoboTools'}

print(f"Total query instances: {len(queries)}")
print(f"Gallery size: {len(gallery)}")
print(f"Synthetic queries (OWID): {len(synthetic_queries)}")
print(f"Real-world queries (RoboTools): {len(real_queries)}")
```

## Retrieval Evaluation Metrics

Standard instance retrieval metrics for evaluation:

```python
# Instance Retrieval Metrics
- mean Average Precision (mAP)
- Recall@K (K=1, 5, 10, 50, 100)
- Precision@K
- Instance-level IoU matching
- Ranking quality measures
```

## Dataset Statistics for Retrieval

```python
import torch
from collections import Counter, defaultdict

anns = torch.load('anns.pt')

# Retrieval statistics
queries = {k: v for k, v in anns.items() if v['is_query']}
gallery = {k: v for k, v in anns.items() if not v['is_query']}

# Instance frequency in gallery (retrieval difficulty)
instance_frequency = defaultdict(int)
for ann in gallery.values():
    for ins_id in ann['ins']:
        instance_frequency[ins_id] += 1

# Retrieval complexity analysis
easy_instances = sum(1 for freq in instance_frequency.values() if freq > 10)
medium_instances = sum(1 for freq in instance_frequency.values() if 5 <= freq <= 10)
hard_instances = sum(1 for freq in instance_frequency.values() if freq < 5)

print(f"=== Instance Retrieval Statistics ===")
print(f"Query instances: {len(queries)}")
print(f"Gallery images: {len(gallery)}")
print(f"Unique instances: {len(instance_frequency)}")
print(f"Average instances per gallery image: {sum(len(v['ins']) for v in gallery.values())/len(gallery):.2f}")
print(f"\n=== Retrieval Difficulty Distribution ===")
print(f"Easy (>10 occurrences): {easy_instances}")
print(f"Medium (5-10 occurrences): {medium_instances}")
print(f"Hard (<5 occurrences): {hard_instances}")
```

## Requirements

- Python 3.7+
- PyTorch
- PIL (Pillow)
- NumPy
- tqdm
- pathlib

## Installation

```bash
# Install required packages
pip install torch pillow numpy tqdm

# Download base VoxDet datasets
# Ensure OWID and RoboTools follow the expected structure

# Process for instance retrieval
python create_dataset.py --voxdet-root /path/to/VoxDet
```

## Instance Retrieval Applications

This dataset supports various instance retrieval research:

1. **Fine-grained Instance Retrieval**: Retrieve specific object instances across diverse conditions
2. **Cross-domain Retrieval**: Train on synthetic (OWID) and evaluate on real (RoboTools)
3. **Few-shot Instance Learning**: Learn instance representations from limited queries
4. **Multi-instance Retrieval**: Handle scenes with multiple target instances
5. **Occlusion-robust Retrieval**: Retrieve partially visible instances
6. **Viewpoint-invariant Retrieval**: Match instances across different viewpoints

## Retrieval-Specific Notes

### OWID for Retrieval Training
- Provides controlled instance variations for learning
- Multiple instances per image for complex retrieval scenarios
- Instance-level segmentation for precise evaluation

### RoboTools for Real-world Retrieval
- Challenging real-world retrieval scenarios
- Industrial/robotic objects with practical applications
- Test-only split for zero-shot retrieval evaluation

### Instance Matching Strategy
- Use instance IDs (`ins` field) for ground truth matching
- Bounding box IoU for spatial retrieval accuracy
- Mask overlap for pixel-level retrieval precision

## Citation

If you use this instance retrieval dataset, please cite:

```bibtex
@inproceedings{green2025findyourneedle,
  author={Green, Michael and Levy, Matan and Tzachor, Issar and Samuel, Dvir and Darshan, Nir and Ben-Ari, Rami},
  title={Find your Needle: Small Object Image Retrieval via Multi-Object Attention Optimization},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}

@dataset{voxdet2024,
  title={VoxDet: Visual Object eXtended Detection Dataset},
  author={VoxDet Contributors},
  year={2024},
  note={Base dataset for instance retrieval tasks}
}
```

## Acknowledgments

This instance retrieval dataset is built upon the VoxDet foundation, extending it specifically for instance-level visual search and retrieval research. We thank the VoxDet contributors for providing the base annotations and structure.
