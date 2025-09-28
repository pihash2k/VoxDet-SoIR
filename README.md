# VoxDet Dataset

A unified visual object detection dataset combining synthetic (OWID) and real-world (RoboTools) data for query-based object detection and visual search tasks.

## Overview

VoxDet is designed for few-shot and query-based object detection tasks where the goal is to find instances of query objects within gallery images. The dataset provides both synthetic data with rich annotations (OWID) and real-world robotic scenarios (RoboTools).

## Dataset Components

### 1. OWID (Open World Instance Detection)
- **Type**: Synthetic dataset
- **Content**: Computer-generated images with precise annotations
- **Structure**: Query-gallery pairs with segmentation masks
- **Split**: Training and validation sets

### 2. RoboTools
- **Type**: Real-world dataset  
- **Content**: Images from robotic/tools scenarios
- **Structure**: Test videos and corresponding scene images
- **Split**: Test set only

## Directory Structure

```
VoxDet/
├── OWID/
│   ├── P1/                    # Query images directory
│   │   ├── <category_id>/     # Category-specific folders
│   │   │   ├── rgb/           # RGB query images
│   │   │   └── mask/          # Binary segmentation masks
│   │   └── ...
│   └── P2/                    # Gallery images directory
│       ├── images/            # Gallery RGB images
│       ├── train_annotations.json  # Training annotations (COCO format)
│       └── val_annotations.json    # Validation annotations (COCO format)
│
├── RoboTools/
│   ├── test/                  # Gallery test images
│   │   ├── scene_gt_coco_all.json  # Scene annotations (COCO format)
│   │   └── *.jpg/png          # Test images
│   └── test_video/            # Query video frames
│       ├── <category_name>_<category_id>/
│       │   ├── rgb/           # RGB query frames
│       │   └── mask/          # Binary segmentation masks
│       └── ...
│
└── anns.pt                    # Processed annotations (output)
```

## Annotation Format

### PyTorch Dictionary Structure (`anns.pt`)

The processed annotations are saved as a PyTorch dictionary with the following structure:

```python
{
    "<image_path>": {
        "bbox": [...],        # Bounding boxes
        "mask": [...],        # Segmentation masks (RLE encoded)
        "is_query": bool,     # True if query image, False if gallery
        "is_val": bool,       # True if validation set, False otherwise
        "ins": [...],         # Instance/category IDs
        "set": str,           # Dataset name ("OWID" or "RoboTools")
        "num_ins": [...]      # Number of instances (OWID only)
    },
    ...
}
```

### Field Descriptions

#### For Query Images
- **bbox**: Single bounding box `(x1, y1, x2, y2)` extracted from mask
- **mask**: RLE-encoded binary segmentation mask
- **is_query**: `True`
- **is_val**: `False`
- **ins**: Single category ID
- **set**: Dataset identifier

#### For Gallery Images
- **bbox**: List of bounding boxes `[(x1, y1, x2, y2), ...]` for each instance
- **mask**: List of RLE-encoded masks (OWID) or empty list (RoboTools)
- **is_query**: `False`
- **is_val**: `True` for validation set, `False` for training
- **ins**: List of category IDs `[cat_id1, cat_id2, ...]`
- **set**: Dataset identifier
- **num_ins**: List of instance counts per annotation (OWID only)

### Bounding Box Format
- **Input format (COCO)**: `[x, y, width, height]`
- **Output format**: `[x1, y1, x2, y2]` (top-left and bottom-right corners)

### Mask Encoding
- Binary masks are encoded using Run-Length Encoding (RLE)
- Masks are thresholded to binary (0 or 1) before encoding
- Query masks are extracted from PNG files with transparency

## Usage

### Processing the Dataset

```bash
# Basic usage
python process_voxdet.py --voxdet-root /path/to/VoxDet

# Custom output path
python process_voxdet.py --voxdet-root /path/to/VoxDet --output-file custom_anns.pt

# Process only OWID
python process_voxdet.py --skip-robotools

# Process only RoboTools
python process_voxdet.py --skip-owid

# Debug mode with limited categories
python process_voxdet.py --debug --max-categories 5
```

### Command-line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--voxdet-root` | `/datasets/agamotto/Agamotto-SO/VoxDet` | Root directory of VoxDet dataset |
| `--owid-path` | `OWID` | Relative path to OWID dataset |
| `--robotools-path` | `RoboTools` | Relative path to RoboTools dataset |
| `--output-file` | `anns.pt` | Output annotation file path |
| `--process-owid` | `True` | Process OWID dataset |
| `--process-robotools` | `True` | Process RoboTools dataset |
| `--skip-owid` | `False` | Skip OWID processing |
| `--skip-robotools` | `False` | Skip RoboTools processing |
| `--debug` | `False` | Enable debug mode with verbose output |
| `--max-categories` | `None` | Process only first N categories (debugging) |

### Loading Annotations in Python

```python
import torch

# Load annotations
anns = torch.load('anns.pt')

# Access specific image annotations
image_path = 'path/to/image.jpg'
if str(image_path) in anns:
    ann = anns[str(image_path)]
    
    # Check if query or gallery image
    if ann['is_query']:
        print(f"Query image for category {ann['ins']}")
        print(f"Bounding box: {ann['bbox']}")
    else:
        print(f"Gallery image with {len(ann['ins'])} instances")
        for i, (bbox, cat_id) in enumerate(zip(ann['bbox'], ann['ins'])):
            print(f"  Instance {i}: Category {cat_id}, BBox: {bbox}")

# Filter by dataset
owid_anns = {k: v for k, v in anns.items() if v['set'] == 'OWID'}
robotools_anns = {k: v for k, v in anns.items() if v['set'] == 'RoboTools'}

# Get all query images
query_images = {k: v for k, v in anns.items() if v['is_query']}

# Get validation set
val_images = {k: v for k, v in anns.items() if v['is_val']}
```

## Dataset Statistics

After processing, you can analyze the dataset:

```python
import torch
from collections import Counter

anns = torch.load('anns.pt')

# Basic statistics
total_images = len(anns)
query_images = sum(1 for v in anns.values() if v['is_query'])
gallery_images = total_images - query_images

# Dataset distribution
dataset_dist = Counter(v['set'] for v in anns.values())

# Category distribution
all_categories = []
for ann in anns.values():
    if ann['is_query']:
        all_categories.append(ann['ins'])
    else:
        all_categories.extend(ann['ins'])
category_dist = Counter(all_categories)

print(f"Total images: {total_images}")
print(f"Query images: {query_images}")
print(f"Gallery images: {gallery_images}")
print(f"Dataset distribution: {dict(dataset_dist)}")
print(f"Number of unique categories: {len(category_dist)}")
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

# Download OWID and RoboTools datasets to appropriate directories
# Ensure directory structure matches the expected format

# Process the dataset
python process_voxdet.py --voxdet-root /path/to/VoxDet
```

## Data Format Notes

### OWID Annotations
- Uses standard COCO format with additional `num_ins` field
- Includes segmentation masks in polygon format
- Separated into training and validation sets

### RoboTools Annotations
- COCO format for bounding boxes
- Query masks provided as separate PNG files
- Test set only (no training/validation split)

### Mask Processing
- Binary masks are extracted from PNG alpha channels
- Automatic thresholding applied for multi-value masks
- Bounding boxes computed from mask regions

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{green2025findyourneedle,
  author={Green, Michael and Levy, Matan and Tzachor, Issar and Samuel, Dvir and Darshan, Nir and Ben-Ari, Rami},
  title={Find your Needle: Small Object Image Retrieval via Multi-Object Attention Optimization},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025},
}

@dataset{voxdet2024,
  title={VoxDet: A Unified Dataset for Query-based Object Detection},
  author={Your Name},
  year={2024},
  publisher={Your Institution}
}
```
