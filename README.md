# VoxDet-SoIR: Small Object Instance Retrieval Dataset

A comprehensive instance retrieval dataset built upon VoxDet, focusing on small object retrieval in cluttered scenes with both synthetic (OWID) and real-world (RoboTools) data.

## Overview

VoxDet-SoIR (Small object Instance Retrieval) extends VoxDet for **instance retrieval** applications, where the primary task is to retrieve all occurrences of a query object instance from a large gallery of images. Unlike traditional object detection that focuses on category-level detection, this dataset emphasizes **instance-level retrieval** - finding the exact same object or highly similar instances across different viewpoints, lighting conditions, and contexts.

### Key Features
- **Small object focus**: Average object size of 1.1% of image area
- **Dense annotations**: Multiple objects per image (avg. 5.8 annotated objects)
- **Instance-level annotations** for precise object retrieval
- **Query-gallery paradigm** optimized for retrieval benchmarking  
- **Cross-domain evaluation** with synthetic-to-real transfer capabilities

## Dataset Comparison

VoxDet-SoIR stands out among instance retrieval datasets with its focus on small objects and multiple annotations. With an average object size of just 1.1% of the image area—significantly smaller than existing benchmarks—and multiple annotated objects per image, VoxDet-SoIR addresses the challenging yet practical scenario of retrieving small instances from cluttered scenes.

| Dataset | #Obj Annot. | #Obj OVD | Obj. Size (in %) |
|---------|-------------|----------|------------------|
| **VoxDet-SoIR** | **5.8** | **14.7** | **1.1** |
| PerMiR | 4.7 | 10.4 | 13.3 |
| INSTRE-XS | 1 | 1.8 | 6.6 |
| INSTRE-XXS | 1 | 1.9 | 2.2 |
| INSTRE (S1) | 1 | 1.8 | 21.0 |
| Products-10K | 1 | 2.1 | 27.1 |
| ℛOxford | 1 | 5.9 | 37.6 |
| ℛParis6K | 1 | 4.9 | 41.4 |

*Note: #Obj Annot. denotes the average number of manually annotated objects per image, #Obj OVD indicates the average number of objects detected using an open vocabulary detector, and Obj. Size represents the mean object size as a percentage of the total image area.*

## Download

### VoxDet (Base Dataset)
Download the base VoxDet dataset from the official repository:
- **GitHub Repository**: [https://github.com/Jaraxxus-Me/VoxDet](https://github.com/Jaraxxus-Me/VoxDet)
- Follow the download instructions in the repository to obtain both OWID and RoboTools datasets

### Additional Datasets for Comparison

#### INSTRE Subsets (INSTRE-XS and INSTRE-XXS)
For small object retrieval experiments:
1. **Download INSTRE dataset**: [http://123.57.42.89/instre/home.html](http://123.57.42.89/instre/home.html)
2. **Generate subsets**: Run the parsing script to create INSTRE-XS and INSTRE-XXS variants:
   ```bash
   python create_instre_subsets.py --instre-root /path/to/INSTRE
   ```

#### PerMiR Dataset
For comparison with person multi-instance retrieval:
- **GitHub Repository**: [https://github.com/dvirsamuel/PDM](https://github.com/dvirsamuel/PDM)
- Download following the repository instructions

## Dataset Components

### 1. OWID (Open World Instance Detection)
- **Purpose**: Training and validation for instance retrieval models
- **Type**: Synthetic dataset with controlled variations
- **Content**: Computer-generated images with precise instance annotations
- **Retrieval Setup**: Query-gallery pairs with ground truth matches
- **Split**: Training and validation sets

### 2. RoboTools  
- **Purpose**: Real-world instance retrieval evaluation
- **Type**: Real-world dataset from robotic/industrial scenarios
- **Content**: Challenging real-world instances with occlusions and viewpoint changes
- **Retrieval Setup**: Video frame queries to retrieve from test scenes
- **Split**: Test set only for zero-shot evaluation

## Directory Structure

```
VoxDet-SoIR/
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

## Annotation Format

### PyTorch Dictionary Structure (`anns.pt`)

```python
{
    "<image_path>": {
        "bbox": [...],        # Instance bounding boxes
        "mask": [...],        # Instance segmentation masks (RLE)
        "is_query": bool,     # Query instance flag
        "ins": [...],         # Instance IDs for retrieval matching
        "set": str,           # Dataset source ("OWID" or "RoboTools")
        "num_ins": [...]      # Instance counts (OWID gallery only)
    },
    ...
}
```

### Field Descriptions

#### Query Instances (`is_query: True`)
- **bbox**: Single bounding box `(x1, y1, x2, y2)`
- **mask**: RLE-encoded instance segmentation
- **ins**: Single instance ID for matching
- **set**: Dataset source identifier

#### Gallery Images (`is_query: False`)
- **bbox**: List of bounding boxes for all instances
- **mask**: List of RLE-encoded masks (OWID) or empty (RoboTools)
- **ins**: List of instance IDs present
- **num_ins**: Instance counts per annotation (OWID only)

## Usage

### 1. Processing the Dataset

```bash
# Process full dataset
python create_dataset.py --voxdet-root /path/to/VoxDet

# Custom output
python create_dataset.py --voxdet-root /path/to/VoxDet --output-file anns.pt

# Process only synthetic data
python create_dataset.py --skip-robotools

# Process only real-world data
python create_dataset.py --skip-owid
```

### 2. Verifying Dataset Integrity

```bash
# Basic verification
python verify_dataset.py --ann-file anns.pt

# With custom base path
python verify_dataset.py --ann-file anns.pt --base-path /path/to/VoxDet

# Verbose mode with JSON report
python verify_dataset.py --ann-file anns.pt --verbose
```

The verification script checks:
- Path existence
- Annotation structure completeness
- Data consistency
- Retrieval statistics
- Mask format validation

### 3. Loading and Using Annotations

```python
import torch
from collections import defaultdict

# Load annotations
anns = torch.load('anns.pt')

# Separate queries and gallery
queries = {k: v for k, v in anns.items() if v['is_query']}
gallery = {k: v for k, v in anns.items() if not v['is_query']}

# Analyze by dataset
synthetic_queries = {k: v for k, v in queries.items() if v['set'] == 'OWID'}
real_queries = {k: v for k, v in queries.items() if v['set'] == 'RoboTools'}

print(f"Total queries: {len(queries)}")
print(f"Gallery size: {len(gallery)}")
print(f"Synthetic queries: {len(synthetic_queries)}")
print(f"Real-world queries: {len(real_queries)}")
```

### 4. Computing Retrieval Statistics

```python
# Instance frequency analysis
instance_frequency = defaultdict(int)
for ann in gallery.values():
    for ins_id in ann['ins']:
        instance_frequency[ins_id] += 1

# Difficulty distribution
easy = sum(1 for f in instance_frequency.values() if f > 10)
medium = sum(1 for f in instance_frequency.values() if 5 <= f <= 10)
hard = sum(1 for f in instance_frequency.values() if f < 5)

print(f"Easy instances (>10 occurrences): {easy}")
print(f"Medium instances (5-10): {medium}")
print(f"Hard instances (<5): {hard}")
```

## Evaluation Metrics

Standard instance retrieval metrics:
- **mAP** (mean Average Precision)
- **Recall@K** (K=1, 5, 10, 50, 100)
- **Precision@K**
- **Instance-level IoU** for localization accuracy
- **Ranking quality measures**

## Requirements

```bash
pip install torch pillow numpy tqdm
```

- Python 3.7+
- PyTorch
- PIL (Pillow)
- NumPy
- tqdm

## Research Applications

1. **Small Object Retrieval**: Specialized for objects <2% of image area
2. **Cross-domain Transfer**: Train on synthetic, test on real
3. **Dense Scene Understanding**: Multiple instances per image
4. **Few-shot Instance Learning**: Limited query examples
5. **Occlusion-robust Retrieval**: Partially visible objects
6. **Viewpoint-invariant Matching**: Diverse viewing angles

## Citation

```bibtex
@inproceedings{green2025findyourneedle,
  author={Green, Michael and Levy, Matan and Tzachor, Issar and Samuel, Dvir and Darshan, Nir and Ben-Ari, Rami},
  title={Find your Needle: Small Object Image Retrieval via Multi-Object Attention Optimization},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

## Acknowledgments

This instance retrieval dataset builds upon the VoxDet foundation, extending it specifically for small object instance-level visual search and retrieval research. We thank the VoxDet contributors for providing the base annotations and structure.
