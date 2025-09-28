import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import argparse
from RLE import encode_rle


def convert_xyhw_to_xyxy(bbox):
    """
    Convert bounding box from (x1, y1, w, h) format to (x1, y1, x2, y2) format.

    Args:
    bbox (tuple or list): Bounding box in (x1, y1, w, h) format

    Returns:
    tuple: Bounding box in (x1, y1, x2, y2) format
    """
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    return x1, y1, x2, y2


def get_bbox_from_mask(mask):
    """Get bounding box from binary mask."""
    if mask.any():
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        bbox = (x_min, y_min, x_max, y_max)
    else:
        bbox = (0, 0, mask.shape[1], mask.shape[0])
    return bbox


def add_gall_anns_real(
    anns, im_path, imgs_bboxes, is_query, is_val, ind, set, cat_id
):
    """Add gallery annotations for RoboTools dataset."""
    if str(im_path) not in anns:
        anns[str(im_path)] = {}  # FIX: Initialize dict first
        anns[str(im_path)]["bbox"] = []
        anns[str(im_path)]["num_ins"] = []
        anns[str(im_path)]["mask"] = []
        anns[str(im_path)]["is_query"] = is_query
        anns[str(im_path)]["is_val"] = is_val
        anns[str(im_path)]["ins"] = []
        anns[str(im_path)]["set"] = set
    anns[str(im_path)]["bbox"].append(imgs_bboxes[ind])
    anns[str(im_path)]["ins"].append(cat_id)


def add_query_anns(anns, im_path, cat_id, set, masks_path):
    """Add query annotations."""
    mask_path = masks_path.joinpath(im_path.name)

    # Check if mask exists
    if not mask_path.exists():
        print(f"Warning: Mask not found at {mask_path}, skipping...")
        return

    try:
        mask = Image.open(mask_path)
        mask = np.array(mask)[..., 0]
        if mask.max() > 1:
            th = np.median(np.unique(mask))
            mask[mask < th] = 0
            mask[mask >= th] = 1
        bbox = get_bbox_from_mask(mask)
        encoded = encode_rle(mask)

        anns[str(im_path)] = {}  # FIX: Initialize dict
        anns[str(im_path)]["bbox"] = bbox
        anns[str(im_path)]["mask"] = encoded
        anns[str(im_path)]["is_query"] = True
        anns[str(im_path)]["is_val"] = False
        anns[str(im_path)]["ins"] = cat_id
        anns[str(im_path)]["set"] = set
    except Exception as e:
        print(f"Error processing mask {mask_path}: {e}")


def add_gall_anns(
    anns,
    im_path,
    imgs_bboxes,
    imgs_segs,
    imgs_num_ins,
    is_query,
    is_val,
    ind,
    set,
    cat_id,
):
    """Add gallery annotations for OWID dataset."""
    if str(im_path) not in anns:
        anns[str(im_path)] = {}  # FIX: Initialize dict first
        anns[str(im_path)]["bbox"] = []
        anns[str(im_path)]["num_ins"] = []
        anns[str(im_path)]["mask"] = []
        anns[str(im_path)]["is_query"] = is_query
        anns[str(im_path)]["is_val"] = is_val
        anns[str(im_path)]["ins"] = []
        anns[str(im_path)]["set"] = set
    anns[str(im_path)]["bbox"].append(imgs_bboxes[ind])
    anns[str(im_path)]["ins"].append(cat_id)
    seg = imgs_segs[ind]
    anns[str(im_path)]["mask"].append(seg)
    anns[str(im_path)]["num_ins"].append(imgs_num_ins[ind])


def get_image_info_by_id(images_list, image_id):
    """
    Helper function to get image info by ID.
    Handles both list and dict formats for images.
    """
    # If images_list is a dict with image_id as keys
    if isinstance(images_list, dict):
        return images_list.get(image_id, None)

    # If images_list is a list of image objects
    for img in images_list:
        if img.get("id") == image_id:
            return img

    # If image_id is actually an index (for backwards compatibility)
    try:
        if isinstance(image_id, int) and 0 <= image_id < len(images_list):
            return images_list[image_id]
    except:
        pass

    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process VoxDet dataset annotations"
    )

    # Data paths
    parser.add_argument(
        "--voxdet-root",
        type=str,
        default="/datasets/agamotto/Agamotto-SO/VoxDet",
        help="Root directory of VoxDet dataset",
    )
    parser.add_argument(
        "--owid-path",
        type=str,
        default="OWID",
        help="Relative path to OWID dataset from voxdet-root",
    )
    parser.add_argument(
        "--robotools-path",
        type=str,
        default="RoboTools",
        help="Relative path to RoboTools dataset from voxdet-root",
    )

    # Output
    parser.add_argument(
        "--output-file",
        type=str,
        default="anns.pt",
        help="Output annotation file path",
    )

    # Dataset selection
    parser.add_argument(
        "--process-owid",
        action="store_true",
        default=True,
        help="Process OWID dataset",
    )
    parser.add_argument(
        "--process-robotools",
        action="store_true",
        default=True,
        help="Process RoboTools dataset",
    )
    parser.add_argument(
        "--skip-owid", action="store_true", help="Skip OWID dataset processing"
    )
    parser.add_argument(
        "--skip-robotools",
        action="store_true",
        help="Skip RoboTools dataset processing",
    )

    # Debug options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose output",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=None,
        help="Process only first N categories (for debugging)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set up paths
    voxdet_root = Path(args.voxdet_root)
    output_file = Path(args.output_file)

    # Initialize annotations dictionary
    anns = defaultdict(dict)

    # Process OWID Dataset
    if args.process_owid and not args.skip_owid:
        base_path_syn = voxdet_root / args.owid_path
        gallery_path_syn = base_path_syn / "P2" / "images"
        query_path_syn = base_path_syn / "P1"
        train_anns_path_syn = base_path_syn / "P2" / "train_annotations.json"
        val_anns_path_syn = base_path_syn / "P2" / "val_annotations.json"

        # Check paths exist
        if not train_anns_path_syn.exists():
            print(
                f"Error: OWID train annotations not found at {train_anns_path_syn}"
            )
            if not args.debug:
                return
        else:
            print(f"Loading OWID annotations from {base_path_syn}")

            with open(train_anns_path_syn, "r") as f:
                train_anns_syn = json.load(f)
            with open(val_anns_path_syn, "r") as f:
                val_anns_syn = json.load(f)

            query_dirs = [q for q in query_path_syn.iterdir() if q.is_dir()]

            # Limit categories for debugging
            if args.max_categories:
                query_dirs = query_dirs[: args.max_categories]

            print(f"Processing {len(query_dirs)} OWID categories...")
            loader = tqdm(query_dirs, desc="OWID")

            for cat_id_dir in loader:
                loader.set_description(f"OWID class: {cat_id_dir.name}")
                cat_id = int(cat_id_dir.name)

                rgb_dir = cat_id_dir / "rgb"
                if not rgb_dir.exists():
                    print(f"Warning: RGB directory not found at {rgb_dir}")
                    continue

                query_imgs = list(rgb_dir.iterdir())

                # Get indices for this category
                gall_inds_train = [
                    idx
                    for idx, ann in enumerate(train_anns_syn["annotations"])
                    if ann["category_id"] == cat_id
                ]
                gall_inds_val = [
                    idx
                    for idx, ann in enumerate(val_anns_syn["annotations"])
                    if ann["category_id"] == cat_id
                ]

                # Process training gallery
                for ind in gall_inds_train:
                    ann = train_anns_syn["annotations"][ind]
                    img_info = get_image_info_by_id(
                        train_anns_syn["images"], ann["image_id"]
                    )

                    if img_info:
                        img_path = (
                            gallery_path_syn / Path(img_info["file_name"]).name
                        )
                        bbox = convert_xyhw_to_xyxy(ann["bbox"])
                        num_ins = ann.get("num_ins", 1)
                        seg = ann.get("segmentation", [])

                        add_gall_anns(
                            anns,
                            img_path,
                            [bbox],
                            [seg],
                            [num_ins],
                            False,
                            False,
                            0,
                            "OWID",
                            cat_id,
                        )

                # Process validation gallery
                for ind in gall_inds_val:
                    ann = val_anns_syn["annotations"][ind]
                    img_info = get_image_info_by_id(
                        val_anns_syn["images"], ann["image_id"]
                    )

                    if img_info:
                        img_path = (
                            gallery_path_syn / Path(img_info["file_name"]).name
                        )
                        bbox = convert_xyhw_to_xyxy(ann["bbox"])
                        num_ins = ann.get("num_ins", 1)
                        seg = ann.get("segmentation", [])

                        add_gall_anns(
                            anns,
                            img_path,
                            [bbox],
                            [seg],
                            [num_ins],
                            False,
                            True,
                            0,
                            "OWID",
                            cat_id,
                        )

                # Process query images
                for img in query_imgs:
                    if img.suffix in [".jpg", ".png"]:
                        add_query_anns(
                            anns, img, cat_id, "OWID", cat_id_dir / "mask"
                        )

    # Process RoboTools Dataset
    if args.process_robotools and not args.skip_robotools:
        base_path_real = voxdet_root / args.robotools_path
        gallery_path_real = base_path_real / "test"
        query_path_real = base_path_real / "test_video"
        real_anns_path = gallery_path_real / "scene_gt_coco_all.json"

        if not real_anns_path.exists():
            print(f"Error: RoboTools annotations not found at {real_anns_path}")
            if not args.debug:
                return
        else:
            print(f"Loading RoboTools annotations from {base_path_real}")

            with open(real_anns_path, "r") as f:
                real_anns = json.load(f)

            query_dirs_real = [
                x for x in query_path_real.iterdir() if x.is_dir()
            ]

            # Limit categories for debugging
            if args.max_categories:
                query_dirs_real = query_dirs_real[: args.max_categories]

            print(f"Processing {len(query_dirs_real)} RoboTools categories...")

            for real_query_im_path in tqdm(query_dirs_real, desc="RoboTools"):
                cat_id = int(real_query_im_path.name.split("_")[-1])

                # Get gallery indices for this category
                gall_real_inds = [
                    idx
                    for idx, ann in enumerate(real_anns["annotations"])
                    if ann["category_id"] == cat_id
                ]

                # Process gallery images
                for ind in gall_real_inds:
                    ann = real_anns["annotations"][ind]
                    img_info = get_image_info_by_id(
                        real_anns["images"], ann["image_id"]
                    )

                    if img_info:
                        img_path = gallery_path_real / img_info["file_name"]
                        bbox = convert_xyhw_to_xyxy(ann["bbox"])

                        add_gall_anns_real(
                            anns,
                            img_path,
                            [bbox],
                            False,
                            False,
                            0,
                            "RoboTools",
                            cat_id,
                        )

                # Process query images
                rgb_path_query = real_query_im_path / "rgb"
                if rgb_path_query.exists():
                    for im_path in rgb_path_query.iterdir():
                        if im_path.suffix in [".jpg", ".png"]:
                            add_query_anns(
                                anns,
                                im_path,
                                cat_id,
                                "RoboTools",
                                real_query_im_path / "mask",
                            )

    # Save annotations
    print(f"\nSaving annotations to {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(anns), output_file)
    print(f"Successfully saved {len(anns)} annotations")

    if args.debug:
        print(f"\nSample annotation keys: {list(anns.keys())[:5]}")
        if anns:
            first_key = list(anns.keys())[0]
            print(f"Sample annotation structure: {anns[first_key].keys()}")


if __name__ == "__main__":
    main()
