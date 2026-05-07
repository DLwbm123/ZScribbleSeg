import os
import re
import numpy as np
from pathlib import Path
from torch.utils import data
import nibabel as nib

import data.transforms as T


def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header


class btcvSeg(data.Dataset):
    """BTCV 2D slice dataset.

    - Loads 3D volumes from NIfTI files.
    - Assumes arrays are shaped (X, Y, Z) where Z is the last axis.
    - Generates examples as axial slices along Z.
    - Pairs images with labels named like `labelXXXX` (digits extracted from image stem).
    """

    def __init__(self, img_folder, lab_folder, lab_values, transforms):
        self._transforms = transforms
        self.lab_values = lab_values

        img_paths = sorted(list(img_folder.iterdir()))
        lab_paths = sorted(list(lab_folder.iterdir()))

        self.examples = []
        self.img_dict = {}
        self.lab_dict = {}

        # Index labels by numeric id extracted from filename (e.g., label0074 -> 0074)
        lab_by_id = {}
        for lab_path in lab_paths:
            lab_stem = lab_path.stem
            # handle .nii.gz stems (Path.stem strips only last suffix; label0074.nii.gz -> label0074.nii)
            if lab_stem.endswith('.nii'):
                lab_stem = Path(lab_stem).stem
            digits = ''.join(ch for ch in lab_stem if ch.isdigit())
            if digits != '':
                lab_by_id[digits] = lab_path

        for img_path in img_paths:
            img_stem = img_path.stem
            if img_stem.endswith('.nii'):
                img_stem = Path(img_stem).stem
            digits = ''.join(ch for ch in img_stem if ch.isdigit())
            if digits == '' or digits not in lab_by_id:
                # Skip if no matching label
                continue

            lab_path = lab_by_id[digits]

            img = self.read_image(str(img_path))
            lab = self.read_label(str(lab_path))

            img_name = img_path.stem
            lab_name = Path(lab_path).stem

            # Normalize stems for .nii.gz
            if img_name.endswith('.nii'):
                img_name = Path(img_name).stem
            if lab_name.endswith('.nii'):
                lab_name = Path(lab_name).stem

            self.img_dict[img_name] = img
            self.lab_dict[lab_name] = lab

            # Sanity: Z last axis
            assert img[0].shape[2] == lab[0].shape[2], f"Z mismatch: {img[0].shape} vs {lab[0].shape}"

            for z in range(img[0].shape[2]):
                self.examples.append((img_name, lab_name, z))

    def __getitem__(self, idx):
        img_name, lab_name, z = self.examples[idx]

        vol_img, scale_vector_img = self.img_dict[img_name]
        vol_lab, scale_vector_lab = self.lab_dict[lab_name]

        # Axial slice along Z (last axis)
        img2d = vol_img[:, :, z]
        lab2d = vol_lab[:, :, z]

        img2d = np.expand_dims(img2d, 0)
        lab2d = np.expand_dims(lab2d, 0)

        target = {
            'name': lab_name,
            'slice': z,
            'masks': lab2d.astype(np.int64),
            'orig_size': lab2d.shape
        }

        if self._transforms is not None:
            img2d, target = self._transforms([img2d, scale_vector_img], [target, scale_vector_lab])

        return img2d, target

    def __len__(self):
        return len(self.examples)

    def read_image(self, img_path):
        img_dat = load_nii(img_path)
        img = img_dat[0]
        pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])

        # Keep the same target resolution used elsewhere in this repo; Rescale() will use this vector.
        target_resolution = (1.36719, 1.36719)
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])

        img = img.astype(np.float32)
        # Per-volume normalization
        mean = float(img.mean())
        std = float(img.std()) if float(img.std()) > 0 else 1.0
        img = (img - mean) / std
        return [img, scale_vector]

    def read_label(self, lab_path):
        lab_dat = load_nii(lab_path)
        lab = lab_dat[0]
        pixel_size = (lab_dat[2].structarr['pixdim'][1], lab_dat[2].structarr['pixdim'][2])

        target_resolution = (1.36719, 1.36719)
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])

        return [lab, scale_vector]


def make_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize()
    ])

    if image_set == 'train':
        return T.Compose([
            T.Rescale(),
            T.RandomHorizontalFlip(),
            T.RandomRotate((0, 360)),
            T.PadOrCropToSize([212, 212]),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.Rescale(),
            T.PadOrCropToSize([212, 212]),
            normalize
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path('/root/ZScribble/data/' + args.dataset)

    PATHS = {
        'train': (root / 'train' / 'images', root / 'train' / 'labels'),
        'val': (root / 'val' / 'images', root / 'val' / 'labels')
    }

    img_folder, lab_folder = PATHS[image_set]
    dataset_dict = {}

    for task, value in args.tasks.items():
        lab_values = value.get('lab_values', [])
        dataset = btcvSeg(img_folder, lab_folder, lab_values, transforms=make_transforms(image_set))
        dataset_dict[task] = dataset

    return dataset_dict


if __name__ == '__main__':
    import argparse
    import torch
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BTCV')
    parser.add_argument('--task', type=str, default='CT')
    parser.add_argument('--index', type=int, default=0, help='slice example index to visualize/inspect')
    parser.add_argument('--num_probe', type=int, default=20, help='how many random samples to probe for crop size')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--viz', type=bool, default=True, help='visualize one raw vs transformed sample')
    args_cli = parser.parse_args()

    # Your tasks definition
    tasks = {
        # 'MR': {'lab_values': [0, 200, 500, 600], 'out_channels': 4}
        'CT': {'lab_values': [0, 1, 2, 3, 4, 5, 6]}
    }

    class _Args:
        def __init__(self, dataset, tasks):
            self.dataset = dataset
            self.tasks = tasks

    args = _Args(args_cli.dataset, tasks)

    # Build datasets via build() so we see the real train transforms
    ds_dict_train = build('train', args)
    if args_cli.task not in ds_dict_train:
        raise KeyError(f"Task {args_cli.task} not found. Available: {list(ds_dict_train.keys())}")

    ds_train = ds_dict_train[args_cli.task]

    # Raw (no-transform) dataset for original sizes
    root = Path('/root/ZScribble/data/' + args.dataset)
    img_folder = root / 'train' / 'images'
    lab_folder = root / 'train' / 'labels'
    ds_raw = btcvSeg(img_folder, lab_folder, lab_values=args.tasks[args_cli.task]['lab_values'], transforms=None)

    print(f"[build test] dataset={args.dataset} task={args_cli.task}")
    print(f"  raw slices: {len(ds_raw)} | train(transformed) slices: {len(ds_train)}")

    # Probe output sizes after train transforms
    rng = np.random.default_rng(0)
    probe_n = min(int(args_cli.num_probe), len(ds_train))
    probe_indices = rng.choice(len(ds_train), size=probe_n, replace=False).tolist() if probe_n > 0 else []

    sizes = []
    for idx in probe_indices:
        img_tr, _ = ds_train[int(idx)]
        if isinstance(img_tr, torch.Tensor):
            h, w = int(img_tr.shape[-2]), int(img_tr.shape[-1])
        else:
            h, w = int(img_tr.shape[-2]), int(img_tr.shape[-1])
        sizes.append((h, w))

    if len(sizes) > 0:
        uniq_sizes, counts = np.unique(np.array(sizes), axis=0, return_counts=True)
        print("\n[train-time output sizes distribution]")
        for (h, w), c in zip(uniq_sizes.tolist(), counts.tolist()):
            print(f"  ({h}, {w}) : {c}")

        top_i = int(np.argmax(counts))
        top_h, top_w = uniq_sizes[top_i].tolist()
        print(f"\n[train-time crop/pad result] most common effective size = ({top_h}, {top_w})")
    else:
        print("\n[train-time output sizes distribution] (empty)")

    # Inspect one specific index
    idx = int(args_cli.index)
    idx = max(0, min(idx, len(ds_train) - 1))

    img_raw, tgt_raw = ds_raw[idx]
    img_tr, tgt_tr = ds_train[idx]

    raw_img = img_raw[0]
    raw_mask = tgt_raw['masks'][0]

    tr_img = img_tr[0].detach().cpu().numpy() if isinstance(img_tr, torch.Tensor) else img_tr[0]
    tr_mask_t = tgt_tr['masks']
    tr_mask = tr_mask_t[0].detach().cpu().numpy() if isinstance(tr_mask_t, torch.Tensor) else tr_mask_t[0]

    print("\n[single-slice shapes]")
    print("  raw image:", raw_img.shape, "raw label:", raw_mask.shape)
    print("  train image:", tr_img.shape, "train label:", tr_mask.shape)

    print("\n[label unique values]")
    print("  raw unique (<=50):", np.unique(raw_mask)[:50])
    print("  train unique (<=50):", np.unique(tr_mask)[:50])

    if args_cli.viz:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        axes[0, 0].imshow(raw_img.T, cmap='gray', origin='lower')
        axes[0, 0].set_title(f'RAW Image (H,W)={raw_img.shape}')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(raw_mask.T, cmap='tab20', origin='lower', vmin=0)
        axes[0, 1].set_title('RAW Label (colored)')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(tr_img.T, cmap='gray', origin='lower')
        axes[1, 0].set_title(f'TRAIN Image after transforms (H,W)={tr_img.shape}')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(tr_mask.T, cmap='tab20', origin='lower', vmin=0)
        axes[1, 1].set_title('TRAIN Label after transforms (colored)')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

    # One batch sanity check
    dl = DataLoader(ds_train, batch_size=args_cli.batch_size, shuffle=True,
                    num_workers=args_cli.num_workers, pin_memory=True)
    imgs, targets = next(iter(dl))
    masks = targets['masks']

    print("\n[one train batch]")
    print('  imgs:', tuple(imgs.shape), imgs.dtype, 'min/max:', float(imgs.min()), float(imgs.max()))
    print('  masks:', tuple(masks.shape), masks.dtype)
    print('  first sample name:', targets['name'][0], 'slice:', int(targets['slice'][0]))