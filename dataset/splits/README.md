# Frozen dataset membership

Snapshot date: **2026-11-05**, before implementation or training. These lists preserve the supplied Roboflow export membership; no images were moved or reassigned. This snapshot belongs to the initial planning commit on `dev-test01`.

| Split | Filename list | Images | Source annotation file |
| --- | --- | ---: | --- |
| Training | [train.txt](train.txt) | 50 | `dataset/train/_annotations.coco.json` |
| Validation | [valid.txt](valid.txt) | 15 | `dataset/valid/_annotations.coco.json` |
| Test | [test.txt](test.txt) | 7 | `dataset/test/_annotations.coco.json` |
| Total | | **72** | Approximately 70/20/10; preserve actual membership. |

Each text file contains one exact image path per line, relative to `dataset/`, sorted by filename. For example, a line starting `train/` resolves inside `dataset/train/`. `valid.txt` is the validation split; it is not a second training split.

Commit these lists and the planning documents before implementation code. Keep test images and all derivatives out of training, tuning and checkpoint selection. Future conversion or undistortion must retain source membership and record any new processed filenames against these frozen paths. Changes require an explicit dated explanation; do not silently reshuffle the holdout.

## Inventory and checks performed

- Every listed image exists, and each list exactly matches the `images[].file_name` entries in its source COCO JSON.
- There are 72 unique exported filenames and 72 unique raw-source capture identities. No source capture is shared across splits, and no exact-byte duplicate exported images were found. This does not establish that visually similar captures are independent.
- Export names follow `IMG_<date>_<time>_jpg.rf.<hash>.jpg`. Replacing `_jpg.rf.<hash>.jpg` with `.jpg` maps every listed export uniquely to its existing original in `dataset/Raw/`.
- `dataset/Raw/` has 73 images. **`Raw/IMG_20261104_200608.jpg` is unassigned**, excluded from all three lists and unused for training or evaluation. Its exclusion reason is unknown; preserve it and resolve provenance before proposing any use.
- [SHA256SUMS.txt](SHA256SUMS.txt) records SHA-256 hashes for these three lists and the three source COCO JSON files, with paths relative to `dataset/`. It records the initial snapshot, not a hash of every image. Label repairs must be documented against this snapshot.
- The initial planning commit tracks the lists and metadata. Source photographs and COCO JSON files remain local for the later data commit; their recorded hashes allow that commit to be checked against this snapshot. `.gitattributes` preserves LF line endings for the lists and original bytes for COCO JSON files.
- Only filenames, file integrity and annotation metadata were checked for the freeze. Held-out test visuals were not reviewed to tune the planned system.

## Preparation still pending

All exported images are **432 x 432**. The export README reports auto-orientation and stretch resizing. The 73 raw originals contain 49 landscape 4080 x 2296 images and 24 portrait 2296 x 4080 images. Resolve the original/export coordinate transformation before applying calibration to labels; correct images and masks together and preserve membership.

Mask boundaries have not been validated. Some training/validation captures have multiple Calculator annotations, so review for duplicate or incorrect instances during label preparation. No calibration, corrected-label, training, measurement or Git-commit completion is implied by this filename freeze.
