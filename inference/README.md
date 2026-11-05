# Original-image calculator measurement

The reference is the user's **flat, unmarked paper square, 50 x 50 mm**, measured with a millimetre ruler and confirmed on 2026-11-05. The default [configuration](config.json) records that measurement and detects its actual outer boundary.

From the repository root, using the existing environment and local frozen checkpoint:

```powershell
.venv\Scripts\python.exe -m inference.measure path/to/new_original.jpg --output inference/runs/my_photo
```

Use a fresh output directory. The reference dimensions are already configured as 50 x 50 mm; if a replacement square has different physically measured equal edges, explicitly pass both `--reference-width-mm` and `--reference-height-mm`, or update their values and measurement source in the configuration. `--device cpu` is an explicit alternative; CUDA is the default. No normal-inference option skips undistortion. Success exits 0 and handled failures return 2 from Python.

## Image and coordinate contract

Input must be an uncropped original from the calibrated Redmi Note 12 with the same lens, focus and zoom. The shared [camera reader/undistorter](../calibration/camera.py) rejects EXIF orientation other than 1 and incompatible stored resolution. Configured portrait **2296 x 4080** inputs use rotation 0; **4080 x 2296** inputs use the user's previously confirmed **90-degree clockwise** normalization. The normalized array must match the camera profile before mandatory undistortion. Other sizes, including 432 x 432 stretched exports, are refused. If EXIF camera model is present it must match the configured model. Shape/metadata cannot prove the absence of a same-size crop or changed focus; keeping capture conditions fixed remains necessary.

Undistortion retains **P=K**, **2296 x 4080**, the full frame and black invalid borders. Segmentation uniformly resizes to short side 512/long side at most 960 (512 x 910 here), matching the frozen training configuration. Mask probability and calculator score cutoffs remain **0.5**. The highest-scoring calculator instance is selected; predicted boxes are not used for dimensions. Its thresholded mask is restored to the corrected full frame using exact nearest-neighbour interpolation. All external mask components contribute contour points; holes do not affect the enclosing rectangle. Mask/source boundaries overlapping invalid undistortion pixels are refused.

The paper detector searches a bounded preview of the corrected image for bright, uniform, convex four-sided boundaries with approximately equal sides and near-right angles. Fixed brightness, contrast, geometry and area checks are recorded in the configuration. Candidate proposals combine Canny edges with fixed bright-intensity regions, following the general [OpenCV square-detector approach](https://github.com/opencv/opencv/blob/4.10.0/samples/cpp/squares.cpp). This handles textured backgrounds whose edges join or break the paper outline. Overlapping proposals of one boundary are merged using interior uniformity and boundary contrast, with area only breaking a quality tie; multiple distinct candidates are refused as ambiguous. Pixel-centre coordinates are restored from the preview before `cornerSubPix` refines corners on the **full corrected image**. Ordered corners are clockwise TL, TR, BR, BL. Because the paper is an unmarked square with equal edges, cyclic rotations do not change the physical scale. A single plausible white quadrilateral is a geometric hypothesis, not semantic proof of its identity; use a clearly isolated paper square on a contrasting, uncluttered surface. Dark, occluded, strongly tilted or ambiguous references can be refused.

## One conversion function

[measurement.planar.measure_mask_contour_mm](../measurement/planar.py#L11) is the single pixel-to-mm conversion:

1. Map the ordered outer paper corners to `(0,0), (50,0), (50,50), (0,50)` mm using a four-point homography.
2. Transform the **actual predicted mask contour points** into that millimetre plane.
3. Fit `minAreaRect` to those **metric points**, define **width as the shorter side** and **height as the longer side**, and inverse-project the metric rectangle only to draw it on the corrected image.

The code follows the [OpenCV perspective-transform API](https://docs.opencv.org/4.10.0/da/d54/group__imgproc__transform.html) and [contour/rectangle API](https://docs.opencv.org/4.10.0/d3/dc0/group__imgproc__shape.html). It does not scale a pixel bounding box. Degenerate corners/contours, nonfinite transforms and crossings of the planar horizon are refused. The paper and measured calculator outline **must be coplanar**; calculator thickness, paper curvature and plane offset can bias dimensions. Place the paper on a rigid support at the outline's height when needed. Coplanarity is a physical scene condition that the program cannot establish from four corners alone.

## Outputs and failures

Runtime photographs, corrected images, masks, overlays and diagnostic files remain local under the ignored `inference/runs/` and `inference/diagnostics/` folders. They are excluded from this code push. Paths labelled "local output" describe files on the development machine; run the CLI on your own input to generate them. Published verification summaries retain results and hashes, and independent physical accuracy remains unverified.

Each fresh directory saves `result.json`, effective configuration, command, source hashes and environment. Once valid undistortion succeeds it also saves full-frame `corrected.jpg` and `overlay.jpg`; a detected calculator saves binary `mask.png`. A successful conversion additionally saves `geometry.json`, containing the actual corrected-pixel contours, metric contour, homography and metric/image rectangle corners. JPEGs are display evidence; the binary mask is lossless.

`result.json` includes `status`, `width_mm`, `height_mm`, `detection_confidence`, reference corners/settings/physical measurements, input/calibration hashes, frozen checkpoint SHA-256 and epoch, training/inference configuration IDs, coordinate system, output paths and `failure_reason`. Confidence is the model's detection score, **not a guarantee of dimensional accuracy**. Every failed result has `width_mm=null` and `height_mm=null`.

- `object_not_found`: no nonempty calculator mask at the frozen score cutoff.
- `reference_not_found`: no qualifying outer square, or multiple ambiguous candidates. The calculator score/mask may still be saved.
- `invalid_camera_geometry`: unsupported size/orientation/camera metadata; no corrected overlay is invented when undistortion cannot be performed.
- `reference_size_required`, `invalid_object_geometry`, `invalid_reference_geometry`, `invalid_planar_geometry`, `checkpoint_error` and `inference_error` retain explicit reasons and no object dimensions.

## Recorded verification and remaining scope

[Verification record](verification.json): **88 tests pass** in the actual `.venv`, including 12 measurement tests and the wood-grain candidate-generation regression. Known synthetic perspective geometry recovers a 75 x 160 mm non-rectangular contour after a 27-degree rotation; this is a mathematical check, not a ruler-based accuracy result. Synthetic paper plus a stub predictor exercises successful full-frame output and proves that the supplied detection box does not determine the dimensions. Tests also cover preview corner restoration, rotated/ambiguous/missing paper, missing object, mandatory undistortion and wrong-resolution rejection.

Two initial implementation runs used the **real frozen network and real saved calibration**; they retain historical source/configuration hashes. The revised detector also continues to reject their saved no-paper corrected frames:

| Run | Input provenance | Result |
| --- | --- | --- |
| object_missing_verified (local output: `inference/runs/object_missing_verified/result.json`) | [Explicitly synthetic empty scene](examples/synthetic_blank.png) | `object_not_found`, null width/height, undistortion applied; overlay (local output: `inference/runs/object_missing_verified/overlay.jpg`) |
| reference_missing_verified (local output: `inference/runs/reference_missing_verified/result.json`) | Unassigned original `dataset/Raw/IMG_20261104_200608.jpg`, outside frozen train/valid/test | Calculator score **0.996219933**, `reference_not_found`, null width/height; full overlay (local output: `inference/runs/reference_missing_verified/overlay.jpg`), display preview (local output: `inference/runs/reference_missing_verified/preview.jpg`) |

Both corrected overlays retain 2296 x 4080 dimensions. The earlier `runs/reference_missing` is a development no-reference run before the corner-array memory-layout fix; those historical runs bind their original source hashes. The current successful run below binds the revised detector. No held-out test image was used to develop this detector, and no model threshold/checkpoint changed. Frozen training/test-evaluation results remain historical records.

### First supplied real referenced photo and detector fix

The user's `calc.jpg.jpeg` initially returned `reference_not_found` in `runs/my_photo`. The paper was visible; the edge-only detector failed to form a closed boundary against the wood grain. Diagnostics are saved under `diagnostics/my_photo/`. Added bright-region proposals at configured levels 140, 160, 180, 200, 220 and 240 while retaining every existing geometric/brightness/contrast/uniformity acceptance check. Overlap selection now prefers a uniform interior and clear outer transition instead of largest area alone. No calculator score/mask threshold, checkpoint, calibration or input geometry changed.

The rerun (local output: `inference/runs/my_photo_retry/result.json`) succeeds on that same original image: **width 80.466370 mm, height 170.600113 mm**, calculator score **0.995487988**. Paper corners were obtained from the intensity-200 proposal and refined on the corrected full frame. Overlay (local output: `inference/runs/my_photo_retry/overlay.jpg`), display preview (local output: `inference/runs/my_photo_retry/preview.jpg`), metric geometry (local output: `inference/runs/my_photo_retry/geometry.json`), verification (local output: `inference/runs/my_photo_retry/verification.json`) and artifact hashes (local output: `inference/runs/my_photo_retry/artifact_sha256.json`) are saved; the original failed run remains unchanged. The paper boundary/mask alignment were checked visually by the agent.

This is a successful pipeline execution, **not a physical-accuracy verdict**. Coplanarity between the paper and calculator outline is unverified; a paper square lying on the table can have a different plane from a raised calculator body. The independent ten-placement ruler experiment remains **not run**. No millimetre-accuracy claim follows from one output, detection confidence, segmentation AP or calibration RMS.
