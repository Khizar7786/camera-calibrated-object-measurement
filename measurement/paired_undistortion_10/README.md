# Paired raw versus corrected measurement - 2026-11-06

**Completed: 20 fresh runs on the same ten original accuracy photos.** Corrected mode accepted **9/10**, refused **1**; experimental raw mode accepted **7/10**, refused **3**. The primary comparison uses **7 jointly valid accuracy scenes**, not the separate seven-image segmentation test split. Refusals remain missing measurements and are never counted as zero error. **The normal corrected MO-1 target remains not met.**

## Frozen design and reproducible commands

The original [accuracy manifest](../placements_10/manifest.json), [70 x 160 mm hand readings](../placements_10/hand_readings.csv), all ten original photos and original run remain untouched. The paper reference is **50 x 50 mm**. The set is one rigid calculator photographed repeatedly, with nine new photos and one reused demo (case_01). Seven jointly valid pairs include that reused demo; the scope is repeatability, not product generalization or ten new blind scenes. Camera focus/zoom and matching reference/outline plane were user-confirmed; physical support height, flatness and thickness effects are not independently measured.

[Pre-inference declaration](manifest.json) freezes the input/reading/configuration/source hashes, analysis and sign convention before either mode ran. Both use the same epoch-2 checkpoint, exact effective inference configuration, 0.5 score and mask probability thresholds, aspect-preserving 512/960 sizing, seed 42, CUDA device, paper detector settings, corner refinement and `measurement.planar.measure_mask_contour_mm` conversion. Width is the shorter metric rectangle side; height is longer. No training, checkpoint reselection, data edits or threshold/detector changes were made from these results.

Checkpoint SHA-256: `f46337538e462e09c5438a7c7d0aa675538a5ae819573d3ed3b84a96a0043a39`.
Inference configuration SHA-256: `42429a62c08f51e8b7c856565fa932c5c9781c0464c35cc79fe30f00e5217d9a`.
The unchanged normal implementation is bound to source at commit `f8191d12a0a8c014b603f988f90baae6dc3fccd6`; new experimental sources are separately hash-bound in the declaration.

```powershell
.venv\Scripts\python.exe -m measurement.compare_undistortion --manifest measurement/paired_undistortion_10/manifest.json --output measurement/paired_undistortion_10/results --device cuda
```

The saved `results` directory is never overwritten. Use a fresh output such as `measurement/paired_undistortion_10/reproduction_01` for reproduction. [Exact 20 per-image commands](results/commands.txt), [run record](results/run.json), [raw mode/error CSV (20 rows)](results/per_mode.csv), [paired CSV (10 rows)](results/paired.csv), [full summary](results/summary.json) and each mode's JSON, mask, overlay, frame, geometry, console, environment and source hashes are saved.

The raw bypass is deliberately separate from normal inference and requires an explicit flag:

```powershell
.venv\Scripts\python.exe -m measurement.raw_inference measurement/placements_10/inputs/calc.jpg.jpeg --experimental-skip-undistortion --output measurement/raw_example_01 --device cuda
```

Normal `inference.measure` remains byte-identical and always undistorts. The raw entry validates the same camera geometry/orientation but omits only the lens remap. It runs reference detection and segmentation afresh on that raw frame. Raw JSON says `experimental_only=true`, `undistortion_applied=false`, `coordinate_system="raw_distorted_full_frame"`; corner/homography/rectangle fields explicitly say `raw`. The common detector/geometry routines have legacy corrected-frame field names for normal use; the adapter labels the freshly calculated raw values without importing any corrected coordinates. The raw overlay has an experimental-bypass banner. Each mode projects its own mask contour using its own ordered reference corners, then fits the same metric rectangle. Raw images have no remap-created invalid border; bounds/frame-touch checks remain enforced.

## Per-instance outputs, errors and refusals

W/H mean shorter/longer sides. `absolute_error_mm = abs(predicted_mm - hand_mm)` and `percentage_error = 100 * absolute_error_mm / hand_mm`. Display values are rounded; CSV/JSON retains full precision. Delta is **raw error minus corrected error**: positive means skipping worsened accuracy. Unpaired deltas are missing, not zero.

| Scene / original | Hand W / H (mm) | Corrected W / H (mm) | Raw W / H (mm) | Confidence C / R | Paired valid | Refusal reason |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| [case_01 (reused demo)](../placements_10/inputs/calc.jpg.jpeg) | 70 / 160 | [80.466 / 170.600; success](results/corrected/case_01/result.json) | [80.872 / 174.823; success](results/raw/case_01/result.json) | 0.995488 / 0.995532 | Yes | None |
| [case_02](../placements_10/inputs/IMG_20261106_155037.jpg.jpeg) | 70 / 160 | [85.474 / 173.548; success](results/corrected/case_02/result.json) | [86.657 / 175.673; success](results/raw/case_02/result.json) | 0.996183 / 0.996047 | Yes | None |
| [case_03](../placements_10/inputs/IMG_20261106_155047.jpg.jpeg) | 70 / 160 | [73.780 / 162.841; success](results/corrected/case_03/result.json) | [missing; reference_not_found](results/raw/case_03/result.json) | 0.994135 / 0.994413 | No | raw: Multiple plausible paper squares; reference is ambiguous |
| [case_04](../placements_10/inputs/IMG_20261106_155109.jpg.jpeg) | 70 / 160 | [83.244 / 169.978; success](results/corrected/case_04/result.json) | [80.165 / 166.869; success](results/raw/case_04/result.json) | 0.996307 / 0.996198 | Yes | None |
| [case_05](../placements_10/inputs/IMG_20261106_155117.jpg.jpeg) | 70 / 160 | [86.165 / 176.596; success](results/corrected/case_05/result.json) | [83.229 / 172.005; success](results/raw/case_05/result.json) | 0.996455 / 0.996496 | Yes | None |
| [case_06](../placements_10/inputs/IMG_20261106_155128.jpg.jpeg) | 70 / 160 | [87.837 / 178.859; success](results/corrected/case_06/result.json) | [88.487 / 179.628; success](results/raw/case_06/result.json) | 0.991796 / 0.992108 | Yes | None |
| [case_07](../placements_10/inputs/IMG_20261106_155143.jpg.jpeg) | 70 / 160 | [missing; reference_not_found](results/corrected/case_07/result.json) | [missing; reference_not_found](results/raw/case_07/result.json) | 0.993914 / 0.993867 | No | corrected: No isolated bright unmarked square boundary passed the configured checks; raw: No isolated bright unmarked square boundary passed the configured checks |
| [case_08](../placements_10/inputs/IMG_20261106_155147.jpg.jpeg) | 70 / 160 | [84.820 / 171.404; success](results/corrected/case_08/result.json) | [97.647 / 185.033; success](results/raw/case_08/result.json) | 0.996336 / 0.996286 | Yes | None |
| [case_09](../placements_10/inputs/IMG_20261106_155827.jpg.jpeg) | 70 / 160 | [75.220 / 168.353; success](results/corrected/case_09/result.json) | [78.439 / 171.502; success](results/raw/case_09/result.json) | 0.995801 / 0.996708 | Yes | None |
| [case_10](../placements_10/inputs/IMG_20261106_155829.jpg.jpeg) | 70 / 160 | [79.583 / 166.482; success](results/corrected/case_10/result.json) | [missing; reference_not_found](results/raw/case_10/result.json) | 0.994820 / 0.995056 | No | raw: No isolated bright unmarked square boundary passed the configured checks |

| Scene | Corrected absolute W / H error (mm) | Raw absolute W / H error (mm) | Corrected W / H error (%) | Raw W / H error (%) | Raw minus corrected absolute W / H error (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| case_01 | 10.466 / 10.600 | 10.872 / 14.823 | 14.952 / 6.625 | 15.531 / 9.264 | 0.405 / 4.223 |
| case_02 | 15.474 / 13.548 | 16.657 / 15.673 | 22.105 / 8.467 | 23.796 / 9.795 | 1.183 / 2.125 |
| case_03 | 3.780 / 2.841 | missing | 5.400 / 1.776 | missing | missing |
| case_04 | 13.244 / 9.978 | 10.165 / 6.869 | 18.920 / 6.236 | 14.522 / 4.293 | -3.078 / -3.109 |
| case_05 | 16.165 / 16.596 | 13.229 / 12.005 | 23.093 / 10.372 | 18.898 / 7.503 | -2.936 / -4.591 |
| case_06 | 17.837 / 18.859 | 18.487 / 19.628 | 25.481 / 11.787 | 26.410 / 12.268 | 0.650 / 0.770 |
| case_07 | missing | missing | missing | missing | missing |
| case_08 | 14.820 / 11.404 | 27.647 / 25.033 | 21.172 / 7.128 | 39.496 / 15.645 | 12.827 / 13.628 |
| case_09 | 5.220 / 8.353 | 8.439 / 11.502 | 7.457 / 5.221 | 12.055 / 7.189 | 3.219 / 3.149 |
| case_10 | 9.583 / 6.482 | missing | 13.690 / 4.051 | missing | missing |

## All-valid statistics, each mode's own denominator

These mode summaries describe coverage and observed errors. Their accepted sets differ, so subtracting their means would combine treatment and scene-composition differences. The primary effect estimate uses the common paired set below.

| Statistic | Corrected width | Raw width | Corrected height | Raw height |
| --- | ---: | ---: | ---: | ---: |
| Accepted denominator per dimension | 9 | 7 | 9 | 7 |
| MAE (mm) | 11.843224 | 15.070784 | 10.962348 | 15.076059 |
| Mean percentage error (%) | 16.918891 | 21.529692 | 6.851468 | 9.422537 |
| Median absolute error (mm) | 13.243668 | 13.228676 | 10.600113 | 14.822754 |
| Median percentage error (%) | 18.919525 | 18.898108 | 6.625071 | 9.264221 |
| Worst absolute error (mm) | 17.836929 | 27.647324 | 18.858841 | 25.032730 |
| Worst percentage error (%) | 25.481328 | 39.496177 | 11.786776 | 15.645456 |

Counts across all ten supplied scenes: corrected **9 accepted / 1 refused (90%)**; raw **7 accepted / 3 refused (70%)**. Corrected refuses case_07. Raw refuses case_03 as ambiguous (`Multiple plausible paper squares; reference is ambiguous`), and cases 07 and 10 because no candidate passed the checks. Each refusal retains the calculator confidence, reason and saved overlay with null dimensions.

## Paired-valid statistics, same seven scenes in both modes

The jointly accepted IDs are **case_01, case_02, case_04, case_05, case_06, case_08 and case_09**. Every statistic below uses **7 width or 7 height results per mode**, on exactly these IDs. Cases 03, 07 and 10 remain in the per-instance table but have no paired error change. Paired coverage of the full supplied set is **7/10**; within-set acceptance is 7/7 only because this table conditions on both modes succeeding.

| Statistic | Corrected width | Raw width | Corrected height | Raw height |
| --- | ---: | ---: | ---: | ---: |
| Accepted denominator per dimension | 7 | 7 | 7 | 7 |
| MAE (mm) | 13.317996 | 15.070784 | 12.762610 | 15.076059 |
| Mean percentage error (%) | 19.025709 | 21.529692 | 7.976631 | 9.422537 |
| Median absolute error (mm) | 14.820351 | 13.228676 | 11.404297 | 14.822754 |
| Median percentage error (%) | 21.171929 | 18.898108 | 7.127686 | 9.264221 |
| Worst absolute error (mm) | 17.836929 | 27.647324 | 18.858841 | 25.032730 |
| Worst percentage error (%) | 25.481328 | 39.496177 | 11.786776 | 15.645456 |

| Paired error-change statistic (raw minus corrected) | Width | Height |
| --- | ---: | ---: |
| Mean change (mm) | 1.752788 | 2.313448 |
| Median change (mm) | 0.649925 | 2.124695 |
| Minimum change (mm) | -3.078323 | -4.591156 |
| Maximum change (mm) | 12.826973 | 13.628433 |
| Mean percentage-error change (percentage points) | 2.503983 | 1.445905 |
| Scenes worse when skipping | 5 | 5 |
| Scenes better when skipping | 2 | 2 |

The primary mean paired effect of **skipping** correction is **+1.752788 mm width MAE / +2.313448 mm height MAE**. These correspond to **+2.503983 width / +1.445905 height percentage points** in mean percentage error. Median paired error changes are **+0.649925 / +2.124695 mm**. The optional pooled MAE across **14 paired dimensions** is **13.040303 mm corrected / 15.073421 mm raw**, difference **+2.033118 mm**; it is not either per-dimension MAE.

## Interpretation and recording answer

Correction improves paired mean errors here, but **makes cases 04 and 05 worse** in both dimensions. Raw mode's paired median absolute width error is actually lower (**13.228676 mm raw / 14.820351 mm corrected**), despite its higher mean. Case_08 has the largest worsening when skipping (**+12.826973 mm width / +13.628433 mm height**) and materially affects the mean; it is retained. Mean, median, worst cases and every refusal are reported together rather than selecting the most favorable statistic.

Both modes fail the original physical-accuracy target. This experiment does not establish a general calibration benefit for other scenes/calculators. It changes the full input to segmentation and reference detection; confidence/masks and detected corners/refusal outcomes can change. Every raw mask differs in stored pixels from its corrected counterpart, but the frames differ too: that is not an aligned-mask accuracy comparison or proof that all observed change comes from segmentation. No isolated geometric-scale, corner-error or segmentation-error contribution was measured. User-attested plane alignment, ruler uncertainty, support height and paper/outline flatness remain limitations.

Verified numbers for an answer in your own words; the assessment requires answering without reading notes. The recording itself has not been reviewed:

> On the seven accuracy scenes measured by both modes, skipping undistortion increased width MAE from 13.318 to 15.071 mm, a worsening of 1.753 mm, and height MAE from 12.763 to 15.076 mm, a worsening of 2.313 mm. Corrected mode measured 9 of 10 supplied scenes; raw mode measured 7 of 10. The effect was mixed: skipping improved both dimensions in cases 04 and 05, but worsened them in the other five pairs. This is an end-to-end result: segmentation and reference detection also change, so I cannot attribute the entire difference to geometric scale.

## Verification and review

[Verification](verification.json) confirms all 20 mode outputs, same effective configuration/checkpoint, source/input/reading hashes, full-frame dimensions, independent Decimal-based error/summary calculations, and reconstruction of every accepted measurement from that mode's saved mask and own reference corners. **All ten corrected control results, scores and masks exactly reproduce the prior committed run.** All 134 previously checksummed accuracy artifacts remain unchanged. [100 passing tests](preflight_tests.txt) include bypass isolation, explicit acknowledgement, normal-CLI rejection of the flag, per-frame input/geometry checks, raw refusals and paired/empty-intersection denominators.

```powershell
.venv\Scripts\python.exe -m measurement.paired_undistortion_10.verify_results
```

This replay reads saved outputs and needs no new neural-network inference. [Paired contact sheet](paired_overlay_contact_sheet.jpg) and [local review gallery](review.html) show separately labelled corrected/raw images; no cross-frame overlays are constructed. [Output checksums](results/artifacts_sha256.json) bind the run; complete experiment checksums are in [SHA256SUMS.json](SHA256SUMS.json). The original step-11 report is an immutable historical snapshot; its earlier comparison-not-run wording is superseded by this report.
