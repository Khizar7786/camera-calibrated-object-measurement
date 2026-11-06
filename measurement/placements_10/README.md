# Frozen calculator placement accuracy - 2026-11-06

**Result: MO-1 not met.** All ten supplied scenes were processed with the unchanged frozen command: **9 accepted, 1 refused (90% coverage)**. The accepted-only width/height MAE and worst errors exceed the original targets of <=3 mm MAE per dimension and <=8 mm worst dimensional error. A refusal also prevents a complete baseline pass.

This measures repeatability of **one physical calculator**, not generalization across ten calculator products. The set contains **nine newly captured scenes plus the earlier `calc.jpg.jpeg` demo (case_01)**. That demo was already used during paper-detector debugging; it is explicitly retained in the supplied-set table and is not represented as new or blind. The requirement for ten genuinely new scenes is not fulfilled by this set.

## Hand readings and capture conditions

Before this batch made predictions, the user corrected the calculator reading from 7.5 x 16 cm to **7 x 16 cm = 70 x 160 mm**, confirmed those readings apply to all ten placements, and confirmed locked focus/zoom and the paper in the measured-outline plane. The same **50 x 50 mm unmarked paper square** is used throughout. [Hand readings](hand_readings.csv) repeat the same confirmed rigid-object dimensions; no per-photo size variation is invented. The complete outer calculator body is the outline; width is its shorter side. The instrument is a millimetre ruler; no quantified reading uncertainty was supplied.

These physical conditions are user attestation. Actual support height, thickness and paper flatness were not independently established from photographs. [Manifest](manifest.json) records that limitation, the pre-inference declaration, corrected reading, input hashes/EXIF, source-copy mapping, split-separation checks, configuration/source hashes and the reused demo. All ten inputs are portrait 2296 x 4080 Xiaomi 23021RAAEG captures with orientation 1 and reported focal length 4.22 mm. Their content hashes do not overlap the 217 checked original/export/corrected dataset images; the original 50/15/7 splits remain unchanged.

## Frozen execution

```powershell
.venv\Scripts\python.exe -m measurement.evaluate_placements --manifest measurement/placements_10/manifest.json --output measurement/placements_10/results --device cuda
```

This wrapper invokes the existing `python -m inference.measure` command once for every original photo. It does not change calibration, weights, reference settings, model sizing (512/960 aspect-preserving), score cutoff 0.5 or mask threshold 0.5. Every run applies normal full-frame undistortion. [Exact per-image commands](results/commands.txt), [run record](results/run.json), and each case's console, environment, effective configuration, source hashes and result JSON are saved. Reproduction must use a fresh output directory, for example `--output measurement/placements_10/reproduction_01`.

Checkpoint SHA-256: `f46337538e462e09c5438a7c7d0aa675538a5ae819573d3ed3b84a96a0043a39`.
Inference configuration SHA-256: `42429a62c08f51e8b7c856565fa932c5c9781c0464c35cc79fe30f00e5217d9a`.
Calibration SHA-256: `9ad192f78238887e413df81262b2f276140de9d7c4c5558f1cf4783114cb2b29`.
The frozen inference implementation comes from commit `0480595bc87ef04bd8db33c798fd7ff9e549e510`; the new wrapper and tests are separately hash-bound in the manifest.

## Per-instance results

W/H mean shorter/longer side. Absolute error is `abs(predicted_mm - hand_mm)`; percentage error is `100 * absolute_error_mm / hand_mm`. The table rounds values for display; the [CSV](results/per_instance.csv), [JSON](results/per_instance.json) and [summary](results/summary.json) retain full precision. Accepted/refused is the frozen CLI status, not an accuracy guarantee. All ten scenes remain in the denominator for coverage. Refused dimensions/errors are missing, never zero.

| Image ID / input | Hand W / H (mm) | Predicted W / H (mm) | Absolute W / H error (mm) | W / H error (%) | Confidence | Accepted/refused | Reason / overlay |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| [case_01 (reused demo)](results/../inputs/calc.jpg.jpeg) | 70 / 160 | 80.466 / 170.600 | 10.466 / 10.600 | 14.952 / 6.625 | 0.995488 | Accepted (success) | None; [overlay](results/cases/case_01/overlay.jpg) |
| [case_02](results/../inputs/IMG_20261106_155037.jpg.jpeg) | 70 / 160 | 85.474 / 173.548 | 15.474 / 13.548 | 22.105 / 8.467 | 0.996183 | Accepted (success) | None; [overlay](results/cases/case_02/overlay.jpg) |
| [case_03](results/../inputs/IMG_20261106_155047.jpg.jpeg) | 70 / 160 | 73.780 / 162.841 | 3.780 / 2.841 | 5.400 / 1.776 | 0.994135 | Accepted (success) | None; [overlay](results/cases/case_03/overlay.jpg) |
| [case_04](results/../inputs/IMG_20261106_155109.jpg.jpeg) | 70 / 160 | 83.244 / 169.978 | 13.244 / 9.978 | 18.920 / 6.236 | 0.996307 | Accepted (success) | None; [overlay](results/cases/case_04/overlay.jpg) |
| [case_05](results/../inputs/IMG_20261106_155117.jpg.jpeg) | 70 / 160 | 86.165 / 176.596 | 16.165 / 16.596 | 23.093 / 10.372 | 0.996455 | Accepted (success) | None; [overlay](results/cases/case_05/overlay.jpg) |
| [case_06](results/../inputs/IMG_20261106_155128.jpg.jpeg) | 70 / 160 | 87.837 / 178.859 | 17.837 / 18.859 | 25.481 / 11.787 | 0.991796 | Accepted (success) | None; [overlay](results/cases/case_06/overlay.jpg) |
| [case_07](results/../inputs/IMG_20261106_155143.jpg.jpeg) | 70 / 160 | missing | missing | missing | 0.993914 | Refused (reference_not_found) | No isolated bright unmarked square boundary passed the configured checks; [overlay](results/cases/case_07/overlay.jpg) |
| [case_08](results/../inputs/IMG_20261106_155147.jpg.jpeg) | 70 / 160 | 84.820 / 171.404 | 14.820 / 11.404 | 21.172 / 7.128 | 0.996336 | Accepted (success) | None; [overlay](results/cases/case_08/overlay.jpg) |
| [case_09](results/../inputs/IMG_20261106_155827.jpg.jpeg) | 70 / 160 | 75.220 / 168.353 | 5.220 / 8.353 | 7.457 / 5.221 | 0.995801 | Accepted (success) | None; [overlay](results/cases/case_09/overlay.jpg) |
| [case_10](results/../inputs/IMG_20261106_155829.jpg.jpeg) | 70 / 160 | 79.583 / 166.482 | 9.583 / 6.482 | 13.690 / 4.051 | 0.994820 | Accepted (success) | None; [overlay](results/cases/case_10/overlay.jpg) |

## Summary

| Accepted-only statistic (n=9 per dimension) | Width | Height |
| --- | ---: | ---: |
| MAE (mm) | 11.843224 | 10.962348 |
| Mean percentage error (%) | 16.918891 | 6.851468 |
| Median absolute error (mm) | 13.243668 | 10.600113 |
| Median percentage error (%) | 18.919525 | 6.625071 |
| Worst absolute error (mm) | 17.836929 | 18.858841 |
| Worst percentage error (%) | 25.481328 | 11.786776 |

Counts: **9 accepted / 1 refused / 10 supplied**. Every accepted-only statistic uses **9 width readings or 9 height readings**. The refused case is **case_07, `IMG_20261106_155143.jpg.jpeg`**, with `reference_not_found`: no isolated bright unmarked square boundary passed the configured checks. Calculator confidence was 0.993914; dimensions remain null and its errors are blank in CSV.

For the **nine genuinely new scenes only**, **8 were accepted and 1 refused**. Their accepted-only MAE is **12.015330 mm width / 11.007627 mm height**, with denominator 8 for each dimension. Those subset figures are supplementary; the primary table above preserves every supplied case.

All nine accepted predictions overestimate both confirmed dimensions. Case_06 has the worst absolute width and height errors. The calculator confidence scores do not establish scale accuracy. The experiment does not isolate the source of bias: mask-outline error, reference-corner/scale error and physical plane/flatness effects are possible contributors, not measured diagnoses. No settings were tuned from these results. **The raw-versus-undistorted error comparison remains Not run**, so no measured improvement from undistortion is claimed.

## Saved evidence and verification

- [All corrected overlays contact sheet](overlay_contact_sheet.jpg) and [local review gallery](review.html); each case links to its original, full-frame corrected image, mask, overlay, JSON and geometry where available.
- [Verification](verification.json): independent Decimal-based errors/statistics, unchanged source/configuration/split hashes, byte-identical source copies, pre-inference hand-record timing and full-frame output dimensions verified for all ten cases. [Clean test log](verification_tests.txt): **93 tests pass**, including missing-result/coverage and error-aggregation regressions. The original [preflight log](preflight_tests.txt) is retained with its PowerShell stderr formatting.
- [Raw output hashes](results/artifacts_sha256.json) bind all per-case results and summaries. The [complete experiment checksums](SHA256SUMS.json) bind inputs, hand readings, reports and verification as well. Inputs and hand readings are also bound by the manifest; source photos under `dataset/10 test cases` remain untouched. The committed experiment copies are under `inputs/`.

The nine accepted paper boundaries visually align with the supplied squares at review scale. In case_07 the paper is visible, but the frozen detector refused it; its failure is preserved. Agent overlay inspection does not replace independent corner, physical-plane or ruler validation.
