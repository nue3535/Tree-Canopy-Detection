# Results Comparison (DeepLabV3+ vs SAM2)

Use this sheet in your report/presentation. Replace placeholders with your final run outputs.

## Experimental Setup

- Dataset: Solafune Tree Canopy dataset
- Split protocol: `train.txt` / `val.txt` (fixed split)
- Input resolution: `<fill>`
- Hardware: `<fill>`
- Runtime environment: `<fill>`

## Quantitative Metrics

| Metric | DeepLabV3+ | SAM2 | Better Method | Notes |
|---|---:|---:|---|---|
| Mean IoU | `<fill>` | `<fill>` | `<fill>` | |
| Mean Dice | `<fill>` | `<fill>` | `<fill>` | |
| AP@0.75 / mAP (if computed) | `<fill>` | `<fill>` | `<fill>` | Ensure same protocol |
| Inference latency (sec/image) | `<fill>` | `<fill>` | `<fill>` | |
| Model initialization complexity | `<fill>` | `<fill>` | `<fill>` | |

## Qualitative Assessment

| Scenario | DeepLabV3+ | SAM2 | Observations |
|---|---|---|---|
| Sparse trees | `<fill>` | `<fill>` | |
| Dense/group canopy | `<fill>` | `<fill>` | |
| Mixed background clutter | `<fill>` | `<fill>` | |
| Boundary smoothness | `<fill>` | `<fill>` | |

Add screenshot references from `docs/screenshots/`:

- `docs/screenshots/deeplab_example_01.png`
- `docs/screenshots/sam2_example_01.png`
- `docs/screenshots/gui_upload_prediction.png`

## Selected Method and Rationale

**Chosen method for deployment:** `<DeepLabV3+ / SAM2 / hybrid>`

**Why:**

1. `<quantitative reason>`
2. `<qualitative reason>`
3. `<operational/dependency reason>`

## Suggested Improvements

1. Add checkpoint management/versioning for reproducible deployment.
2. Expand evaluation on additional scenes/resolutions.
3. Improve post-processing (morphological cleaning, component filtering).
4. Consider ensemble/hybrid strategy (DeepLab prior + SAM refinement).
