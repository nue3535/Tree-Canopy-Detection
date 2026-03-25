## Expected Data Layout

Place the Solafune competition files in the project like this:

```text
data/
  raw/
    annotations/
      train_annotations.json
    train_images_tif/
      *.tif
    evaluation_images_tif/
      *.tif
  processed/
    train.txt
    val.txt
    masks/
      *.png
```

Notes:

- `backend/scripts/deeplab_v3plus.py` can generate `data/processed/masks/*.png` directly from `train_annotations.json`.
- `backend/scripts/deeplab_v3plus.py` and `backend/scripts/sam2_workflow.py` both support `.tif`, `.tiff`, `.png`, `.jpg`, and `.jpeg` image files.
- If you already converted imagery to PNG, `train_images_png/` and `evaluation_images_png/` are still accepted as fallbacks.
- `train.txt` and `val.txt` should contain image stems only, one per line.

Recommended first step after adding raw training data:

```bash
python backend/scripts/deeplab_v3plus.py prepare-data
```

That command creates the processed segmentation masks required by the DeepLabV3+ workflow.
