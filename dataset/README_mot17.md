# MOT17 Dataset Layout

The MOT17 pipeline requires raw frames and ground-truth files in the following directory structure.
No pre-computed CSV files are committed; everything is derived from the raw dataset.

## Required structure

```
data/mot17/train/
    MOT17-02-FRCNN/
        img1/
            000001.jpg
            000002.jpg
            ...
        gt/
            gt.txt
        seqinfo.ini
    MOT17-04-FRCNN/
        ...
    MOT17-05-FRCNN/
        ...
    MOT17-09-FRCNN/
        ...
    MOT17-10-FRCNN/
        ...
    MOT17-11-FRCNN/
        ...
    MOT17-13-FRCNN/
        ...
```

Only the **FRCNN** variants of the 7 train sequences are needed (DPM and SDP share the same
frames and are not processed to avoid redundant computation).

## Obtaining the data

Download MOT17 from the official benchmark: https://motchallenge.net/data/MOT17.zip

After extracting, copy or symlink the 7 FRCNN train sequences into `data/mot17/train/`.

## `gt.txt` format

MOT17 ground-truth files use the MOTChallenge CSV format (no header):

```
frame_id, track_id, x, y, w, h, mark, class_id, visibility
```

- `mark == 1` — annotation is active (mark <= 0 is ignored)
- `class_id == 1` — pedestrian (only class used for evaluation)
- `visibility` — occlusion ratio in [0, 1] (all values accepted, i.e. >= 0.0)

## `seqinfo.ini` format

```ini
[Sequence]
name=MOT17-02-FRCNN
imDir=img1
frameRate=30
seqLength=600
imWidth=1920
imHeight=1080
imExt=.jpg
```

`seqLength` is used to determine the total number of frames per sequence.

## Pipeline steps (run in order)

1. **Generate XML detections** (weak + strong models, COCO pretrained, auto-downloaded):
   ```bash
   python scoring/compute_mot17_detections.py --skip-existing
   ```
   Output: `data/mot17/pred_xml/TRAIN/{model_name}/{seq}/*.xml`

2. **Train IoU-Net and GFL, generate per-frame scores**:
   ```bash
   python scoring/compute_mot17_iounet_gfl.py --method both
   ```
   Output: `scores/iounet/mot17/{seq}.csv`, `scores/gfl/mot17/{seq}.csv`
   Weights: `weights/neural/iou_net_mot17.pth`, `weights/neural/gfl_mot17.pth`

3. **Compute PTC-IoU scores**:
   ```bash
   python scoring/compute_ptc.py --datasets mot17
   ```
   Output: `scores/ptc/mot17/{seq}.csv`

4. **Compute XGBoost scores** (if applicable):
   ```bash
   python scoring/compute_xgboost.py --datasets mot17
   ```

5. **Compute routing curves and pairwise accuracy** (Table 2, Fig 5):
   ```bash
   python scoring/compute_mot17_routing.py
   ```
   Output: `scores/mot17_routing_curves.csv`, `scores/mot17_routing_pairwise.csv`,
   `scores/mot17_oracle_frame_scores.csv`

6. **Reproduce figures and table**:
   ```bash
   python reproduce/tab2.py
   python reproduce/fig5.py
   python reproduce/fig6.py
   ```

## Val / train split

Split: 60% train, 40% val, stratified by base video ID, `random_state=42`.

Val sequences used for evaluation: `MOT17-11-FRCNN`, `MOT17-13-FRCNN`
(and `MOT17-02-FRCNN` for some metrics — see `VAL_SEQS` in `compute_mot17_routing.py`).

## Evaluation settings

| Parameter       | Value                              |
|-----------------|------------------------------------|
| IoU threshold   | 0.7                                |
| Conf threshold  | 0.1 (applied to XML detections)    |
| GT class        | 1 (person)                         |
| GT visibility   | >= 0.0 (all accepted)              |
| AP type         | Global AP (pool all frames, one PR curve) |
