# Football Event Detection — Inference Pipeline

End-to-end inference pipeline for detecting football events in match videos using the **CALF (Context-Aware Loss Function)** model. Given a raw match video, the system extracts visual features and outputs timestamped event predictions in JSON format.

## Pipeline

```
Video (.mkv) → ResNet152 (2048-d) → PCA (512-d) → CALF Model → Predictions-v2.json
```

## Detected Events (17 Classes)

| | | |
|---|---|---|
| Penalty | Kick-off | Goal |
| Substitution | Offside | Shots on target |
| Shots off target | Clearance | Ball out of play |
| Throw-in | Foul | Indirect free-kick |
| Direct free-kick | Corner | Yellow card |
| Red card | Yellow→Red card | |

## Project Structure

```
├── app.py                          # Main entry point — runs the full pipeline
├── VideoFeatureExtractor.py        # ResNet152 + PCA feature extraction
├── model.py                        # CALF model architecture
├── dataset.py                      # Dataset loader for inference
├── preprocessing.py                # NMS, batch/timestamp utilities
├── json_io.py                      # Writes Predictions-v2.json
├── config/
│   └── classes.py                  # Event class definitions
├── models/
│   └── CALF_finetuned/
│       └── model.pth.tar           # Fine-tuned model checkpoint
├── pca_512_TF2.pkl                 # PCA transform (2048-d → 512-d)
├── average_512_TF2.pkl             # Feature scaler
└── predictions/                    # Output folder
```

## Requirements

- Python 3.10+
- PyTorch
- TensorFlow 2.x (`pip install tensorflow==2.20.0`)
- SoccerNet (`pip install SoccerNet`)
- NumPy, OpenCV

## Usage

### Full pipeline (extraction + inference)

```bash
python app.py --video "path/to/1_224p.mkv" --output_path "predictions/MyMatch"
```

### Skip feature extraction (if `.npy` already exists)

```bash
python app.py --video "path/to/1_224p.mkv" --output_path "predictions/MyMatch" --skip_extraction
```

### GPU support

```bash
python app.py --video "path/to/1_224p.mkv" --gpu 0
```

### All options

| Argument | Default | Description |
|---|---|---|
| `--video` | required | Path to input video (`1_224p.mkv` or `2_224p.mkv`) |
| `--output_path` | `predictions/result` | Output folder for JSON |
| `--model_ckpt` | `models/CALF_finetuned/model.pth.tar` | Model checkpoint path |
| `--gpu` | `-1` (CPU) | GPU device ID |
| `--fps` | `2.0` | Frames per second for extraction |
| `--chunk_size` | `120` | Chunk size in seconds |
| `--receptive_field` | `40` | Receptive field in seconds |
| `--skip_extraction` | `False` | Skip feature extraction step |

## Output

The pipeline produces a `Predictions-v2.json` file compatible with the SoccerNet evaluation framework:

```json
{
  "UrlLocal": "...",
  "predictions": [
    { "gameTime": "1 - 03:22", "label": "Goal", "position": "202000", "half": "1", "confidence": "0.93" },
    ...
  ]
}
```

## Model

The **CALF** (Context-Aware Loss Function) model is a temporal action spotting model designed for football broadcast videos. It uses a multi-scale temporal pyramid with capsule-based detection heads to simultaneously perform action spotting and segmentation.

- Input: 512-d PCA-reduced ResNet152 features at 2 fps
- Output: 17-class event timestamps with confidence scores
