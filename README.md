# Suicide Risk Prediction System

## Project Overview

This project is a deep learning-based suicide risk prediction system designed to predict users' suicide risk levels by analyzing their social media posts and behavioral patterns. The system can help mental health professionals identify high-risk individuals in a timely manner and provide intervention.

## Features

- Multi-model architecture support: Enhanced model and cascaded model
- Multi-level text feature extraction
- Temporal feature analysis and pattern recognition
- Debiasing attention mechanism to improve prediction accuracy
- Support for multi-post sequence analysis

## Model Architecture

### Enhanced Model (EnhancedSuicideRiskClassifier)
- Based on DeBERTa-v3-large pre-trained model
- Debiasing attention mechanism to reduce model bias
- Multi-level feature extractor to fuse semantic information from different layers
- Enhanced temporal feature projector to process temporal features
- GRU network to process post sequences

### Cascaded Model (CascadedSuicideRiskClassifier)
Adopting a hierarchical classification strategy:
1. Step 1: Distinguish Level 0 (no risk) vs Levels 1-3 (risk)
2. Step 2: Distinguish Level 1 (low risk) vs Levels 2-3 (medium-high risk)
3. Step 3: Distinguish Level 2 (medium risk) vs Level 3 (high risk)

## Data Requirements

### Input Data Format
```json
{
  "post_sequence": ["Post 1 content", "Post 2 content", "..."],
  "post_time_sequence": ["2023-01-01 10:00:00", "2023-01-02 15:30:00", "..."],
  "suicide_risk": 0-3 (required for training, optional for prediction)
}
```

### Feature Description
- **Text Features**: User's historical post content
- **Temporal Features**: Post publication time sequence
- **Temporal Pattern Features**: Posting frequency, interval patterns, etc.

## Installation Dependencies

```bash
conda env create -f environment.yml
conda activate bigdata2025
pip install torch
pip install transformers
pip install sentencepiece
pip install openpyxl
```

You should navigate to the project directory in your terminal window.

```bash
cd Flash_Team
```

## Quick Prediction

### Notes

1. ***Model Availability (HuggingFace Dependency)**  
   
   - The system requires access to **HuggingFace** in order to download the `microsoft/deberta-v3-large` model.  
   - If access to HuggingFace is unavailable, you must provide a locally downloaded copy of the model. To use this local model, you can add the argument `--model_name` followed by the local file path to the `deberta-v3-large` model (e.g., `--model_name /path/to/your/local/deberta-v3-large`).
   
1. **quick prediction**  
   
   - If you want quick prediction, simply run:  
     ```bash
     python predictor.py \
       --enhanced_model_paths enhanced_f10.4299_best.pt enhanced_f10.5158_best.pt \
       --enhanced_model_weights 1.0 1.0  
     ```

```bash
# Basic usage
python predictor.py

# Specify custom parameters
python predictor.py --model_name microsoft/deberta-v3-large --test_csv_path evaluate_on_leaderboard.csv --output_path results.xlsx --device cuda:0 --cascaded_model_paths /path/to/cascaded1.pt /path/to/cascaded2.pt --enhanced_model_paths /path/to/enhanced1.pt /path/to/enhanced2.pt --enhanced_model_weights 1.0 1.0

# Specify model paths
python predictor.py --cascaded_model_paths cascaded_enhanced_f10.5109_best.pt  --enhanced_model_paths enhanced_f10.4299_best.pt  --enhanced_model_weights 1.0

# Use only cascaded model (without enhanced model)
python predictor.py --cascaded_model_paths cascaded_enhanced_f10.5109_best.pt --enhanced_model_paths
```

## Command Line Parameter Description

### Basic Parameters
| Parameter | Type | Default Value | Description |
|-----------|------|---------------|-------------|
| `--save_path` | string | `./` | Results save path |
| `--model_name` | string | `microsoft/deberta-v3-large` | Pretrained model name |
| `--test_csv_path` | string | `evaluate_on_leaderboard.csv` | Test CSV file path |
| `--output_path` | string | `Flash.xlsx` | Output file path |
| `--device` | string | Auto-selected | Device (cuda:0, cpu, etc.) |

### Model Path Parameters
| Parameter | Type | Default Value | Description |
|-----------|------|---------------|-------------|
| `--cascaded_model_paths` | string list | `cascaded_enhanced_f10.5109_best.pt` | Cascaded model file paths |
| `--enhanced_model_paths` | string list | `None` | Enhanced model file paths |

### Model Weight Parameters
| Parameter | Type | Default Value | Description |
|-----------|------|---------------|-------------|
| `--cascaded_weights` | float list | `[0.5109, 0.5109]` | Cascaded classifier ensemble weights |
| `--enhanced_weights` | float list | `[0.5158, 0.5158]` | Enhanced model ensemble weights |
| `--cascaded_model_weights` | float list | `[1.0]` | Internal model weights for cascaded classifiers |
| `--enhanced_model_weights` | float list | `[]` | Internal model weights for enhanced models |

### Usage Examples
```bash
# Specify basic parameters
python predictor.py --save_path /path/to/save/ --model_name microsoft/deberta-v3-large --test_csv_path /path/to/test.csv --output_path results.xlsx --device cuda:0

# Specify model paths
python predictor.py --cascaded_model_paths /path/to/cascaded1.pt /path/to/cascaded2.pt --enhanced_model_paths /path/to/enhanced1.pt /path/to/enhanced2.pt

# Specify model weights
python predictor.py --cascaded_model_weights 0.6 0.4 --enhanced_model_weights 0.5 0.3 0.2 --cascaded_weights 0.7 0.3

# Use only one cascaded model, without enhanced model
python predictor.py --cascaded_model_paths /path/to/cascaded_model.pt --enhanced_model_paths

# Combined usage of multiple parameters
python predictor.py --save_path /results/ --test_csv_path test_data.csv --cascaded_model_paths model1.pt model2.pt --enhanced_model_paths model3.pt model4.pt --cascaded_model_weights 0.6 0.4 --enhanced_model_weights 0.5 0.5
```

## Using a Single Cascaded Model

If you want to use only a single cascaded model without an enhanced model, you can do the following:

1. Provide only the cascaded model path, without providing enhanced model paths:
```bash
python predictor.py --cascaded_model_paths /path/to/your/cascaded_model.pt --enhanced_model_paths
```

2. Or completely omit enhanced model related parameters:
```bash
python predictor.py --cascaded_model_paths /path/to/your/cascaded_model.pt
```

In this case, the system will use only the cascaded model for prediction without model fusion.

## Detailed Usage

### 1. Data Preprocessing
```python
from enhanced_feature_utils import extract_enhanced_time_features, extract_temporal_patterns

# Extract temporal features
time_features = extract_enhanced_time_features(dataframe)
pattern_features = extract_temporal_patterns(dataframe)
```

### 2. Model Loading and Prediction
```python
import torch
from enhanced_model import EnhancedSuicideRiskClassifier

# Load model
model = EnhancedSuicideRiskClassifier()
model.load_state_dict(torch.load('model_checkpoint.pth'))
model.eval()

# Execute prediction
with torch.no_grad():
    logits = model(input_ids, attention_mask, time_features, pattern_features)
    predictions = torch.argmax(logits, dim=-1)
```

## Risk Level Definition

- **Level 0**: No suicide risk
- **Level 1**: Low suicide risk
- **Level 2**: Moderate suicide risk
- **Level 3**: High suicide risk

## Project Structure

```
.
├── enhanced_model.py          # Enhanced model definition
├── cascaded_model.py          # Cascaded model definition
├── enhanced_data_deal.py      # Data processing module
├── enhanced_feature_utils.py  # Feature extraction tools
├── feature_utils.py           # Basic feature tools
├── environment.yml            # Environment dependency configuration
├── CODE_GUIDE.md             # Code usage guide
└── README.md                 # Project documentation
```

## Technology Stack

- Python 3.13
- PyTorch 2.8.0
- Transformers (Hugging Face)
- DeBERTa-v3-large pre-trained model
- Data processing libraries such as NumPy, Pandas

## Notes

1. This system is for research and auxiliary diagnostic purposes only and cannot replace professional medical diagnosis
2. Compliance with relevant privacy protection regulations is required when using
3. Model prediction results should be interpreted and verified by professionals
4. It is recommended to run in a GPU environment for optimal performance

## Disclaimer

This project is for academic research purposes only. Prediction results are for reference only and cannot be the sole basis for clinical diagnosis. If suicide risk is detected, please contact professional medical institutions and psychological crisis intervention hotlines in a timely manner.
