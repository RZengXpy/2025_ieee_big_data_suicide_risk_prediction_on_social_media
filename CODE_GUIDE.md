# Suicide Risk Prediction Code Usage Guide

## Project Overview

This project contains multiple deep learning models for predicting users' suicide risk levels (0-3), where:
- Level 0: No suicide risk
- Level 1: Low suicide risk
- Level 2: Moderate suicide risk
- Level 3: High suicide risk

## Model Architecture

### 1. EnhancedSuicideRiskClassifier (Enhanced Model)
Main model with the following features:
- Debiasing attention mechanism
- Multi-level feature extraction
- Enhanced temporal feature fusion
- Relative position encoding enhancement

### 2. CascadedSuicideRiskClassifier (Cascaded Model)
Adopting a cascaded classification strategy:
- Step 1: Distinguish 0 vs (1,2,3)
- Step 2: Distinguish 1 vs (2,3)
- Step 3: Distinguish 2 vs 3

## Data Preparation

### Input Data Format
Prediction requires the following data:

1. **Text Data**: User's historical post sequence
   - Format: String list, e.g., `["Post 1", "Post 2", "Post 3"]`
   - Supports up to 5 posts

2. **Temporal Features**:
   - Basic temporal features (12 dimensions): Including cyclic encoding of hours, weekdays, dates, months, and temporal marker features
   - Temporal pattern features (6 dimensions): Including posting intervals, frequency, nighttime posting ratio, etc.

### Feature Extraction

Use functions in `enhanced_feature_utils.py` to extract features:

```python
from enhanced_feature_utils import extract_enhanced_time_features, extract_temporal_patterns

# Extract basic temporal features
time_features = extract_enhanced_time_features(dataframe)

# Extract temporal pattern features
pattern_features = extract_temporal_patterns(dataframe)
```

## Prediction Process

### 1. Load Model

```python
import torch
from enhanced_model import EnhancedSuicideRiskClassifier
from cascaded_model import CascadedSuicideRiskClassifier

# Load enhanced model
model = EnhancedSuicideRiskClassifier()
model.load_state_dict(torch.load('model_path.pth'))
model.eval()

# Or load cascaded model
# cascaded_model = CascadedSuicideRiskClassifier()
# cascaded_model.load_state_dict(torch.load('cascaded_model_path.pth'))
# cascaded_model.eval()
```

### 2. Data Preprocessing

```python
from transformers import AutoTokenizer
from enhanced_data_deal import EnhancedSuicideDataset

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')

# Create dataset
dataset = EnhancedSuicideDataset(
    dataframe=test_df,
    tokenizer=tokenizer,
    max_len=128,
    max_posts=5,
    with_labels=False,  # Labels not needed for prediction
    time_features=time_features,
    pattern_features=pattern_features
)

# Create data loader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
```

### 3. Execute Prediction

#### Prediction using Enhanced Model:

```python
import torch.nn.functional as F

predictions = []
with torch.no_grad():
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        time_features = batch['time_features']
        pattern_features = batch['pattern_features']
        
        # Forward propagation
        logits = model(input_ids, attention_mask, time_features, pattern_features)
        
        # Get prediction results
        probs = F.softmax(logits, dim=-1)
        batch_predictions = torch.argmax(probs, dim=-1)
        
        predictions.extend(batch_predictions.cpu().numpy())
```

#### Prediction using Cascaded Model:

```python
from cascaded_model import cascaded_predict

predictions = []
with torch.no_grad():
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        time_features = batch['time_features']
        pattern_features = batch['pattern_features']
        
        # Forward propagation
        stage1_logits, stage2_logits, stage3_logits = cascaded_model(
            input_ids, attention_mask, time_features, pattern_features
        )
        
        # Use cascaded prediction function
        batch_predictions = cascaded_predict(stage1_logits, stage2_logits, stage3_logits)
        
        predictions.extend(batch_predictions.cpu().numpy())
```

## Output Interpretation

Prediction results are integers from 0-3, representing different suicide risk levels:
- 0: No suicide risk
- 1: Low suicide risk
- 2: Moderate suicide risk
- 3: High suicide risk

## Notes

1. Ensure input data format is correct, especially post sequences and temporal features
2. Models require GPU environment for optimal performance
3. Set model to evaluation mode during prediction (model.eval())
4. Use torch.no_grad() context manager to save memory
5. Select appropriate model based on actual needs (enhanced or cascaded version)