import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DebertaV2Tokenizer
from tqdm import tqdm
import json
from datetime import datetime
from typing import List, Dict, Union
import argparse

# 导入增强版模块
from enhanced_data_deal import EnhancedSuicideDataset
from enhanced_model import EnhancedSuicideRiskClassifier
from enhanced_feature_utils import extract_enhanced_time_features, extract_temporal_patterns

# 导入级联分类器模块
from cascaded_model import CascadedSuicideRiskClassifier, cascaded_predict

class FusionEnsembleSuicideRiskPredictor:
    """
    Fusion Ensemble Suicide Risk Predictor
    Supports hybrid ensemble prediction of cascaded and enhanced classifiers
    """
    
    def __init__(self, cascaded_model_paths: List[str], enhanced_model_paths: List[str],
                 cascaded_weights: List[float] = None, enhanced_weights: List[float] = None,
                 cascaded_model_weights: List[float] = None, enhanced_model_weights: List[float] = None,
                 model_name=None, device=None, save_path=None):
        """
        Initialize the fusion ensemble predictor
        
        Args:
            cascaded_model_paths: List of cascaded classifier model file paths
            enhanced_model_paths: List of enhanced model file paths
            cascaded_weights: Cascaded classifier ensemble weights
            enhanced_weights: Enhanced model ensemble weights
            cascaded_model_weights: Internal model weights for cascaded classifiers
            enhanced_model_weights: Internal model weights for enhanced models
            model_name: Pretrained model name
            device: Device, automatically selected by default
            save_path: Path to save results
        """
        self.cascaded_model_paths = cascaded_model_paths
        self.enhanced_model_paths = enhanced_model_paths
        self.num_cascaded_models = len(cascaded_model_paths)
        self.num_enhanced_models = len(enhanced_model_paths)
        self.model_name = model_name if model_name else 'microsoft/deberta-v3-large'
        self.save_path = save_path if save_path else './'
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Process cascaded classifier ensemble weights
        if cascaded_weights is None:
            self.cascaded_weights = [1.0 / 2] * 2  # Cascaded classifier and enhanced model each have half weight
        else:
            assert len(cascaded_weights) == 2, "Cascaded classifier ensemble weights must be 2"
            # Normalize weights
            total_weight = sum(cascaded_weights)
            self.cascaded_weights = [w / total_weight for w in cascaded_weights]
        
        # Process enhanced model ensemble weights
        if enhanced_weights is None:
            self.enhanced_weights = [1.0 / 2] * 2  # Cascaded classifier and enhanced model each have half weight
        else:
            assert len(enhanced_weights) == 2, "Enhanced model ensemble weights must be 2"
            # Normalize weights
            total_weight = sum(enhanced_weights)
            self.enhanced_weights = [w / total_weight for w in enhanced_weights]
        
        # Process internal cascaded model weights
        if cascaded_model_weights is None:
            self.cascaded_model_weights = [1.0 / self.num_cascaded_models] * self.num_cascaded_models
        else:
            assert len(cascaded_model_weights) == self.num_cascaded_models, "Cascaded classifier internal model weights must match the number of models"
            # Normalize weights
            total_weight = sum(cascaded_model_weights)
            self.cascaded_model_weights = [w / total_weight for w in cascaded_model_weights]
        
        # Process internal enhanced model weights
        if enhanced_model_weights is None:
            self.enhanced_model_weights = [1.0 / self.num_enhanced_models] * self.num_enhanced_models
        else:
            assert len(enhanced_model_weights) == self.num_enhanced_models, "Enhanced model internal model weights must match the number of models"
            # Normalize weights
            total_weight = sum(enhanced_model_weights)
            self.enhanced_model_weights = [w / total_weight for w in enhanced_model_weights]
        
        print(f"🤖 Initializing Fusion Ensemble Predictor")
        print(f"📊 Cascaded classifier ensemble weights: {[f'{w:.3f}' for w in self.cascaded_weights]}")
        print(f"📊 Enhanced model ensemble weights: {[f'{w:.3f}' for w in self.enhanced_weights]}")
        print(f"📊 Internal cascaded model weights: {[f'{w:.3f}' for w in self.cascaded_model_weights]}")
        print(f"📊 Internal enhanced model weights: {[f'{w:.3f}' for w in self.enhanced_model_weights]}")
        
        # Risk level mapping
        self.risk_labels = {
            0: 0,
            1: 1, 
            2: 2,
            3: 3
        }
        
        # Load tokenizer
        print("🔤 Loading tokenizer...")
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_name)
        
        # Load cascaded classifier models
        self.cascaded_models = []
        for i, model_path in enumerate(cascaded_model_paths):
            print(f"🤖 Loading cascaded classifier model {i+1}/{self.num_cascaded_models}: {model_path}")
            
            model = CascadedSuicideRiskClassifier(
                model_name=self.model_name,
                hidden_dim=512,
                time_feature_dim=12,
                pattern_feature_dim=6,
                use_debiased_attention=True
            )
                
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.cascaded_models.append(model)
        
        # Load enhanced models
        self.enhanced_models = []
        for i, model_path in enumerate(enhanced_model_paths):
            print(f"🤖 Loading enhanced model {i+1}/{self.num_enhanced_models}: {model_path}")
            
            model = EnhancedSuicideRiskClassifier(
                model_name=self.model_name,
                hidden_dim=512,
                num_labels=4,
                time_feature_dim=12,
                pattern_feature_dim=6,
                use_debiased_attention=True
            )
                
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.enhanced_models.append(model)
        
        print("✅ All models loaded!")

    def predict_batch(self, df, batch_size=2, save_results=True, output_path=None):
        """
        Batch fusion ensemble prediction
        
        Args:
            df: DataFrame containing data to predict
            batch_size: Batch size
            save_results: Whether to save results
            output_path: Results save path
            
        Returns:
            list: List of prediction results
        """
        print(f"📊 Starting batch fusion ensemble prediction, total {len(df)} samples...")
        print(f"🤖 Using {self.num_cascaded_models} cascaded classifier models and {self.num_enhanced_models} enhanced models")
        
        # Extract time features
        time_features = extract_enhanced_time_features(df)
        pattern_features = extract_temporal_patterns(df)
        
        # Create dataset
        dataset = EnhancedSuicideDataset(
            df, 
            self.tokenizer, 
            max_len=256, 
            max_posts=5,
            with_labels=False,
            time_features=time_features,
            pattern_features=pattern_features
        )
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_predictions = []
        all_confidences = []
        
        # Batch prediction
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Fusion ensemble prediction")):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                time_features_batch = batch['time_features'].to(self.device)
                pattern_features_batch = batch['pattern_features'].to(self.device)
                
                # Cascaded classifier prediction
                cascaded_all_stage1_logits = []
                cascaded_all_stage2_logits = []
                cascaded_all_stage3_logits = []
                
                for i, model in enumerate(self.cascaded_models):
                    stage1_logits, stage2_logits, stage3_logits = model(
                        input_ids, attention_mask, time_features_batch, pattern_features_batch
                    )
                    cascaded_all_stage1_logits.append(stage1_logits * self.cascaded_model_weights[i])
                    cascaded_all_stage2_logits.append(stage2_logits * self.cascaded_model_weights[i])
                    cascaded_all_stage3_logits.append(stage3_logits * self.cascaded_model_weights[i])
                
                # Ensemble cascaded classifier prediction
                cascaded_stage1_logits = torch.sum(torch.stack(cascaded_all_stage1_logits), dim=0)
                cascaded_stage2_logits = torch.sum(torch.stack(cascaded_all_stage2_logits), dim=0)
                cascaded_stage3_logits = torch.sum(torch.stack(cascaded_all_stage3_logits), dim=0)
                
                # Use cascaded prediction function
                cascaded_preds = cascaded_predict(cascaded_stage1_logits, cascaded_stage2_logits, cascaded_stage3_logits)
                
                # Calculate cascaded classifier confidence
                prob_stage1 = torch.sigmoid(cascaded_stage1_logits)  # P(>0)
                prob_stage2 = torch.sigmoid(cascaded_stage2_logits)  # P(>1)
                prob_stage3 = torch.sigmoid(cascaded_stage3_logits)  # P(>2)
                
                cascaded_confidences = []
                for i, pred in enumerate(cascaded_preds):
                    if pred == 0:
                        # For class 0, confidence is 1 - P(>0)
                        confidence = 1 - prob_stage1[i].item()
                    elif pred == 1:
                        # For class 1, confidence is P(>0) * (1 - P(>1))
                        confidence = prob_stage1[i].item() * (1 - prob_stage2[i].item())
                    elif pred == 2:
                        # For class 2, confidence is P(>1) * (1 - P(>2))
                        confidence = prob_stage2[i].item() * (1 - prob_stage3[i].item())
                    else:  # pred == 3
                        # For class 3, confidence is P(>2)
                        confidence = prob_stage3[i].item()
                    cascaded_confidences.append(confidence)
                
                # Handle case where there are no enhanced models
                if self.num_enhanced_models == 0:
                    # Use only cascaded model predictions
                    for i in range(len(cascaded_preds)):
                        final_pred = cascaded_preds[i].item()
                        final_confidence = cascaded_confidences[i]
                        
                        all_predictions.append(final_pred)
                        all_confidences.append(final_confidence)
                else:
                    # Enhanced model prediction
                    enhanced_all_logits = []
                    for i, model in enumerate(self.enhanced_models):
                        logits = model(input_ids, attention_mask, time_features_batch, pattern_features_batch)
                        enhanced_all_logits.append(logits * self.enhanced_model_weights[i])
                    
                    # Ensemble enhanced model prediction
                    enhanced_logits = torch.sum(torch.stack(enhanced_all_logits), dim=0)
                    enhanced_probs = F.softmax(enhanced_logits, dim=1)
                    enhanced_predictions = torch.argmax(enhanced_probs, dim=1)
                    
                    # Calculate enhanced model confidence
                    enhanced_confidences = []
                    for i, pred in enumerate(enhanced_predictions):
                        enhanced_confidences.append(enhanced_probs[i][pred].item())
                    
                    # Fusion of two model prediction results
                    for i in range(len(cascaded_preds)):
                        # Fusion prediction results based on weights
                        cascaded_pred = cascaded_preds[i].item()
                        enhanced_pred = enhanced_predictions[i].item()
                        
                        # Simple voting fusion: If two models predict the same, use that prediction; otherwise use the model with higher confidence
                        if cascaded_pred == enhanced_pred:
                            final_pred = cascaded_pred
                            # Fusion confidence
                            final_confidence = (cascaded_confidences[i] * self.cascaded_weights[0] + 
                                              enhanced_confidences[i] * self.enhanced_weights[1])
                        else:
                            # Use the model with higher confidence
                            if cascaded_confidences[i] * self.cascaded_weights[0] > enhanced_confidences[i] * self.enhanced_weights[1]:
                                final_pred = cascaded_pred
                                final_confidence = cascaded_confidences[i] * self.cascaded_weights[0]
                            else:
                                final_pred = enhanced_pred
                                final_confidence = enhanced_confidences[i] * self.enhanced_weights[1]
                        
                        all_predictions.append(final_pred)
                        all_confidences.append(final_confidence)
        
        # Build results
        results = []
        for i, (pred, confidence) in enumerate(zip(all_predictions, all_confidences)):
            result = {
                'sample_id': int(i),  # Convert to native Python int type
                'suicide_risk': int(pred),  # Convert to native Python int type
                'risk_label': int(self.risk_labels[pred]),  # Convert to native Python int type
                'confidence': float(confidence),  # Convert to native Python float type
                'num_cascaded_models': int(self.num_cascaded_models),  # Convert to native Python int type
                'num_enhanced_models': int(self.num_enhanced_models)  # Convert to native Python int type
            }
            
            # If original data has user_id, add to result
            if 'user_id' in df.columns:
                result['user_id'] = int(df.iloc[i]['user_id']) if not pd.isna(df.iloc[i]['user_id']) else None
            
            results.append(result)
        
        # Save results
        if save_results:
            # Generate output path
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"{self.save_path}fusion_ensemble_{self.num_cascaded_models}cascaded_{self.num_enhanced_models}enhanced_{timestamp}.xlsx"
            
            # Save as Excel format
            results_df = pd.DataFrame(results)
            
            try:
                # Select columns to save
                columns_to_save = ['sample_id', 'suicide_risk', 'confidence']
                if 'user_id' in results_df.columns and results_df['user_id'].notna().any():
                    columns_to_save = ['user_id', 'suicide_risk', 'confidence']
                
                results_df_export = results_df[columns_to_save]
                results_df_export.to_excel(output_path, index=False)
                print(f"📊 Prediction results saved to: {output_path}")
            except ImportError:
                print("⚠️ openpyxl not installed, skipping Excel save")
            except Exception as e:
                print(f"❌ Error saving Excel file: {e}")
        
        return results

    def predict_from_csv(self, csv_path, output_path=None):
        """
        Read data from CSV file and perform fusion ensemble prediction
        
        Args:
            csv_path: CSV file path
            output_path: Results save path
        
        Returns:
            list: Prediction results
        """
        print(f"📁 Reading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        results = self.predict_batch(
            df, 
            save_results=True, 
            output_path=output_path
        )
        
        return results

    def get_ensemble_statistics(self, results):
        """
        Get statistics of fusion ensemble prediction
        
        Args:
            results: List of prediction results
            
        Returns:
            dict: Statistics
        """
        risk_counts = {}
        total_samples = len(results)
        confidence_scores = []
        
        for result in results:
            risk_level = result['risk_label']
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
            confidence_scores.append(result['confidence'])
        
        statistics = {
            'total_samples': total_samples,
            'num_cascaded_models': results[0]['num_cascaded_models'] if results else 0,
            'num_enhanced_models': results[0]['num_enhanced_models'] if results else 0,
            'risk_distribution': {
                risk: {
                    'count': count,
                    'percentage': round(count / total_samples * 100, 2)
                }
                for risk, count in risk_counts.items()
            },
            'confidence_stats': {
                'mean': np.mean(confidence_scores),
                'std': np.std(confidence_scores),
                'min': np.min(confidence_scores),
                'max': np.max(confidence_scores)
            }
        }
        
        return statistics

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Suicide Risk Prediction')
    parser.add_argument('--save_path', type=str, default='./', 
                        help='Path to save results')
    parser.add_argument('--model_name', type=str, 
                        default='microsoft/deberta-v3-large',
                        help='Pretrained model name')
    parser.add_argument('--test_csv_path', type=str, 
                        default='evaluate_on_leaderboard.csv',
                        help='Test CSV file path')
    parser.add_argument('--output_path', type=str, default='Flash.xlsx',
                        help='Output file path')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda:0, cpu, etc.)')
    parser.add_argument('--cascaded_model_paths', type=str, nargs='+',
                        help='Paths to cascaded model files')
    parser.add_argument('--enhanced_model_paths', type=str, nargs='+',
                        help='Paths to enhanced model files')
    parser.add_argument('--cascaded_weights', type=float, nargs='+',
                        help='Cascaded classifier ensemble weights')
    parser.add_argument('--enhanced_weights', type=float, nargs='+',
                        help='Enhanced model ensemble weights')
    parser.add_argument('--cascaded_model_weights', type=float, nargs='+',
                        help='Internal model weights for cascaded classifiers')
    parser.add_argument('--enhanced_model_weights', type=float, nargs='+',
                        help='Internal model weights for enhanced models')
    
    return parser.parse_args()

def main():
    """
    Main function: Demonstrate the use of fusion ensemble predictor
    """
    args = parse_args()
    
    # Configure model paths
    cascaded_model_paths = args.cascaded_model_paths if args.cascaded_model_paths else [
        'cascaded_enhanced_f10.5109_best.pt'
    ]
    
    enhanced_model_paths = args.enhanced_model_paths if args.enhanced_model_paths else [
    ]
    
    # Model weights (optional, use average weights if not provided)
    cascaded_weights = args.cascaded_weights if args.cascaded_weights else [0.5109, 0.5109]  # Cascaded classifier internal model weights
    enhanced_weights = args.enhanced_weights if args.enhanced_weights else [0.5158, 0.5158]  # Enhanced model internal model weights
    cascaded_model_weights = args.cascaded_model_weights if args.cascaded_model_weights else [1.0]  # Cascaded classifier ensemble weights
    enhanced_model_weights = args.enhanced_model_weights if args.enhanced_model_weights else []  # Enhanced model ensemble weights
    
    # Initialize fusion ensemble predictor
    print("🚀 Initializing fusion ensemble predictor")
    fusion_predictor = FusionEnsembleSuicideRiskPredictor(
        cascaded_model_paths=cascaded_model_paths,
        enhanced_model_paths=enhanced_model_paths,
        cascaded_weights=cascaded_weights,
        enhanced_weights=enhanced_weights,
        cascaded_model_weights=cascaded_model_weights,
        enhanced_model_weights=enhanced_model_weights,
        model_name=args.model_name,
        device=torch.device(args.device) if args.device else None,
        save_path=args.save_path
    )
    
    if os.path.exists(args.test_csv_path):
        print(f"\n📊 Fusion ensemble prediction:")
        results = fusion_predictor.predict_from_csv(
            csv_path=args.test_csv_path,
            output_path=args.output_path
        )
        
        # Display statistics
        stats = fusion_predictor.get_ensemble_statistics(results)
        
        print(f"\n📈 Fusion ensemble prediction statistics:")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Cascaded classifier models used: {stats['num_cascaded_models']}")
        print(f"Enhanced models used: {stats['num_enhanced_models']}")
        print(f"Average confidence: {stats['confidence_stats']['mean']:.3f}")
        print("Risk distribution:")
        for risk, info in stats['risk_distribution'].items():
            print(f"  {risk}: {info['count']} samples ({info['percentage']}%)")
    else:
        print(f"❌ Test file does not exist: {args.test_csv_path}")

if __name__ == "__main__":
    main()