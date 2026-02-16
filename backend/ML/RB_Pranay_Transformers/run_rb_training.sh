#!/bin/bash
# Script to run RB model training and ensemble evaluation
# Make sure virtual environment is activated: source .venv/bin/activate

set -e  # Exit on error

echo "=========================================="
echo "RB Model Training & Evaluation Pipeline"
echo "=========================================="

cd "$(dirname "$0")/../../.."

# Step 1: Train Transformer Model
echo ""
echo "Step 1: Training Transformer Model..."
echo "-----------------------------------"
python backend/ML/RB_Pranay_Transformers/Player_Model_RB.py

# Step 2: Run Ensemble in VALIDATION mode
echo ""
echo "Step 2: Running Ensemble in VALIDATION mode..."
echo "-----------------------------------"
# Temporarily change MODE to VALIDATION
sed -i.bak 's/MODE = "DREAM"/MODE = "VALIDATION"/' backend/ML/RB_Pranay_Transformers/RB_Ensemble.py
python backend/ML/RB_Pranay_Transformers/RB_Ensemble.py
# Restore original
mv backend/ML/RB_Pranay_Transformers/RB_Ensemble.py.bak backend/ML/RB_Pranay_Transformers/RB_Ensemble.py

# Step 3: Run Ensemble in DREAM mode
echo ""
echo "Step 3: Running Ensemble in DREAM mode..."
echo "-----------------------------------"
python backend/ML/RB_Pranay_Transformers/RB_Ensemble.py

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Transformer model: backend/ML/RB_Pranay_Transformers/rb_best_classifier.pth"
echo "  - Scaler: backend/ML/RB_Pranay_Transformers/rb_player_scaler.joblib"
echo "  - XGBoost model: backend/ML/RB_Pranay_Transformers/rb_best_xgb.joblib"
echo "  - Validation results: backend/ML/RB_Pranay_Transformers/RB_2024_Validation_Results.csv"
echo "  - Dream mode results: backend/ML/RB_Pranay_Transformers/RB_2025_Final_Rankings.csv"
