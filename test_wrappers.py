#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, '/Users/pranaynandkeolyar/Documents/NFLSalaryCap')

from backend.agent.lb_model_wrapper import LBModelInference
from backend.agent.cb_model_wrapper import CBModelInference
from backend.agent.s_model_wrapper import SModelInference

tests = [
    ('LB', LBModelInference, 'backend/ML/LB_Pranay_Transformers/lb_best_classifier.pth', 'backend/ML/LB_Pranay_Transformers/lb_player_scaler.joblib'),
    ('CB', CBModelInference, 'backend/ML/CB_Transformers/cb_best_transformer.pth', 'backend/ML/CB_Transformers/cb_player_scaler.joblib'),
    ('S', SModelInference, 'backend/ML/S_Transformers/s_best_transformer.pth', 'backend/ML/S_Transformers/s_player_scaler.joblib'),
]

for name, cls, trans, scaler in tests:
    trans_path = os.path.abspath(trans)
    scaler_path = os.path.abspath(scaler)
    try:
        model = cls(trans_path, scaler_path, None)
        print(f'✓ {name} Model loaded')
    except Exception as e:
        print(f'✗ {name} Error: {str(e)[:150]}')
