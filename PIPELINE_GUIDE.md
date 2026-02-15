# NFL Player Performance & ROI Pipeline

This guide explains how the QB performance model is trained, tested, and integrated into the GM Agent for salary cap analysis.

## 🏗️ Overall Architecture

The pipeline follows a **Transformer-based Regression** approach:
1. **Data Prep**: Historical QB stats are loaded and converted into 3-year "sequences."
2. **Training**: A PyTorch Transformer model learns to predict the **next year's PFF grade** (0-100).
3. **Inference**: The `PlayerModelInference` wrapper handles feature normalization and zero-padding for young players.
4. **ROI Logic**: The GM Agent uses the predicted grade to calculate player value vs. their market salary.

---

## 📂 Project Structure

- `backend/ML/QB.csv`: The master historical dataset.
- `backend/ML/transformers/Player_Model_QB.py`: The core training script (converts sequences -> MSE loss).
- `backend/ML/transformers/test_model.py`: Evaluation tool to test specific years and view "Virtual Tiers."
- `backend/agent/model_wrapper.py`: The bridge between the raw model and the GM Agent.
- `backend/agent/agent_graph.py`: The decision engine that uses predictions to make hiring choices.

---

## 🚀 How To: Train & Test

### 1. Training the Model
If you add new data to `QB.csv`, you should re-train the model:
```bash
# From project root
python backend/ML/transformers/Player_Model_QB.py
```
*   **Input**: `backend/ML/QB.csv`
*   **Output**: `backend/ML/transformers/best_classifier.pth` (The weights)

### 2. Testing Predictions
To see how the model performs on a specific season (e.g., 2024):
```bash
python backend/ML/transformers/test_model.py
```
*   **Virtual Tiers**: Even though the model predicts a number (e.g., 82.5), the test script maps this to Elite/Starter/Poor buckets for easy scanning.
*  - **Native History Support**: The model supports players with only 1 or 2 years of history (e.g., CJ Stroud). We use the actual available sequence length natively (no padding), taking up to a maximum of **4 years** of history.
*   **Split Metrics**: The test script reports separate accuracies for "Full History" vs. "Short History" players so you can gauge prediction reliability.

---

## 🤖 Using the Agent (ROI Logic)

The "Magic" happens in the conversion from **Regression -> ROI**:

1.  **Predict Grade**: The agent asks the model: *"What will Patrick Mahomes score in 2025?"* -> Result: **85.4**.
2.  **Calculate ROI**:
    *   `Value = (Predicted Grade / Avg Starter Grade) * Position Market Rate`
    *   If `Value > Asking Salary`, the Agent flags the player as a **"Value Buy."**
3.  **Risk Assessment**: The Agent looks at the prediction confidence (historically +/- 9 grade points) to determine if the signing is High Risk.

---

## 🛠️ Advanced Maintenance

### Changing Features
If you want to add new stats (e.g., "Air Yards"), update the `features` list in `Player_Model_QB.py` AND `model_wrapper.py`.

### Adjusting Tiers
The "Virtual Tiers" are defined in `get_tier()` within `Player_Model_QB.py`.
- **Elite**: Grade >= 80
- **Starter**: Grade 60 - 80
- **Reserve**: Grade < 60
