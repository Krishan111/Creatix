# Soil Classification System

### **See Project Architecture diagram in docs/cards/architecture.png**

An optimized computer vision pipeline for soil classification using ensemble one-class models.


## Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/krishan111/soil-classification_annam.git
```

## Install dependencies
```bash
pip install -r requirements.txt
```
##Key Features
1. Image Processing: Automatic format conversion & feature extraction

2. Ensemble Models: 5 Isolation Forest + 3 One-Class SVM models

3. Optimized Thresholding: Targets 44% positive class ratio

4. Robust Fallbacks: Handles failed image processing gracefully

# To run the ML model please follow following commands:
```bash
# 1. Preprocessing - Feature Extraction
python -c "
from src.preprocessing import Preprocessor
from src.config import Config

cfg = Config(base_path='data/soil_competition-2025')
prep = Preprocessor()

# Convert and process images
prep.convert_all_images(cfg.TRAIN_IMAGES_DIR, 'data/train_processed')
prep.convert_all_images(cfg.TEST_IMAGES_DIR, 'data/test_processed')

# Extract features
train_features = prep.load_training_features(cfg.TRAIN_LABELS_CSV, 'data/train_processed')
preprocessed_data = prep.prepare_data(train_features)
"

# 2. Training
python -c "
from src.training import Trainer
from src.preprocessing import Preprocessor

prep = Preprocessor()
trainer = Trainer()
models = trainer.train_ensemble(prep.pca.transform(prep.scaler.transform(train_features)))
"

# 3. Inference
python -c "
from src.inference import InferenceEngine
from src.postprocessing import Postprocessor
from src.config import Config

cfg = Config(base_path='data/soil_competition-2025')
engine = InferenceEngine(models)
post = Postprocessor()

# Process test images
test_features = [prep.extract_features(f'data/test_processed/{img}') for img in os.listdir('data/test_processed')]
test_features = prep.pca.transform(prep.scaler.transform(test_features))

# Predict and optimize
predictions = engine.predict(test_features)
optimized_preds = post.optimize_ratio(predictions)
post.create_submission(test_ids, optimized_preds, 'results/final_predictions.csv')
```