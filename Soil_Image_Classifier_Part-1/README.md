# Soil Classification System Part-1

### **See Project Architecture diagram in docs/cards/architecture.png**


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
## **Key Features**

### Fundamental:
1. **Objective**: Classify soil images into 4 types with ≥0.90 F1 score
2. **Approach**: Transfer learning with fine-tuning and class imbalance handling
3. **Pipeline**: 
   - Image preprocessing → Model training → Evaluation → Submission
4. **Optimizations**: GPU acceleration, mixed precision, and data augmentation

---

### Advanced:

#### 1. **Advanced Data Augmentation**
```python
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.3,
    brightness_range=[0.7, 1.3],
    channel_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)
```
- **Purpose**: Combats overfitting and improves generalization
- **Unique Aspects**:
  - Channel shifting for lighting variation simulation
  - Aggressive spatial transforms (40° rotation, 30% shifts)
  - Triple dataset replication for minority classes

#### 2. **Class-Weighted Loss Optimization**
```python
class_weights = {
    0: 3.0,  # Clay soil (boosted)
    1: 1.8,  # Black Soil 
    2: 1.0,  # Others...
    3: 1.0
}
```
- **Purpose**: Addresses imbalanced soil type distribution
- **Strategy**:
  - 3× weight for rare "Clay soil"
  - 1.8× weight for "Black Soil"
  - Automated weight calculation based on class frequency

#### 3. **Enhanced DenseNet Architecture**
```python
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
```
- **Modifications**:
  - Unfrozen last 30 layers for fine-tuning
  - Added 3 dense layers with dropout (0.3-0.5)
  - Batch normalization between layers
- **Performance**: Targets macro F1 score instead of accuracy

#### 4. **Precision Training Setup**
```python
# GPU Optimization
mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

# Callbacks
callbacks = [
    EarlyStopping(patience=4),
    ReduceLROnPlateau(factor=0.5, min_lr=0.00001)
]
```
- **Hardware Utilization**:
  - Mixed precision (FP16/FP32) for faster training
  - XLA compilation for GPU acceleration
- **Training Control**:
  - Dynamic learning rate reduction
  - Early stopping with best weights restoration

---

### **Unique Differentiators**
1. **Soil-Specific Augmentations**: Simulates real-world field conditions with color/brightness shifts
2. **Multi-Metric Evaluation**: Tracks both macro F1 and per-class scores
3. **Progressive Unfreezing**: Balances transfer learning and fine-tuning
4. **Kaggle-Optimized**: Ready for GPU execution with memory growth configuration


## How to run:
```bash
jupyter nbconvert --to notebook --execute notebooks/preprocessing.ipynb
jupyter nbconvert --to notebook --execute notebooks/training.ipynb
jupyter nbconvert --to notebook --execute notebooks/postprocessing.ipynb

```