"""
Team Name: Creatix
Team Members: Siddharth Malkania, Krishan Verma , Rishi Mehrotra
Leaderboard Rank: 117

"""

def preprocessing():
    """
    Enhanced data preparation for soil classification targeting F1 ≥ 0.95
    Based on soil3.ipynb success with optimizations for Clay soil performance
    """
    print("Starting enhanced soil classification preprocessing...")
    
    # Import required libraries
    import os
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.model_selection import train_test_split
    from PIL import Image
    import shutil
    
    # Configuration parameters
    IMG_SIZE = 224
    BATCH_SIZE = 64
    
    # Paths
    TRAIN_DIR = '/kaggle/input/soilcl/soil_classification-2025/train'
    TEST_DIR = '/kaggle/input/soilcl/soil_classification-2025/test'
    TRAIN_CSV = '/kaggle/input/soilcl/soil_classification-2025/train_labels.csv'
    TEST_CSV = '/kaggle/input/soilcl/soil_classification-2025/test_ids.csv'
    PROCESSED_TRAIN_DIR = '/kaggle/working/train'
    PROCESSED_TEST_DIR = '/kaggle/working/test'
    
    # Create directories
    os.makedirs(PROCESSED_TRAIN_DIR, exist_ok=True)
    os.makedirs(PROCESSED_TEST_DIR, exist_ok=True)
    
    def convert_to_jpg(source_dir, target_dir, file_mapping=None):
        """Fast image conversion and format standardization"""
        if file_mapping is None:
            file_mapping = {}
        
        for filename in os.listdir(source_dir):
            source_path = os.path.join(source_dir, filename)
            
            if not os.path.isfile(source_path):
                continue
            
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext in ['.jpg', '.jpeg']:
                target_path = os.path.join(target_dir, filename)
                shutil.copy2(source_path, target_path)
                file_mapping[filename] = filename
            else:
                try:
                    new_filename = os.path.splitext(filename)[0] + '.jpg'
                    target_path = os.path.join(target_dir, new_filename)
                    
                    with Image.open(source_path) as img:
                        img = img.convert('RGB')
                        img.save(target_path, 'JPEG', quality=95)
                    
                    file_mapping[filename] = new_filename
                except Exception as e:
                    print(f"Error converting {filename}: {e}")
                    try:
                        target_path = os.path.join(target_dir, filename)
                        shutil.copy2(source_path, target_path)
                        file_mapping[filename] = filename
                    except:
                        print(f"Could not process {filename}")
        
        return file_mapping
    
    # Image format conversion
    print("Converting images to JPG format...")
    train_file_mapping = convert_to_jpg(TRAIN_DIR, PROCESSED_TRAIN_DIR)
    test_file_mapping = convert_to_jpg(TEST_DIR, PROCESSED_TEST_DIR)
    
    # Load CSV files
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    # Map processed filenames
    train_df['processed_image_id'] = train_df['image_id'].map(
        lambda x: train_file_mapping.get(x, x))
    test_df['processed_image_id'] = test_df['image_id'].map(
        lambda x: test_file_mapping.get(x, x))
    
    # Train-validation split with stratification
    train_data, val_data = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df['soil_type'])
    
    print("Training class distribution:")
    print(train_data['soil_type'].value_counts())
    
    # Enhanced data augmentation for improved performance
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest',
        channel_shift_range=0.1
    )
    
    # Validation and test data generators (no augmentation)
    valid_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Dataset repetition for better class balance (especially Clay soil)
    repeated_train_data = train_data.loc[np.repeat(train_data.index.values, 3)]
    
    # Create data generators
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=repeated_train_data,
        directory=PROCESSED_TRAIN_DIR,
        x_col='processed_image_id',
        y_col='soil_type',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=val_data,
        directory=PROCESSED_TRAIN_DIR,
        x_col='processed_image_id',
        y_col='soil_type',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=PROCESSED_TEST_DIR,
        x_col='processed_image_id',
        y_col=None,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False
    )
    
    # Enhanced class weights targeting Clay soil improvement
    class_weights = {}
    total_samples = len(train_data)
    soil_counts = train_data['soil_type'].value_counts()
    
    for i, soil_type in enumerate(train_generator.class_indices):
        count = soil_counts.get(soil_type, 0)
        if count > 0:
            class_weights[i] = (1 / count) * (total_samples / len(soil_counts))
            # Special boost for underrepresented classes
            if soil_type == 'Clay soil':
                class_weights[i] *= 3.0  # Triple boost for Clay soil
            elif soil_type == 'Black Soil':
                class_weights[i] *= 1.8  # Enhanced boost for Black Soil
    
    print(f"Class indices: {train_generator.class_indices}")
    print(f"Enhanced class weights: {class_weights}")
    
    return train_generator, valid_generator, test_generator, train_data, val_data, test_df, class_weights
