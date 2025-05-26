"""

Team Name: Creatix
Team Members: Siddharth Malkania, Krishan Verma , Rishi Mehrotra
Leaderboard Rank: 117

"""

# Here you add all the post-processing related details for the task completed from Kaggle.

def postprocessing():
    """
    Enhanced postprocessing for soil classification targeting F1 ≥ 0.95
    Based on soil5-1.ipynb success with comprehensive evaluation and submission generation
    """
    print("Starting enhanced soil classification postprocessing...")
    
    # Import required libraries
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    
    def evaluate_enhanced_model(model, valid_generator, class_indices):
        """Enhanced evaluation targeting F1 ≥ 0.95"""
        valid_generator.reset()
        y_pred_probs = model.predict(valid_generator, steps=int(np.ceil(valid_generator.samples/64)))
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = valid_generator.classes
        
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_individual = f1_score(y_true, y_pred, average=None)
        
        idx_to_class = {v: k for k, v in class_indices.items()}
        class_f1_scores = {idx_to_class[i]: score for i, score in enumerate(f1_individual)}
        
        print(f"Macro F1 Score: {f1_macro:.4f}")
        print("Individual F1 Scores:")
        for name, score in class_f1_scores.items():
            print(f" {name}: {score:.4f}")
        
        if f1_macro >= 0.95:
            print("TARGET ACHIEVED: F1 Score ≥ 0.95!")
        else:
            print(f"Progress: {f1_macro:.4f}/0.95 ({(f1_macro/0.95)*100:.1f}%)")
        
        return f1_macro
    
    def generate_test_predictions(model, test_generator, test_df, class_indices):
        """Generate predictions for test set and create submission file"""
        print("Generating test predictions...")
        test_generator.reset()
        test_preds = model.predict(test_generator, steps=int(np.ceil(test_generator.samples/64)))
        test_classes = np.argmax(test_preds, axis=1)
        
        # Create submission
        idx_to_class = {v: k for k, v in class_indices.items()}
        test_class_names = [idx_to_class[idx] for idx in test_classes]
        
        submission_df = pd.DataFrame({
            'image_id': test_df['image_id'],
            'soil_type': test_class_names
        })
        
        submission_df.to_csv('enhanced_f1_95_submission.csv', index=False)
        
        return submission_df
    
    def execute_postprocessing(model, valid_generator, test_generator, test_df, class_indices):
        """Execute complete postprocessing workflow from soil5-1.ipynb"""
        
        # Evaluate model
        print("Evaluating enhanced model...")
        f1_score_result = evaluate_enhanced_model(model, valid_generator, class_indices)
        
        # Generate test predictions
        submission_df = generate_test_predictions(model, test_generator, test_df, class_indices)
        
        # Final results summary
        if f1_score_result >= 0.95:
            print(f"SUCCESS! F1 Score: {f1_score_result:.4f} ≥ 0.95")
        else:
            print(f"Result: F1 Score: {f1_score_result:.4f} (Target: 0.95)")
        
        print("Enhanced submission saved as: enhanced_f1_95_submission.csv")
        
        return f1_score_result, submission_df
    
    return execute_postprocessing

