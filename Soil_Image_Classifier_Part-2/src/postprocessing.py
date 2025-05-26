import numpy as np
import pandas as pd

class Postprocessor:
    @staticmethod
    def optimize_ratio(predictions, target=0.44):
        current = np.mean(predictions)
        if abs(current - target) > 0.02:
            preds_array = np.array(predictions)
            n_samples = len(preds_array)
            target_ones = int(target * n_samples)
            current_ones = np.sum(preds_array)

            if current_ones < target_ones:
                zero_indices = np.where(preds_array == 0)[0]
                n_to_flip = min(target_ones - current_ones, len(zero_indices))
                if n_to_flip > 0:
                    preds_array[np.random.choice(zero_indices, n_to_flip)] = 1
            else:
                one_indices = np.where(preds_array == 1)[0]
                n_to_flip = min(current_ones - target_ones, len(one_indices))
                if n_to_flip > 0:
                    preds_array[np.random.choice(one_indices, n_to_flip)] = 0
                    
            return preds_array.tolist()
        return predictions

    @staticmethod
    def create_submission(image_ids, predictions, output_path):
        df = pd.DataFrame({'image_id': image_ids, 'label': predictions})
        df.to_csv(output_path, index=False)
        return df