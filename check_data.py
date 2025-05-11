import numpy as np
import sys
import pickle

def check_dataset(file_path):
    print(f"Checking dataset: {file_path}")
    if file_path.endswith('.npz'):
        data = np.load(file_path)
        X = data['X']
        y = data['y']
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        return X.shape
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model type: {type(model)}")
        if hasattr(model, 'model'):
            print(f"Model internal type: {type(model.model)}")
        if hasattr(model, 'get_params'):
            params = model.get_params()
            print(f"Model parameters: {params}")
        return None
    else:
        print(f"Unsupported file format: {file_path}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_dataset(sys.argv[1])
    else:
        print("Please provide a dataset path")
