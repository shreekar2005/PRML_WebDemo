import ANN_model as ANN_model
import CNN_preprocessing as CNN_preprocessing
import numpy as np
import pandas as pd
import joblib

scaler = joblib.load('scaler.pkl')
label = joblib.load('label_enc.pkl')

load_params = np.load("ann_params.npy", allow_pickle=True).item()

nn_loaded = ANN_model.NeuralNetwork(load_params["layers"], activations=load_params["activation"])

nn_loaded.weights = load_params["weights"]
nn_loaded.biases = load_params["biases"]

def predict(image_path):
    features= CNN_preprocessing.extract_features(image_path).reshape(-1,1)
    features = np.array(features).reshape(1,-1)
    print(features)
    predictions = nn_loaded.predict(features)
    original_label = label.inverse_transform(predictions)
    print(original_label[0])
    predicted_name = original_label[0]
    predicted_image_path= "base_images/"+predicted_name+".jpg"
    return predicted_name, predicted_image_path

if __name__ == "__main__":
    print(predict("Tony_Blair_0109.jpg")) 
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    # Folder path
    base_path = "C:/Users/shree/Downloads/atleast_80_images/"
    names = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]

    # Create subplots
    n = len(names)
    cols = 3  # Number of columns in the subplot grid
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for idx, name in enumerate(names):
        path = os.path.join(base_path, name)
        images = os.listdir(path)
        predicted_names = []

        for image in images:
            pred_name, _ = predict(os.path.join(path, image))  # Assumes predict returns (label, confidence)
            short_name = pred_name.split('_')[0]  # Get first name only
            predicted_names.append(short_name)

        unique_names, counts = np.unique(predicted_names, return_counts=True)

        ax = axes[idx]
        ax.bar(unique_names, counts)
        ax.set_xticks(range(len(unique_names)))
        ax.set_xticklabels(unique_names, rotation=90)
        ax.set_xlabel('Predicted Name')
        ax.set_ylabel('Count')
        ax.set_title(f"{name} ({len(images)} test images)")

    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


