import DecisionTree.DTree as DTree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import pickle

print("libraries loaded!! ")

def load_data(path):
    url_filtered_CNN_features_dataset = path #for CNN limited
    df_CNN = pd.read_csv(url_filtered_CNN_features_dataset) # reading url for extracted CNN_features_dataset_limited.csv
    df_CNN.drop('Unnamed: 0', axis=1, inplace=True)

    #dropping those labels whose number of datapoints are less than 80
    # Get the counts of each label
    label_counts = df_CNN['2048'].value_counts()

    # Filter out labels with counts less than 80
    labels_to_keep = label_counts[label_counts >= 80].index

    # Filter the DataFrame
    df_CNN = df_CNN[df_CNN['2048'].isin(labels_to_keep)]

    df_CNN = df_CNN.groupby('2048').apply(lambda x: x.sample(min(len(x), 190), random_state=42)).reset_index(drop=True)
    # limiting dataset to have max 105 images pre prson


    # Separate features and labels
    X = df_CNN.iloc[:, :-1]
    y = df_CNN.iloc[:, -1]

    # Encode labels (alphabetically)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    # encoding is benificial as working on numbers is lot easier than working on string

    # Ensure stratified split of atleast (64 training, 16 testing per class)
    X_train_CNN, X_test_CNN, y_train_CNN, y_test_CNN = train_test_split(X, y_encoded, test_size=1/5, random_state=42, stratify=y_encoded)

    # Normalize features
    scaler = StandardScaler()
    X_train_CNN = scaler.fit_transform(X_train_CNN)
    X_test_CNN = scaler.transform(X_test_CNN)

    print(f"Dataset size: {df_CNN.shape}")
    print(f"After applying LDA => Training size: {X_train_CNN.shape}, Testing size: {X_test_CNN.shape}")

    # applying LDA and splitting test into test and val datatypes.
    # Apply LDA with at most (number of classes - 1) components
    lda = LDA(n_components=min(500, len(set(y_train_CNN)) - 1))
    lda.fit_transform(X_train_CNN, y_train_CNN)

    X_train_CNN = lda.transform(X_train_CNN)
    X_test_CNN = lda.transform(X_test_CNN)

    print(f"After applying LDA => Training size: {X_train_CNN.shape}, Testing size: {X_test_CNN.shape}")

    return X_train_CNN, y_train_CNN, X_test_CNN, y_test_CNN

if __name__ == "__main__":
    url ='https://raw.githubusercontent.com/AgarwalMayank2/Face_Detection/refs/heads/main/processed_dataset/filtered_CNN_features_dataset.csv'
    X_train_CNN, y_train_CNN, X_test_CNN, y_test_CNN = load_data(url)
    tree_CNN = DTree.DecisionTree(max_depth=6)
    tree_CNN.train(X_train_CNN, y_train_CNN)

    y_pred_CNN = DTree.predict(tree_CNN, X_test_CNN)

    accuracy_CNN = np.mean(y_pred_CNN == y_test_CNN)
    print(f"Test Accuracy for CNN: {accuracy_CNN:.2f}")

    y_pred_train_CNN = DTree.predict(tree_CNN, X_train_CNN)
    accuracy_train_CNN = np.mean(y_pred_train_CNN == y_train_CNN)
    print(f"Train Accuracy for CNN: {accuracy_train_CNN:.2f}")
    with open("tree_CNN_model.pkl", "wb") as f:
        pickle.dump(tree_CNN, f)





