import pickle
import CNN_preprocessing
import DTree

# Load the saved model from the file
with open("tree_CNN_model.pkl", "rb") as f:
    tree_CNN_loaded = pickle.load(f)

name_image_dict = {
    

} # name : path of his base image

def predict(image_path):
    features= CNN_preprocessing.extract_features(image_path)
    predicted_name = DTree.predict(tree_CNN_loaded,[features])
    predicted_image_path= None
    return int(predicted_name[0]), predicted_image_path

if __name__ == "__main__":
    print(predict("testimage.jpg"))