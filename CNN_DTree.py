import pickle
import CNN_preprocessing
import DTree

# Load the saved model from the file
with open("tree_CNN_model.pkl", "rb") as f:
    tree_CNN_loaded = pickle.load(f)

name_image_dict = {
    0:"Colin_Powell",
    1:"Donald_Rumsfeld",
    2:"George_W_Bush",
    3:"Gerhard_Schroeder",
    4:"Tony_Blair"

} # name : path of his base image

def predict(image_path):
    features= CNN_preprocessing.extract_features(image_path).reshape(-1,1)
    print(features)
    prediction = DTree.predict(tree_CNN_loaded,[features])
    print(prediction)
    predicted_name = name_image_dict[int(prediction[0])]
    predicted_image_path= "base_images/"+predicted_name+".jpg"
    return predicted_name, predicted_image_path

if __name__ == "__main__":
    print()
    print(predict("Tony_Blair_0109.jpg"))