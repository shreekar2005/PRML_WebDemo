from flask import Flask, render_template, request
import CNN_DTree
import LBP_DTree
import os

app=Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image file part", 400

        file = request.files['image']

        if file.filename == '':
            return "No selected file", 400

        # Save the image to the upload folder
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return file_path

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/process_input', methods=['GET','POST'])
def process_input():
    feature_extraction_method =request.form['feature_extraction_method']
    image_path=save_image()
    predicted_name = None
    predicted_image_path = None
    if feature_extraction_method == 'CNN_extraction':
        predicted_name, predicted_image_path= CNN_DTree.predict(image_path)
    elif feature_extraction_method == 'LBP_extraction':
        predicted_name, predicted_image_path= LBP_DTree.predict(image_path)
    else :
        print("this cannot happen!")
    image_path="uploads/"+image_path[15:]
    print(image_path, predicted_image_path)
    return render_template('result.html',input_image=image_path, predicted_name=predicted_name, predicted_image=predicted_image_path) 


if __name__ == "__main__":
    app.run(debug=True, threaded=True)