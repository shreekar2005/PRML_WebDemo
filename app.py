from flask import Flask, render_template, request, redirect, url_for
import CNN_ANN as CNN_ANN
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
    image_path=save_image()
    predicted_name = None
    predicted_image_path = None
    predicted_name, predicted_image_path= CNN_ANN.predict(image_path)
    image_path="uploads/"+image_path[15:]
    print(image_path, predicted_image_path)
    return render_template('result.html',input_image=image_path, predicted_name=predicted_name, predicted_image=predicted_image_path) 

@app.route('/clear_and_try_another')
def clear_and_try_another():
    # Get the image_path from the query parameter
    image_path = request.args.get('image_path')
    image_path = "static/"+image_path
    # Make sure the path is safe (this is important to avoid security risks)
    if image_path and os.path.exists(image_path):
        try:
            # Delete the image at the specified path
            os.remove(image_path)
            print(f"Image {image_path} deleted successfully.")
        except Exception as e:
            print(f"Error deleting image: {e}")
    
    # Redirect the user to another page (e.g., the home page or upload page)
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, threaded=True, processes=1)
