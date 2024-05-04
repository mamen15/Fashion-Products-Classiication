from flask import Flask, request, jsonify
from PIL import Image
import os
import torch
import torch.nn as nn
from torchvision import models, transforms

PATH = r"C:\Users\21263\cache"
os.environ['HF_HOME'] =  r"C:\Users\21263\cache/HUGGINGFACE"
os.environ['TORCH_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH

# Define the preprocess_image function
def preprocess_image(image):
    image = Image.open(image)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Define the predict_class function
def predict_class(image, product_categories):
    # Set the model to evaluation mode
    model.eval()
    
    # Pass the preprocessed image through the model
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    predicted_class = product_categories[predicted.item()]
    
    return predicted_class

# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
num_classes = 10  # Assuming you have 10 classes
model.fc = nn.Linear(num_features, num_classes)

# Load the fine-tuned model
model.load_state_dict(torch.load( 'fashion_classifier.pth' , map_location=torch.device('cpu')))
model.eval()

# Define the product categories
product_categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Flask app initialization and route definition
app = Flask(__name__)


# HTML code
index_html = '''
<!DOCTYPE html>
<html>
<head>
<title>Fashion Product Classifier</title>
<style>
/* CSS code */
body {
    font-family: Arial, sans-serif;
    text-align: center;
}
h1 {
    margin-top: 30px;
}
form {
    margin-top: 20px;
}
#resultContainer {
    margin-top: 20px;
}
</style>
</head>
<body>
<h1>Fashion Product Classifier</h1>
<form id="uploadForm" enctype="multipart/form-data">
<input type="file" name="image" accept="image/*" required>
<button type="submit">Classify</button>
</form>
<div id="resultContainer"></div>
<script>
// JavaScript code
document.getElementById('uploadForm').addEventListener('submit',
function(e) {
    e.preventDefault();
    var formData = new FormData();
    var fileInput = document.querySelector('input[type="file"]');
    formData.append('image', fileInput.files[0]);
    // Make the API request to the Flask backend
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Display the classification result
        var resultContainer = document.getElementById('resultContainer');
        resultContainer.innerHTML = '<p>Predicted Class: ' +
        data.predicted_class + '</p>';
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
</script>
</body>
</html>
'''

@app.route('/')
def index():
    return index_html

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'})

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected image'})

    # Preprocess the image
    image = preprocess_image(image_file)
    
    # Predict the class
    predicted_class = predict_class(image, product_categories)

    return jsonify({'predicted_class': predicted_class})


if __name__ == '__main__':
    app.run(debug=True)
