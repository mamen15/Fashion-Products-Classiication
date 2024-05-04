Fashion Product Classifier - README.md

This project presents a simple web application for classifying fashion product images into predefined categories. It leverages the Flask framework for the backend and a pre-trained ResNet-18 model for image classification.

Project Overview:
The application allows users to upload an image of a fashion product, and it predicts the product's category based on the trained model. The possible categories include:
    T-shirt/top
    Trouser
    Pullover
    Dress
    Coat
    Sandal
    Shirt
    Sneaker
    Bag
    Ankle boot

Running the Application Locally

Requirements:
Python 3.6 or later
Flask
Pillow (PIL)
PyTorch
torchvision

Steps:
Clone or download this repository.
Install the required libraries: pip install flask pillow torch torchvision
Download the pre-trained model fashion_classifier.pth and place it in the project directory.
Run the application: python app.py
Open your web browser and navigate to http://127.0.0.1:5000/.
Upload an image and click "Classify" to see the predicted category.

Additional Information
The ResNet-18 model used in this project is pre-trained on the ImageNet dataset and fine-tuned on a fashion product dataset.
The preprocess_image function performs necessary image transformations before feeding it to the model.
The predict_class function utilizes the model to predict the class and returns the corresponding category label.
The Flask app defines two routes:
    /: Serves the HTML page with the image upload form and result display area.
    /predict: Handles image upload, prediction, and returns the predicted class as a JSON response.