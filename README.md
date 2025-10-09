# CloudClassifier
Of course. Here is a comprehensive README file for your GitHub project. You can copy and paste this into a file named `README.md` in your project's main directory.

-----

# ☁️ Real-Time Cloud Classifier

[](https://www.python.org/downloads/)
[](https://www.tensorflow.org/)
[](https://gradio.app/)
[](https://huggingface.co/spaces)

A deep learning-powered web application that classifies cloud types in real-time from a webcam feed or uploaded image. This project uses a TensorFlow/Keras model and is deployed as an interactive web app using Gradio on Hugging Face Spaces.

-----

## ✨ Features

  * **Real-Time Classification**: Classify clouds instantly using your device's webcam.
  * **Image Upload**: Upload an image file for classification.
  * **7 Cloud Types**: Trained to recognize 7 distinct cloud categories.
  * **Confidence Thresholding**: Rejects images that are not confidently identified as clouds, displaying a "No Cloud Detected" message.
  * **Image Preprocessing**: Automatically enhances image contrast using CLAHE for better feature detection before prediction.
  * **Interactive UI**: Simple and intuitive web interface powered by Gradio.

-----

## 🧠 Model Details

  * **Architecture**: The model uses **Transfer Learning** with the **MobileNetV2** architecture, pre-trained on the ImageNet dataset and fine-tuned for cloud classification.
  * **Dataset**: The model was trained on the ["Clouds Photos" dataset from Kaggle](https://www.kaggle.com/datasets/jockeroika/clouds-photos).
  * **Classes**: The model is trained to classify the following 7 types of clouds:
    1.  `Cirriform`
    2.  `Clear Sky`
    3.  `Cumuliform`
    4.  `Cumulonimbus`
    5.  `Cumulus`
    6.  `Stratiform`
    7.  `Stratocumulus`

-----

## 🛠️ Tech Stack

  * **Backend & Model**: TensorFlow, Keras, OpenCV, NumPy
  * **Web Interface**: Gradio
  * **Hosting**: Hugging Face Spaces

-----

## ⚙️ How to Run Locally

To run this application on your own machine, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YourUsername/YourRepoName.git
    cd YourRepoName
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the model file:**
    Ensure you have the trained model file (e.g., `cloud_classifier.keras`) in the main project directory.

5.  **Run the application:**

    ```bash
    python app.py
    ```

    The application will now be running on your local server (usually at `http://127.0.0.1:7860`).

-----

## 📁 File Structure

```
.
├── 📄 app.py                # The main Gradio application script
├── 📄 requirements.txt      # Python dependencies for the project
├── 🧠 cloud_classifier.keras # The trained Keras model file
└── 📁 examples/             # Optional folder with example images
    └── 🖼️ cumulus.jpg
```

-----

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
