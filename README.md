# multi_class_fish_images

🐟 Multiclass Fish Image Classification
-
🎯 Overview
-

This project focuses on classifying multiple fish species using Deep Learning and Transfer Learning.
A user-friendly Streamlit web app allows real-time fish image classification, visualization of model metrics, and easy interaction.

🧠 Objectives
-

🔹 Classify fish images into their respective species.

🔹 Evaluate and compare multiple CNN architectures.

🔹 Build and deploy a Streamlit web application for real-time predictions.

🔹 Present model insights — accuracy, confusion matrix, and key metrics.

🧰 Tech Stack
-
| Category                | Tools / Libraries                                       |
| ----------------------- | ------------------------------------------------------- |
| Programming Language    | Python                                                  |
| Deep Learning Framework | TensorFlow, Keras                                       |
| Web Application         | Streamlit                                               |
| Model Architectures     | MobileNet, VGG16, ResNet50, InceptionV3, EfficientNetB0 |
| Data Processing         | NumPy, Pandas                                           |
| Visualization           | Matplotlib, Seaborn                                     |
| Metrics                 | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |


🗂 Dataset
-
Dataset contains multiple fish species, each stored in separate folders.

Loaded using TensorFlow’s image_dataset_from_directory.

Preprocessing Steps:

Resize all images to 224×224 pixels

Normalize pixel values between 0 and 1

Apply data augmentation — rotation, zoom, and flipping

⚙️ Project Workflow
-
1️⃣ Data Preprocessing
-
Normalized images and applied augmentation to reduce overfitting.

2️⃣ Model Training
-
Trained a basic CNN and several Transfer Learning models:

VGG16

ResNet50

MobileNet

InceptionV3

EfficientNetB0

Fine-tuned MobileNet as the final model.

Saved the best-performing model as mobilenet_fish_final.keras.

3️⃣ Model Evaluation
-
Compared all models based on accuracy, loss, and F1-score.

Visualized training curves and confusion matrix for deeper insights.

4️⃣ Deployment
-
Created a Streamlit dashboard with the following sections:

🏠 Home: Introduction and overview

📂 Upload Image: Upload a fish image

🐟 Classify: Predict fish species

📊 Model Insights: View metrics and confusion matrix

ℹ️ About: Advantages, challenges, and future work

🏆 Model Performance
-
| Model          | Validation Accuracy | Validation Loss |
| -------------- | ------------------- | --------------- |
| CNN (Scratch)  | 0.864               | 0.388           |
| VGG16          | 0.786               | 0.933           |
| ResNet50       | 0.319               | 1.961           |
| **MobileNet**  | **0.987**           | **0.050**       |
| InceptionV3    | 0.961               | 0.130           |
| EfficientNetB0 | 0.171               | 2.309           |


Best Model: 🏅 MobileNet
Final Accuracy: 98.7%
Final Loss: 0.05

🖥️ Streamlit App Highlights
-
✅ Real-time image upload and fish species prediction
📊 Displays class probabilities for all categories
🧠 Uses MobileNet (Transfer Learning) for fast and accurate results
🌈 Clean and interactive interface with sidebar navigation


🔮 Future Enhancements
-
📈 Train on larger and more diverse datasets

📱 Develop a mobile version for on-the-go predictions

☁ Deploy model on AWS / GCP / Azure

🎥 Add live webcam fish detection

🧩 Experiment with Vision Transformers (ViT)

⚠️ Challenges & Solutions
-
| Challenge           | Solution                                                 |
| ------------------- | -------------------------------------------------------- |
| Imbalanced dataset  | Applied augmentation and class weights                   |
| Varying image sizes | Standardized to 224×224                                  |
| Overfitting         | Used dropout, batch normalization, and transfer learning |
| Slow training       | Switched to lightweight **MobileNet**                    |


🌍 Real-World Applications
-
🌊 Marine Biology: Automate fish species identification

🎣 Fisheries Management: Support sustainable aquaculture

🏪 Food Industry: Classify seafood species

🎓 Education & Research: Assist AI-based biological studies

📘 Skills Gained
-
Deep Learning & CNNs

Transfer Learning

Data Preprocessing & Augmentation

TensorFlow / Keras Modeling

Model Evaluation & Visualization

Streamlit Web App Deployment

🧾 Deliverables
-
✅ Trained model → mobilenet_fish_final.keras

✅ Streamlit app → app.py

✅ Jupyter notebook → Multiclass Fish Image Classification.ipynb

✅ Project report → Multiclass Fish Image Classification.pdf

✅ README documentation

👩‍💻 Author
-
kasi rajan
💡 Data Science Enthusiast | Deep Learning | Model Deployment | AI for Sustainability
