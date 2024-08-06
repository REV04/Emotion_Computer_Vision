# Emotion_Computer_Vision

This project, Emotion Computer Vision, is a part of my Hacktiv8 coursework. I chose this topic to deepen my skills in image processing, expanding beyond my experience in text processing. My other personal project can be found at the [Richard Edgina Virgo](https://github.com/REV04)

#### -- Project Status: Finished

## Project Intro/Objective

The purpose of this project is to classify emotions based on facial images. I sourced the emotion dataset from Kaggle (https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer). After loading the data into Google Colab, I split it for modeling purposes. Using this image data, I performed Exploratory Data Analysis (EDA) and trained a model using an Artificial Neural Network (ANN) sequentially. To enhance the model's performance, I applied data augmentation and further improved it through transfer learning. I created this project with a fictional background, made solely for the purpose of the project. The background scenario is about enhancing the gaming experience for customers, as a gaming company aims to integrate emotion recognition technology into their games.

### Methods Used

- Artificial Neural Network
- Computer Vision
- Data Visualization
- Deployment

### Technologies

- Python

## Project Description

The dataset used for creating this model is sourced from Kaggle (https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer). This dataset contains images categorized into seven emotions: angry, disgusted, fearful, happy, neutral, sad, and surprised. The model built using this dataset achieved the following results:

- Accuracy before using transfer learning: 0.4224
- Accuracy after using transfer learning with DenseNet: 0.5861

The model shows weakness in classifying emotions with a limited amount of data. We developed this model to gain insights from discussions using computer vision. The most significant challenge before training the model was the limited data, which impacted the model's performance.

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Exploratory Data Analysis (EDA), Model Training, Model Improvement, Model Evaluation, and summary are being kept in Model/Emotion_Computer_Vision.ipynb
3. Model Inference is being kept in Model/Emotion_Computer_Vision_Inference.ipynb
4. url which contain important link such as hugging face, model, and kaggle link are being kept in Model/url.txt
5. Deployment which contain hugging face deploy such as eda.py, app.py, prediction.py, requirements.txt, and emotion image each class are being kept in deployment folder.

## Featured Notebooks/Analysis/Deliverables

- [Hungging Face for model usage demonstration](https://huggingface.co/spaces/REV04/Emotion)

## Contact

- If you have any question or want to contribute with this project, feel free to ask me in [linkedin](https://www.linkedin.com/in/richard-edgina-virgo-a7435319b/).
