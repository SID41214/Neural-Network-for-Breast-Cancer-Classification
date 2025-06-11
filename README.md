# Neural Network for Breast Cancer Classification

A machine learning project that utilizes a neural network to classify breast cancer tumors as either malignant or benign based on the Wisconsin Breast Cancer dataset.


## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Technologies and Libraries](#-technologies-and-libraries)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results and Evaluation](#-results-and-evaluation)
- [Contributing](#-contributing)
- [License](#-license)



## 📖 Overview

This project aims to build and train a neural network model to accurately classify breast cancer diagnoses. By leveraging a well-known medical dataset, this model can serve as a powerful tool in early cancer detection. The primary goal is to achieve high accuracy and other relevant performance metrics, demonstrating the effectiveness of neural networks for medical diagnosis tasks.



## 📊 Dataset

The model is trained and evaluated on the **Breast Cancer Wisconsin (Diagnostic) Dataset**. This is a classic dataset used for binary classification tasks in machine learning.

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Number of Instances:** 569
- **Number of Features:** 30 real-valued, positive features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
- **Classes:** 2 (Malignant, Benign)
- **Class Distribution:** 212 Malignant, 357 Benign

The features describe characteristics of the cell nuclei present in the image, such as radius, texture, perimeter, and area.



## 🧠 Methodology

The project follows a standard machine learning workflow:

1.  **Data Loading:** The dataset is loaded using the `pandas` library.
2.  **Data Preprocessing:**
    * The features and target variable are separated.
    * The data is scaled using `StandardScaler` to ensure all features have a similar scale, which is crucial for the optimal performance of neural networks.
3.  **Data Splitting:** The dataset is split into training and testing sets to evaluate the model's performance on unseen data.
4.  **Model Building:** A sequential neural network is constructed using the `Keras` library with `TensorFlow` as the backend. The model architecture consists of an input layer, one or more hidden layers with `ReLU` activation functions, and an output layer with a `sigmoid` activation function for binary classification.
5.  **Model Compilation:** The model is compiled with the `Adam` optimizer and `binary_crossentropy` as the loss function, which is suitable for binary classification problems.
6.  **Model Training:** The neural network is trained on the training data for a specified number of epochs.
7.  **Model Evaluation:** The trained model's performance is evaluated on the test set using various metrics.



## 🛠️ Technologies and Libraries

This project is implemented in Python and utilizes the following libraries:

- **[TensorFlow](https://www.tensorflow.org/)**: An end-to-end open-source platform for machine learning.
- **[Keras](https://keras.io/)**: A high-level neural networks API, running on top of TensorFlow.
- **[Scikit-learn](https://scikit-learn.org/)**: For data preprocessing (e.g., `StandardScaler`, `train_test_split`) and model evaluation.
- **[Pandas](https://pandas.pydata.org/)**: For data manipulation and analysis.
- **[NumPy](https://numpy.org/)**: For numerical operations.
- **[Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/)**: For data visualization and plotting the results.



## ⚙️ Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SID41214/Neural-Network-for-Breast-Cancer-Classification.git](https://github.com/SID41214/Neural-Network-for-Breast-Cancer-Classification.git)
    cd Neural-Network-for-Breast-Cancer-Classification
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You may need to create a `requirements.txt` file by running `pip freeze > requirements.txt` in your local project environment after installing the necessary libraries.)*



## 🚀 Usage

Once the installation is complete, you can run the main script (e.g., `main.py` or a Jupyter Notebook) to train the model and see the results.

```bash
# If it's a .py file
python your_main_script_name.py

# If it's a Jupyter Notebook
jupyter notebook your_notebook_name.ipynb
```

The script will load the data, preprocess it, build, train, and evaluate the neural network, printing the performance metrics at the end.


## 📈 Results and Evaluation

The model's performance is evaluated using the following metrics:

- **Accuracy:** The proportion of correctly classified instances.
- **Precision:** The ability of the classifier not to label a negative sample as positive.
- **Recall (Sensitivity):** The ability of the classifier to find all the positive samples.
- **F1-Score:** The weighted average of Precision and Recall.
- **Confusion Matrix:** A table that visualizes the performance of the classification model.



| Metric    | Score |
| :-------- | :---- |
| Accuracy  | 98.5% |
| Precision | 97.9% |
| Recall    | 98.2% |
| F1-Score  | 98.0% |

A confusion matrix and other visualizations are also generated to provide a more in-depth understanding of the model's performance.



## 🤝 Contributing

Contributions are what make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion to improve this project, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request
