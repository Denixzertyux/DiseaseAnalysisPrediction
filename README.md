# Disease Prediction System from Symptoms 🩺

An end-to-end machine learning project that predicts potential diseases based on patient symptoms. This system employs an ensemble of classification models (SVM, Naive Bayes, and Random Forest) to provide a robust final prediction. The project demonstrates a complete ML workflow, from data preprocessing and handling class imbalance with `RandomOverSampler` to model evaluation and creating a final predictive function.



---

## 📋 Table of Contents

- [About The Project](#about-the-project)
  - [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the Script](#running-the-script)
  - [Making a Prediction](#making-a-prediction)
- [Methodology](#methodology)
- [Results](#results)
- [Dataset](#dataset)
- [License](#license)

---

## About The Project

This project builds a multi-model system for disease prediction. Given a set of symptoms, it aims to identify the most likely disease. A significant challenge in medical datasets is class imbalance (some diseases are far more common than others), which this project addresses by using the **RandomOverSampler** technique to create a balanced training set.

The core of the system is a voting ensemble of three powerful classifiers:
* **Support Vector Machine (SVM)**
* **Gaussian Naive Bayes (NB)**
* **Random Forest (RF)**

The final prediction is determined by a majority vote from these three models, leveraging their diverse strengths to improve overall accuracy and reliability.

### Built With

* [Python](https://www.python.org/)
* [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) for data manipulation
* [Scikit-learn](https://scikit-learn.org/) for modeling and evaluation
* [Imbalanced-learn](https://imbalanced-learn.org/stable/) for handling class imbalance
* [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) for data visualization

---

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

You need **Python 3.x** and **pip** installed on your system.

### Installation

1.  **Clone the repository**
    ```sh
    git clone [https://github.com/](https://github.com/)[your-username]/[your-repo-name].git
    ```
2.  **Navigate to the project directory**
    ```sh
    cd [your-repo-name]
    ```
3.  **Install the required packages**
    (It's highly recommended to use a virtual environment)
    ```sh
    pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn
    ```
4.  **Add the dataset**
    Place the `improved_disease_dataset.csv` file in the root directory of the project.

---

## Usage

### Running the Script

To execute the full analysis, model training, and evaluation pipeline, run the script from your terminal:

```sh
python your_script_name.py
