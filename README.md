# IMDBReviewClassifier


This project performs sentiment analysis on IMDB movie reviews using **Random Forest** and **Naive Bayes** algorithms. It aims to classify movie reviews as positive or negative, utilizing both custom implementations and the `scikit-learn` library for comparison.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage Instructions](#usage-instructions)
- [Algorithms Implemented](#algorithms-implemented)
- [Model Evaluation](#model-evaluation)
- [Visualization](#visualization)
- [Authors](#authors)
- [License](#license)

## Overview

The **IMDB Review Classifier** project is a sentiment analysis application that processes IMDB movie reviews, classifying them as either positive or negative. This is accomplished using custom implementations of **Random Forest** and **Naive Bayes** algorithms, and comparing them to models provided by the `scikit-learn` library.

## Features

- **Text Processing**: Converts movie reviews into binary feature vectors based on the presence of words.
- **Custom Implementations**: Implements Random Forest and Naive Bayes algorithms from scratch.
- **Scikit-learn Comparison**: Evaluates the custom implementations against the `scikit-learn` models.
- **Model Evaluation**: Computes metrics like accuracy, precision, recall, and F1 score.
- **Visualization**: Plots learning curves for both custom and library models.

## Installation

### Prerequisites

- Python 3.x
- Libraries:
  - NumPy
  - Matplotlib
  - Scikit-learn
  - NLTK

### Setup

Install the required libraries using:
```bash
pip install numpy matplotlib scikit-learn nltk
```

## Usage Instructions

1. **Load the Data**: Ensure that your movie reviews are organized in folders for positive and negative reviews.
2. **Run the Code**: Execute the script to process the data, train models, and evaluate performance.
3. **Modify Paths**: Update file paths in the code to match your local system setup if necessary.

## Algorithms Implemented

### Random Forest

- **TreeNode Class**: Represents nodes in the decision tree.
- **calculate_entropy**: Computes the entropy of a dataset.
- **split_data**: Splits data based on a feature and a value.
- **find_best_split**: Finds the best feature and value to split data.
- **build_tree**: Constructs the decision tree.
- **random_forest**: Implements the Random Forest algorithm by building multiple trees.
- **predict**: Predicts labels for new samples.

### Naive Bayes

- **NaiveBayesClassifier Class**: Implements the Naive Bayes algorithm.
- **fit**: Trains the model using training data.
- **predict**: Predicts categories for new data points.
- **evaluate_metrics**: Calculates metrics such as true positives, false positives, etc.

## Model Evaluation

- Splits data into training and test sets.
- Trains models and evaluates them at different training set sizes.
- Computes metrics like accuracy, precision, recall, and F1 score.
- Plots learning curves for performance visualization.

## Visualization

### Learning Curves

Below are visualizations of the learning curves for both the custom implementation and the `scikit-learn` models:

**[Insert Image 1 Here]**
![Screenshot 2024-10-22 214701](https://github.com/user-attachments/assets/b6881b12-6ce3-4aab-ac13-8d054ae19d63)
![Screenshot 2024-10-22 214721](https://github.com/user-attachments/assets/990ed529-b4dd-4dab-b4b4-7d4524dc9b29)
![Screenshot 2024-10-22 214741](https://github.com/user-attachments/assets/55bca21d-6a31-4622-94ca-07ac2b8fd21e)
![Screenshot 2024-10-22 214755](https://github.com/user-attachments/assets/e62995a6-465b-4dcd-b502-9039ed05606f)
![Screenshot 2024-10-22 214809](https://github.com/user-attachments/assets/6858da76-cf2d-4943-9162-83c27b2b0be5)





## Authors

- **Lefteris Verouchis** (Α.Μ: 3200019)
- **Sofia Papaioannou** (Α.Μ: 3210157)

