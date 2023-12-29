# Intrusion Detection System

## Overview
![KQ](https://github.com/NgocDung211/IntrusionDetection/assets/122346488/4f17c6f9-37e4-4e9d-8559-a5c4deefb3a7)


This project implements an Intrusion Detection System using machine learning techniques. It aims to classify network data into normal or intrusive behavior. The implemented models include MLPClassifier, DecisionTreeClassifier, LogisticRegression, RandomForestClassifier, and GaussianNB.

## Prerequisites

- Python 3.13
- Required Python packages: pandas, numpy, matplotlib, scikit-learn

## Getting Started

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/NgocDung211/IntrusionDetection.git
2. Navigate to the project directory:

cd intrusion-detection

3. Install the required dependencies:

pip install -r requirements.txt

##Dataset
The project uses three different datasets for training and testing:

1. KDD Cup 1999 Data (located in data1 directory):

Training data: kdd_data.csv
Testing data: kdd_test.csv
NSL-KDD (located in the project root directory):

2. Training data: NSL_KDDTrain+.csv
Testing data: NSL_KDDTest+.csv
UNSW-NB15 (located in data2 directory):

3. Training data: UNSW_NB15_training-set.csv
Testing data: UNSW_NB15_testing-set.csv
Ensure that the dataset files are available at the specified paths in the code.

##Running the Code
To execute the intrusion detection system with the MLPClassifier model, run the main() function in the main.py file.

python main.py
You can modify the train1, train2, train3, test1, test2, and test3 variables in the code to specify your dataset paths.

##Models
The project utilizes the following machine learning models:

MLPClassifier
DecisionTreeClassifier
LogisticRegression
RandomForestClassifier
GaussianNB
Each model is trained and evaluated separately. The chosen model for training and testing can be adjusted by modifying the models list in the code.

##Results
After running the code, the system will output performance metrics such as accuracy, confusion matrix, and execution time for each model.

##License
This project is licensed under the MIT License - see the LICENSE file for details.

##Acknowledgments
The datasets used in this project are sourced from KDD Cup 1999, NSL-KDD, and UNSW-NB15.
Special thanks to contributors and developers of scikit-learn and other related libraries.
