import csv
import pandas
import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import validation_curve
from sklearn.naive_bayes import GaussianNB, CategoricalNB, BernoulliNB
from sklearn.decomposition import PCA



from time import perf_counter

def main():

    train1 = "D:\MyCodeC\BTL_TTCSN\data1\kdd_data.csv"
    test1 = "D:\MyCodeC\BTL_TTCSN\data1\kdd_test.csv"
    test2 = r"D:\MyCodeC\BTL_TTCSN\NSL_KDDTest+.csv"
    train2 = r"D:\MyCodeC\BTL_TTCSN\NSL_KDDTest+.csv"
    test3 = r"D:\MyCodeC\BTL_TTCSN\data2\UNSW_NB15_testing-set.csv"
    train3 = r"D:\MyCodeC\BTL_TTCSN\data2\UNSW_NB15_training-set.csv"

    
    
    models = ["MLPClassifier", "DecisionTreeClassifier",
              "LogisticRegression", "RandomForestClassifier", "GaussianNB"]
    training(models[0], train3, test3)
    



def training(selected_model , train_file_path, test_file_path):
    # param_name = "ccp_alpha"
    
    # param_range = np.logspace(-7, 3, 10)
    t1 = perf_counter()
    img_path = ""
    if(selected_model is None):
        raise Exception("Data usage model")
        return -1
    if(train_file_path is None):
        raise Exception("Data usage train file")
        return -1
    if(test_file_path is None):
        raise Exception("Data usage test file")
        return -1
    # Load data from spreadsheet and split into train and test sets
    X_train, labels_train = load_data(train_file_path)
    X_test, labels_test = load_data(test_file_path)
    # scaler = preprocessing.StandardScaler()
    # X_train = scaler.fit_transform(X_train)

    
    # X_test = scaler.transform(X_test)
    
    # pca = PCA(n_components=35)
    # pca = pca.fit(X_train)
    # x_reduced = pca.transform(X_train)
    # pca = pca.fit(test_data)
    # test_reduced = pca.transform(test_data)
    # print("Number of original features is {} and of reduced features is {}".format(X_train.shape[1], x_reduced.shape[1]))
    # Train model and make predictions
    model = train_model(X_train, labels_train, selected_model)

    # model = train_model(X_train, labels_train, selected_model, cv)
    #cv_scores = cross_val_score(model, X_train, labels_train, cv=StratifiedKFold(n_splits=cv), scoring='accuracy')
    # print(f"Cross-validation scores: {cv_scores}")
    # print(f"Mean Accuracy: {cv_scores.mean()}")
    model.fit(X_train, labels_train)

    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(labels_test, predictions)
    t2 = perf_counter()
    print(f"Thời gian {(t2 - t1):.2f}s")
    # Print results
    print(f"this is the result of {selected_model}")
    print(f"TPR: {100 * sensitivity:.2f}%")
    print(f"TNR: {100 * specificity:.2f}%")
    print(f"FNR: {100 * (1-specificity):.2f}%")
    # train_scores, valid_scores = validation_curve(
    #     model, X_train, labels_train, param_name=param_name, param_range=param_range, cv=StratifiedKFold(n_splits=cv), scoring='accuracy'
    # )
    # plt.figure(figsize=(10, 6))
    # plt.plot(param_range, np.mean(train_scores, axis=1), label='Training score', marker='o')
    # plt.plot(param_range, np.mean(valid_scores, axis=1), label='Validation score', marker='o')
    # plt.xscale('log')
    # plt.xlabel(param_name)
    # plt.ylabel('Score')
    # plt.title('Validation Curve')
    # plt.legend()
    # plt.savefig('validation_curve1.png')

    # plt.show()
    return sensitivity , specificity, labels_test, predictions

def load_test(filename):
    evidence = []
    # read csv file
    evidence_df = pandas.read_csv(filename)
    label_encoder = LabelEncoder()
    features = []
    if("kdd" in filename):   
        features = ['protocol_type' ,'service', 'flag', 'is_host_login']
        # evidence_df['protocol_type'] = label_encoder.fit_transform(evidence_df['protocol_type'])
        # # Encode the 'service' column
        # evidence_df['service'] = label_encoder.fit_transform(evidence_df['service'])

        # # Encode the 'flag' column
        # evidence_df['flag'] = label_encoder.fit_transform(evidence_df['flag'])

        # # Replace boolean values with 0/1
        # evidence_df['is_host_login'] = label_encoder.fit_transform(evidence_df['is_host_login'])
    for feature in features:
        evidence_df[feature] = label_encoder.fit_transform(evidence_df[feature])
  
    
  
    # convert dataframes to lists
    evidence_list = evidence_df.values.tolist()
    return evidence_list
def load_data(filename):

    evidence = []
    labels = []
    print("this is read data step")
    # read filename
    csv_file = pandas.read_csv(filename)
    labels_df = []

    # different file name has different features
    label_encoder = LabelEncoder()
    if("kdd" in filename):   
        print("====DATASET KDD=====")
        csv_file['label'] = csv_file['label'].apply(lambda x: 1 if x == 'normal.' else 0)
        csv_file.loc[csv_file['label'] == "normal", "label"] = 0
        csv_file.loc[csv_file['label'] != 0, "label"] = 1

        labels_df = csv_file['label']
        evidence_df = csv_file.drop(columns=['label'])
        features = ['protocol_type' ,'service', 'flag', 'is_host_login']
        
    if("UNSW" in filename):
        print("====DATASET UNWS=====")
        labels_df = csv_file['label']
        # Exclude non-numeric columns
        evidence_df = csv_file.drop(['id', 'attack_cat', 'label'], axis=1)
        evidence_df = csv_file.iloc[:, :-2]
        features = ['proto', 'service', 'state']
    
    if("NSL" in filename):
        print("====DATASET NSL=====")
        #csv_file['outcome'] = csv_file['outcome'].apply(lambda x: 1 if x == 'normal' else 0)
        csv_file.loc[csv_file['outcome'] == "normal", "outcome"] = 0
        csv_file.loc[csv_file['outcome'] != 0, "outcome"] = 1
        labels_df = csv_file['outcome']
        evidence_df = csv_file.drop(columns=['outcome','level'])
        features = ['protocol_type' ,'service', 'flag']
        
    
    for feature in features:
        evidence_df[feature] = label_encoder.fit_transform(evidence_df[feature])
    evidence_list = evidence_df.values.tolist()
    labels_list = labels_df.values.tolist()

    return evidence_list, labels_list



def train_model(evidence, labels, selected_model):

    print("this is train model step")


    print(selected_model)
    if selected_model == "LogisticRegression":
        #model = LogisticRegression(solver="saga")
        model = LogisticRegression(solver="saga", C=1.0, penalty="l2")

    elif selected_model == "RandomForestClassifier":
        #model = RandomForestClassifier(n_estimators=100, random_state=42)
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=None, min_samples_split=2, min_samples_leaf=1)

    elif selected_model == "MLPClassifier":
        model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(
            100, 50), random_state=1, max_iter=50, verbose=10, learning_rate_init=0.001)

        
        #model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100,50), random_state=1, max_iter= 50, verbose=10, learning_rate_init=0.001)
    elif selected_model == "GaussianNB":
        model = BernoulliNB()

    else:
        model = DecisionTreeClassifier(
            criterion='log_loss', splitter='best', max_depth=80, min_samples_split=4, min_samples_leaf=1)

        #model = DecisionTreeClassifier()

    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    print("this is evaluate step")
    cm = confusion_matrix(labels, predictions)
    print(cm)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
 

    return sensitivity, specificity

if __name__ == "__main__":
    main()
