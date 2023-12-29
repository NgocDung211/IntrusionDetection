import tensorflow as tf
from tensorflow import keras 
import pydot

from keras.src.utils.vis_utils import plot_model

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from keras import regularizers
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
pd.set_option('display.max_columns',None)
warnings.filterwarnings('ignore')


data_train = pd.read_csv(r"C:\MyCode\BTL_TTCSN\NSL_KDDTrain+.csv")
data_test = pd.read_csv(r"C:\MyCode\BTL_TTCSN\NSL_KDDTest+.csv")
data_train.loc[data_train['outcome'] == "normal", "outcome"] = 'normal'
data_train.loc[data_train['outcome'] != 'normal', "outcome"] = 'attack'
data_test.loc[data_test['outcome'] == "normal", "outcome"] = 'normal'
data_test.loc[data_test['outcome'] != 'normal', "outcome"] = 'attack'
def pie_plot(df, cols_list, rows, cols):
    fig, axes = plt.subplots(rows, cols)
    for ax, col in zip(axes.ravel(), cols_list):
        df[col].value_counts().plot(ax=ax, kind='pie', figsize=(15, 15), fontsize=10, autopct='%1.0f%%')
        ax.set_title(str(col), fontsize = 12)
    plt.show()


#pie_plot(data_train, ['protocol_type', 'label'], 1, 2)

def Scaling(df_num, cols):
    std_scaler = RobustScaler()
    std_scaler_temp = std_scaler.fit_transform(df_num)
    std_df = pd.DataFrame(std_scaler_temp, columns =cols)
    return std_df
  

cat_cols = ['is_host_login', 'protocol_type', 'service', 'flag',
            'land', 'logged_in', 'is_guest_login', 'level', 'outcome']
def preprocess(dataframe):
    df_num = dataframe.drop(cat_cols, axis=1)
    num_cols = df_num.columns
    scaled_df = Scaling(df_num, num_cols)

    dataframe.drop(labels=num_cols, axis="columns", inplace=True)
    dataframe[num_cols] = scaled_df[num_cols]

    dataframe.loc[dataframe['outcome'] == "normal", "outcome"] = 0
    dataframe.loc[dataframe['outcome'] != 0, "outcome"] = 1

    dataframe = pd.get_dummies(
        dataframe, columns=['protocol_type', 'service', 'flag'])
    
    return dataframe


scaled_train = preprocess(data_train)
scaled_test = preprocess(data_test)
print(scaled_train.head())
x_train = scaled_train.drop(['outcome', 'level'], axis=1).values
y_train = scaled_train['outcome'].values
y_train = y_train.astype('int')
x_test = scaled_test.drop(['outcome', 'level'], axis=1).values
y_test = scaled_test['outcome'].values
y_test = y_test.astype('int')

# x = scaled_train.drop(['outcome', 'level'], axis=1).values
# y = scaled_train['outcome'].values

# y_reg = scaled_train['level'].values

# pca = PCA(n_components=20)
# pca = pca.fit(x)
# x_reduced = pca.transform(x)
# print("Number of original features is {} and of reduced features is {}".format(
#     x.shape[1], x_reduced.shape[1]))

# y = y.astype('int')
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.2, random_state=42)
# x_train_reduced, x_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
#     x_reduced, y, test_size=0.2, random_state=42)
# x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(
#     x, y_reg, test_size=0.2, random_state=42)

kernal_evals = dict()
def evaluate_classification(model, name, X_train, X_test, y_train, y_test):
  predicted = model.predict(X_test)
  confusion_matrix = metrics.confusion_matrix(y_test, predicted)

  tn, fp, fn, tp = confusion_matrix.ravel()

  sensitivity = tp / (tp + fn)
  specificity = tn / (tn + fp)

  kernal_evals[str(name)] = [sensitivity, specificity, 1 - specificity]
  print(f"this is the result of {name}")
  print(f"TPR: {100 * sensitivity:.2f}%")
  print(f"TNR: {100 * specificity:.2f}%")
  print(f"FNR: {100 * (1-specificity):.2f}%")


lr = LogisticRegression().fit(x_train, y_train)
evaluate_classification(lr, "Logistic Regression",
                        x_train, x_test, y_train, y_test)
    
# knn = KNeighborsClassifier(n_neighbors=20).fit(x_train, y_train)
# evaluate_classification(knn, "KNeighborsClassifier",
#                         x_train, x_test, y_train, y_test)
gnb = GaussianNB().fit(x_train, y_train)
evaluate_classification(gnb, "GaussianNB", x_train, x_test, y_train, y_test)

lin_svc = svm.LinearSVC().fit(x_train, y_train) 
evaluate_classification(lin_svc, "Linear SVC(LBasedImpl)",
                        x_train, x_test, y_train, y_test)

dt = DecisionTreeClassifier(max_depth=3).fit(x_train, y_train)
tdt = DecisionTreeClassifier().fit(x_train, y_train)
evaluate_classification(tdt, "DecisionTreeClassifier",
                        x_train, x_test, y_train, y_test)


# def f_importances(coef, names, top=-1):
#     imp = coef
#     imp, names = zip(*sorted(list(zip(imp, names))))

#     # Show all features
#     if top == -1:
#         top = len(names)

#     plt.figure(figsize=(10, 10))
#     plt.barh(range(top), imp[::-1][0:top], align='center')
#     plt.yticks(range(top), names[::-1][0:top])
#     plt.title('feature importances for Decision Tree')
#     plt.show()


# features_names = data_train.drop(['outcome', 'level'], axis=1)
# f_importances(abs(tdt.feature_importances_), features_names, top=18)

# fig = plt.figure(figsize=(15, 12))
# tree.plot_tree(dt, filled=True)


rf = RandomForestClassifier().fit(x_train, y_train)
evaluate_classification(rf, "RandomForestClassifier",
                        x_train, x_test, y_train, y_test)
# f_importances(abs(rf.feature_importances_), features_names, top=18)

# rrf = RandomForestClassifier().fit(x_train_reduced, y_train_reduced)
# evaluate_classification(rrf, "PCA RandomForest", x_train_reduced,
#                         x_test_reduced, y_train_reduced, y_test_reduced)


# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=64, activation='relu', input_shape=(x_train.shape[1:]),
#                           kernel_regularizer=regularizers.L1L2(
#                               l1=1e-5, l2=1e-4),
#                           bias_regularizer=regularizers.L2(1e-4),
#                           activity_regularizer=regularizers.L2(1e-5)),
#     tf.keras.layers.Dropout(0.4),
#     tf.keras.layers.Dense(units=128, activation='relu',
#                           kernel_regularizer=regularizers.L1L2(
#                               l1=1e-5, l2=1e-4),
#                           bias_regularizer=regularizers.L2(1e-4),
#                           activity_regularizer=regularizers.L2(1e-5)),
#     tf.keras.layers.Dropout(0.4),
#     tf.keras.layers.Dense(units=512, activation='relu',
#                           kernel_regularizer=regularizers.L1L2(
#                               l1=1e-5, l2=1e-4),
#                           bias_regularizer=regularizers.L2(1e-4),
#                           activity_regularizer=regularizers.L2(1e-5)),
#     tf.keras.layers.Dropout(0.4),
#     tf.keras.layers.Dense(units=128, activation='relu',
#                           kernel_regularizer=regularizers.L1L2(
#                               l1=1e-5, l2=1e-4),
#                           bias_regularizer=regularizers.L2(1e-4),
#                           activity_regularizer=regularizers.L2(1e-5)),
#     tf.keras.layers.Dropout(0.4),
#     tf.keras.layers.Dense(units=1, activation='sigmoid'),
# ])
# model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(
#     from_logits=True), metrics=['accuracy'])
# model.summary()
# print(x_train)
# print(y_train)
# print(y_test)
# x_train = np.array(x_train)
# y_train = np.array(y_train)
# x_test = np.array(x_test)
# y_test = np.array(y_test)
# #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# history = model.fit(x_train, y_train, validation_data=(
#     x_test, y_test), epochs=10, verbose=1)


# plt.plot(history.history['loss'], label='loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.xlabel('Epoch')
# plt.ylabel('SCCE Loss')
# plt.legend()
# plt.grid(True)

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)

keys = [key for key in kernal_evals.keys()]
values = [value for value in kernal_evals.values()]
fig, ax = plt.subplots(figsize=(20, 6))
ax.bar(np.arange(len(keys)) - 0.2,
       [value[0] for value in values], color='darkred', width=0.25, align='center')
ax.bar(np.arange(len(keys)) + 0.2,
       [value[2] for value in values], color='y', width=0.25, align='center')
for i, value in enumerate(values):
    ax.text(i - 0.2, value[0] + 0.01, f"{value[0]:.2f}",
            ha='center', va='bottom', color='black')
    ax.text(i + 0.2, value[2] + 0.01, f"{value[2]:.2f}",
            ha='center', va='bottom', color='black')
ax.legend(["TNR", "FNR"])
ax.set_xticklabels(keys)
ax.set_xticks(np.arange(len(keys)))
plt.ylabel("Accuracy")
plt.show()
