import streamlit as st
import pandas as pd
import numpy as np
import os
import time

from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support as score, mean_squared_error
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import plot_confusion_matrix

import pickle

import matplotlib.pyplot as plt
from matplotlib import dates as plotdates
from matplotlib.figure import Figure

import webbrowser

# Constants
MAIN_PATH = os.path.dirname(__file__)

FEATURES_PATH = os.path.join(MAIN_PATH, "dataset/data_train/")

MODEL_PATH = os.path.join(MAIN_PATH, "models/")
#DATA_TRAIN_PATH = os.path.join(MAIN_PATH, "dataset/data_train/")
DATA_TRAIN = FEATURES_PATH + "selected_features_for_training.csv"
DATA_TEST_PATH = os.path.join(MAIN_PATH, "dataset/data_test/")

test_choices = ["1818471", "1066528", "1455390", "1360686", "1449548",
                "2598705", "2638030", "3509524", "3997827"]
test_options = ["1818471.csv", "1066528.csv",
                "1455390.csv", "1360686.csv", "1449548.csv",
                "2598705.csv", "2638030.csv", "3509524.csv", "3997827.csv"]

URL_XGB_DOC = 'https://xgboost.readthedocs.io/en/latest/python/python_intro.html#setting-parameters'
URL_GBM_DOC = 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html'
URL_XGB_SOURCE = 'https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/'
URL_GBM_SOURCE = 'https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/'

#---------------------------------#
# Page Setup
#---------------------------------#
st.set_page_config(page_title='Sleep-2-Learn: Learning',
                   layout="wide", page_icon=":bed:")

# Sidebar
classifier_name = st.sidebar.selectbox("Select Classifier: ", ("Logistic Regression",
                                                               "XGBoost",
                                                               "Gradient Boosting",
                                                               "Random Forest"))

if classifier_name == 'XGBoost':
    if st.sidebar.button('XGB Documentation'):
        webbrowser.open_new_tab(URL_XGB_DOC)
    if st.sidebar.button('XGB Resource'):
        webbrowser.open_new_tab(URL_XGB_SOURCE)
elif classifier_name == 'Gradient Boosting':
    if st.sidebar.button('Gradient Boosting Doc'):
        webbrowser.open_new_tab(URL_GBM_DOC)
    if st.sidebar.button('Gradient Boosting Resource'):
        webbrowser.open_new_tab(URL_GBM_SOURCE)

with st.sidebar.subheader("Select Class"):
    class_option = st.selectbox(
        'Select Class (Binary or Multi)',
        ["Sleep/Wake", "Wake/Light/Deep/REM"])

with st.sidebar.subheader("Select Subject"):
    test_option = st.selectbox(
        'Select Subject ID',
        list(range(len(test_choices))), format_func=lambda x: test_choices[x])


#---------------------------------#
# Classification Functions
#---------------------------------#


def add_parameter_ui(clf_name):
    params = {}

    st.sidebar.header('Classifier Parameters')
    st.sidebar.write("Select Values: ")

    if clf_name == "Logistic Regression":
        C = st.sidebar.slider("Regularization", 0.1, 10.0, step=0.1)
        max_iter = st.sidebar.slider("max_iter", 50, 400, step=50)
        params["C"] = C
        params["max_iter"] = max_iter

    elif clf_name == "Random Forest":
        n_estimators = st.sidebar.slider(
            "n_estimators", 50, 500, step=50, value=100)
        max_depth = st.sidebar.slider("max_depth", 2, 20)
        criterion = st.sidebar.selectbox("criterion", ("gini", "entropy"))
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth
        params["criterion"] = criterion

    elif clf_name == "Gradient Boosting":
        n_estimators = st.sidebar.slider(
            "n_estimators", 50, 500, step=50, value=100)
        learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.5)
        loss = st.sidebar.selectbox("loss", ('deviance', 'exponential'))
        max_depth = st.sidebar.slider("max_depth", 2, 20)
        params["n_estimators"] = n_estimators
        params["learning_rate"] = learning_rate
        params["loss"] = loss
        params["max_depth"] = max_depth

    elif clf_name == "XGBoost":
        # Set starter parameters to best classifier from hyperparameters tuning
        n_estimators = int(st.sidebar.number_input(
            'n_estimators', min_value=0, value=50, step=10))
        learning_rate = st.sidebar.slider(
            "learning_rate", 0.01, 0.5, value=0.1)
        max_depth = int(st.sidebar.number_input(
            'max_depth ', min_value=1, step=1, value=6))
        eta = st.sidebar.slider('eta', min_value=0.0,
                                max_value=1.0, step=0.1, value=0.3)
        st.sidebar.text('Selected: {}'.format(eta))
        verbosity = int(st.sidebar.number_input(
            'verbosity', min_value=0, max_value=3, value=1, step=1))
        objective = st.sidebar.selectbox('objective', ('reg:squarederror', 'reg:squaredlogerror', 'reg:logistic', 'reg:pseudohubererror',
                                                       'binary:logistic', 'binary:logitraw', 'binary:hinge',
                                                       'count:poisson', 'survival:cox', 'survival:aft', 'aft_loss_distribution',
                                                       'multi:softmax', 'multi:softprob',
                                                       'rank:pairwise', 'rank:ndcg', 'rank:map',
                                                       'reg:gamma', 'reg:tweedie'), index=4)
        booster = st.sidebar.radio(
            'booster', ('gbtree', 'gblinear', 'dart'), index=0)
        tree_method = st.sidebar.selectbox(
            'tree_method', ('auto', 'exact', 'approx', 'hist', 'gpu_hist'), index=0)
        n_jobs = int(st.sidebar.number_input(
            'n_jobs', min_value=0, step=1, value=1))
        gamma = float(st.sidebar.number_input(
            'gamma', min_value=0.0, value=0.0, step=0.1))
        reg_alpha = st.sidebar.number_input(
            'reg_alpha', min_value=0.0, value=0.0, step=0.1)
        reg_lambda = st.sidebar.number_input(
            'reg_lambda', min_value=0.0, value=1.0, step=0.1)
        min_child_weight = st.sidebar.number_input(
            'min_child_weight', min_value=0.0, value=1.0)
        max_delta_step = st.sidebar.number_input(
            'max_delta_step', min_value=0.0, value=0.0)
        subsample = st.sidebar.slider(
            'subsample', min_value=0.0, max_value=1.0, step=0.1, value=1.0)
        st.sidebar.text('Selected: {}'.format(subsample))
        colsample_bytree = st.sidebar.slider(
            'colsample_bytree', min_value=0.0, max_value=1.0, step=0.1, value=1.0)
        st.sidebar.text('Selected: {}'.format(colsample_bytree))
        colsample_bylevel = st.sidebar.slider(
            'colsample_bylevel', min_value=0.0, max_value=1.0, step=0.1, value=1.0)
        st.sidebar.text('Selected: {}'.format(colsample_bylevel))
        colsample_bynode = st.sidebar.slider(
            'colsample_bynode', min_value=0.0, max_value=1.0, step=0.1, value=1.0)
        st.sidebar.text('Selected: {}'.format(colsample_bynode))
        scale_pos_weight = st.sidebar.number_input(
            'scale_pos_weight', step=0.1, value=1.0)
        base_score = st.sidebar.number_input('base_score', value=0.5)
        random_state = int(st.sidebar.number_input(
            'random_state', min_value=0, value=0, step=1))
        missing_option = st.sidebar.radio(
            'missing', ('np.nan', 'Set a value in float'))

        if missing_option == 'Set a value in float':
            missing = st.sidebar.number_input('Input for missing value')
        else:
            missing = np.nan

        num_parallel_tree = st.sidebar.number_input(
            'num_parallel_tree', value=1, min_value=0)
        importance_type = st.sidebar.selectbox(
            'importance_type', ('gain', 'weight', 'cover', 'total_gain', 'total_cover'), index=0)
        validate_parameters = st.sidebar.radio(
            'validate_parameters', (True, False), index=1)

        params["learning_rate"] = learning_rate
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth
        params["eta"] = eta
        params["verbosity"] = verbosity
        params["objective"] = objective
        params["booster"] = booster
        params["n_jobs"] = n_jobs
        params["tree_method"] = tree_method
        params["gamma"] = gamma
        params["min_child_weight"] = min_child_weight
        params["max_delta_step"] = max_delta_step
        params["subsample"] = subsample
        params["colsample_bytree"] = colsample_bytree
        params["colsample_bylevel"] = colsample_bylevel
        params["colsample_bynode"] = colsample_bynode
        params["reg_alpha"] = reg_alpha
        params["reg_lambda"] = reg_lambda
        params["scale_pos_weight"] = scale_pos_weight
        params["base_score"] = base_score
        params["random_state"] = random_state
        params["missing"] = missing
        params["num_parallel_tree"] = num_parallel_tree
        params["importance_type"] = importance_type
        params["validate_parameters"] = validate_parameters

    return params


clf_params = add_parameter_ui(classifier_name)


def select_classifier(clf_name, params):
    global classifier

    if clf_name == "Logistic Regression":
        classifier = LogisticRegression(C=params["C"],
                                        max_iter=params["max_iter"])

    elif clf_name == "Random Forest":
        classifier = RandomForestClassifier(n_estimators=params["n_estimators"],
                                            max_depth=params["max_depth"],
                                            criterion=params["criterion"])

    elif clf_name == "Gradient Boosting":
        classifier = GradientBoostingClassifier(n_estimators=params["n_estimators"],
                                                learning_rate=params["learning_rate"],
                                                loss=params["loss"],
                                                max_depth=params["max_depth"])

    elif clf_name == "XGBoost":
        classifier = XGBClassifier(learning_rate=params["learning_rate"],
                                   n_estimators=params["n_estimators"],
                                   max_depth=params["max_depth"],
                                   eta=params["eta"],
                                   verbosity=params["verbosity"],
                                   objective=params["objective"],
                                   booster=params["booster"],
                                   tree_method=params["tree_method"],
                                   n_jobs=params["n_jobs"],
                                   gamma=params["gamma"],
                                   min_child_weight=params["min_child_weight"],
                                   max_delta_step=params["max_delta_step"],
                                   subsample=params["subsample"],
                                   colsample_bytree=params["colsample_bytree"],
                                   colsample_bylevel=params["colsample_bylevel"],
                                   colsample_bynode=params["colsample_bynode"],
                                   reg_alpha=params["reg_alpha"],
                                   reg_lambda=params["reg_lambda"],
                                   scale_pos_weight=params["scale_pos_weight"],
                                   base_score=params["base_score"],
                                   random_state=params["random_state"],
                                   missing=params["missing"],
                                   num_parallel_tree=params["num_parallel_tree"],
                                   importance_type=params["importance_type"],
                                   validate_parameters=params["validate_parameters"]
                                   )

    return classifier


#classifier = select_classifier(classifier_name, clf_params)

#---------------------------------#
# General Functions
#---------------------------------#


def load_file(path, index=False):
    if index:
        df = pd.read_csv(path, index_col=[0])
        df.index = pd.to_datetime(df.index)
    else:
        df = pd.read_csv(path, index_col=[0])

    return df


def get_classifier(flag):
    if flag == "Sleep/Wake":
        conf_plot_labels = ["Wake", "Sleep"]
        clf = select_classifier(classifier_name, clf_params)
    if flag == "Wake/Light/Deep/REM":
        clf = select_classifier(classifier_name, clf_params)
        conf_plot_labels = ["Wake", "Light", "Deep", "REM"]
    return clf, conf_plot_labels


@st.cache(allow_output_mutation=True)
def fit_model(classification, X, y):
    classification.fit(X, y)
    clf_class = ' [Sleep-Wake]'
    if class_option == "Wake/Light/Deep/REM":
        clf_class = ' [Multi-Class]'

    model_filename = MODEL_PATH + classifier_name + clf_class + '_model.sav'
    pickle.dump(classification, open(model_filename, 'wb'))

    return classification


def detect_class_type(flag, df):
    if flag == "Sleep/Wake":
        df["psg_label"] = df["psg_label"].replace(
            [1, 2, 3, 4, 5], "Sleep")
        # Set Sleep = 1
        df["psg_label"] = df["psg_label"].replace(["Sleep"], 1)
        # Wake = 0 (no need to change)
        #df["psg_label"] = df["psg_label"].replace([0], "Wake")
    if flag == "Wake/Light/Deep/REM":
        df["psg_label"] = df["psg_label"].replace([0], "Wake")
        df["psg_label"] = df["psg_label"].replace([1, 2], "Light")
        df["psg_label"] = df["psg_label"].replace([3, 4], "Deep")
        df["psg_label"] = df["psg_label"].replace([5], "REM")
    return df


def sec_to_hours(seconds):
    sec = str(seconds//3600)
    min = str((seconds % 3600)//60)
    hour = str((seconds % 3600) % 60)
    time_hour = ["{} hours {} mins {} seconds".format(sec, min, hour)]
    return time_hour


def change_labels(flag, y_pred, y_true):
    if flag == "Sleep/Wake":
        y_indexes = ["Sleep", "Wake"]
        y_pred = pd.Series(y_pred)
        y_pred = y_pred.replace(1, "Sleep")
        y_pred = y_pred.replace(0, "Wake")
        y_true = y_true.replace(1, "Sleep")
        y_true = y_true.replace(0, "Wake")

    return y_indexes, y_pred, y_true


def make_conf_matrix(clf, X, y, labels):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    disp = plot_confusion_matrix(clf, X, y,
                                 display_labels=labels,
                                 cmap=plt.cm.Blues,
                                 normalize=None,
                                 ax=ax)
    disp.ax_.set_ylabel("Actual labels")
    disp.ax_.set_xlabel("Predicted labels")
    return fig


def make_comparison_plot(y_pred, y_true, class_option, time_values):
    plot_time = sec_to_hours(time_values*30)

    fig = Figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.xaxis.set_major_locator(plotdates.HourLocator())
    ax.xaxis.set_major_formatter(plotdates.DateFormatter('%H'))
    # ax.set_xlabel("Time[h]")
    ax.set_ylabel("Actual")
    ax.plot(plot_time, y_true)

    ax = fig.add_subplot(2, 1, 2)
    ax.xaxis.set_major_locator(plotdates.HourLocator())
    ax.xaxis.set_major_formatter(plotdates.DateFormatter('%H'))
    ax.set_xlabel("Time[h]")
    ax.set_ylabel("Prediction")
    ax.plot(plot_time, y_pred)

    fig.suptitle(class_option, fontsize=16)
    return fig


# ------------------------------
# Data loading
# ------------------------------

df_test = load_file(DATA_TEST_PATH + test_options[test_option], index=False)
df_test = detect_class_type(class_option, df_test)
df_test = df_test.drop(["subject_id", "session_id"], axis=1)

df_train = load_file(DATA_TRAIN)
df_train = detect_class_type(class_option, df_train)
df_train = df_train.drop(["subject_id", "session_id"], axis=1)

# Drop first column of dataframe
# (weird Unnamed: 0 column)
#df_train = df_train.iloc[:, 1:]


def run_classifier():
    # Classification Start time
    start_time = time.time()

    clf, conf_plot_labels = get_classifier(class_option)
    y_train = df_train["psg_label"]
    X_train = df_train.drop(["psg_label"], axis=1)
    clf = fit_model(clf, X_train, y_train)

    # Execution Time
    end_time = time.time()
    st.info(f"Execution Time: {round((end_time - start_time),3)} seconds")

    #---------------------------------#
    st.subheader('Accuracy')
    X_true = df_test.drop(["psg_label"], axis=1)
    y_true = df_test["psg_label"]

    y_pred = clf.predict(X_true)
    acc = accuracy_score(y_true, y_pred)
    st.write(acc)

    #---------------------------------#
    st.subheader('Confusion Matrix')
    fig = make_conf_matrix(clf, X_true, y_true, conf_plot_labels)
    st.write(fig)

    # Calculate Metrics
    if class_option == 'Sleep/Wake':
        acc = accuracy_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        precision, recall, fscore, train_support = score(
            y_true, y_pred, pos_label=1, average='binary')
        st.subheader("Metrics of the model: ")
        st.text('Precision: {} \nRecall: {} \nF1-Score: {} \nAccuracy: {} %\nMean Squared Error: {}'.format(
            round(precision, 3), round(recall, 3), round(fscore, 3), round((acc*100), 3), round((mse), 3)))

    #---------------------------------#
    #st.subheader('Prediction Comparison')
    #fig = make_comparison_plot(y_pred, y_true, class_option, df_test.index.values)
    # st.pyplot(fig)


#---------------------------------#
# Main Function
#---------------------------------#


def main():
    #---------------------------------#
    st.subheader('Data')
    st.write("Shape of data test: ", df_test.shape)
    st.write("Shape of data train: ", df_train.shape)
    if st.checkbox('Show data'):
        st.write(df_test)

    st.header('Classifier: ' + classifier_name)
    if st.button("Run Classifier"):
        run_classifier()


# ----------------------#
# Main page

st.title('Sleep-2-Learn: Learn to Sleep')
st.subheader('Sleep Classification')

main()
