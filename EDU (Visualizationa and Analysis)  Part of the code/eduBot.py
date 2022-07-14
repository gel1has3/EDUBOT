# This project attempted to predict student performance and recommended a range of assistance and coaching services.
# Geletaw Sahle

import streamlit as st
import pandas as pd
from sklearn import linear_model
import plotly.graph_objs as go
import plotly.figure_factory as ff
from sklearn.model_selection import KFold


import plotly.express as px
import seaborn as sns


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

import matplotlib.pyplot as plt


def settings(**args):
    st.info("Cont's Assessment:")
    numbers = [2, 3, 4, 5]
    num_of_Asssment = st.selectbox("Number of assessment: ", numbers)
    if num_of_Asssment == 2:
        continousAssessmentMin = 8
        continousAssessmentMAx = 15
        finalMin = 35
        finalMAx = 70
    elif num_of_Asssment == 3:
        continousAssessmentMin = 5
        continousAssessmentMAx = 10
        finalMin = 35
        finalMAx = 70
    elif num_of_Asssment == 4:
        continousAssessmentMin = 5
        continousAssessmentMAx = 10
        finalMin = 30
        finalMAx = 60
    else:
        continousAssessmentMin = 5
        continousAssessmentMAx = 10
        finalMin = 25
        finalMAx = 50
    return num_of_Asssment, continousAssessmentMin, continousAssessmentMAx, finalMin, finalMAx


# path = Image.open('/Users/geletawsahle/Desktop/CP_2020_WEPAPP/EDUBOTLogo.png')

# st.sidebar.image(path)


def evalutation(**args):
    num_of_Asssment, continousAssessmentMin, continousAssessmentMAx, finalMin, finalMAx = settings()
    i = 0

    eval = {}
    evalFeedbak = {}
    while i < num_of_Asssment:
        assValue = st.slider("Score of assessment {}".format(i+1),
                             min_value=0, value=continousAssessmentMin, max_value=continousAssessmentMAx)
        eval["Assessment {}".format(i+1)] = assValue
        feedback = instantFeedback(assValue, continousAssessmentMin, continousAssessmentMAx)
        evalFeedbak["Assessment {}".format(i+1)] = feedback
        i = i + 1

    # getting the MaX Score of final values

    st.info("Final Assessment (Min={} and Max = {}):".format(finalMin, finalMAx))
    eval_Final = st.slider("Score of final assessment",
                           min_value=finalMin, value=50, max_value=finalMAx)
    eval["Final Assessment"] = eval_Final

    # getting the total values
    total = 0
    for value in eval.values():
        total = total + value
    eval["Total"] = total
    evalFeedbak["Final Feedback and Status"] = finalFeedback(total)

    ####
    return eval, evalFeedbak


def instantFeedback(assValue, continousAssessmentMin, continousAssessmentMAx):
    if assValue < continousAssessmentMin:
        feedback = "Urgent Attention is required, able to do the following tasks"
    elif assValue == continousAssessmentMin:
        feedback = "equlibrium but it requires intensive "
    elif assValue == continousAssessmentMAx:
        feedback = "You scored the max, expetional, keep it up"
    else:
        feedback = "You scored a passed mark"
    return feedback


def finalFeedback(total):
    if total >= 80:
        finalFeedback = "Excellent"
    elif total >= 70:
        finalFeedback = "Very good"
    elif total >= 60:
        finalFeedback = "Good"
    elif total >= 45:
        finalFeedback = "Satisfactory"
    elif total >= 40:
        finalFeedback = "Unsatisfactory"
    else:
        finalFeedback = "Fail"
    return finalFeedback


DATA_URL = (
    '/Users/geletawsahle/Desktop/CP_2020_WEPAPP/C++_C2011_Batch.csv')


@st.cache(allow_output_mutation=True)
# Function to call historical records
def load_data():
    # Function to call historical records

    # raw CP data
    data = pd.read_csv(DATA_URL)
    return data


def splitTargetClass(data):
    X = data.drop('Category', axis=1)
    y = data['Category']
    return X, y


def convert_into_categorical_values(CPdata):
    for col in CPdata:
        CPdata[col] = CPdata[col].astype('category')
    return CPdata


def LabelEncoding(CPdata):
    """Label Encoding: Simply converting each value in a column to a number.
    Label encoding has the advantage that it is straightforward but it has the disadvantage that
    the numeric values can be “misinterpreted” by the algorithms.
    For example, the value of 0 is obviously less than the value of 4 but does
    that really correspond to the data set in real life?
    """
    for col in CPdata:
        CPdata[col] = CPdata[col].cat.codes
    return CPdata


def plot_confusion_matrix(data, labels, output_filename):
    """Plot confusion matrix using heatmap.

    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.

    """
    sns.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))

    plt.title("Confusion Matrix")

    sns.set(font_scale=1.4)
    ax = sns.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set(ylabel="True Label", xlabel="Predicted Label")

    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()


def feedbackClassification():
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # st.subheader("Checkbox")
    data = load_data()
    k = st.sidebar.radio("Show", [
        "Visualize each assessment",
        "Table",
        "Correlation Plots",
        "Hist Plots",
        "3D plots",
        "Regression Model Training",
        "DO KFold",
        "Decision Model Training",

    ])
    if k == "Visualize each assessment":
        st.subheader("Line chart of each assessments")
        st.line_chart(data)
    elif k == "Table":
        st.dataframe(data, width=2000, height=500)
    elif k == "Hist Plots":
        st.subheader("Distributions of each columns")
        options = data.columns
        sel_cols = st.selectbox("select columns", options, 1)
        st.write(sel_cols)
        # f=plt.figure()
        fig = go.Histogram(x=data[sel_cols], nbinsx=50)
        st.plotly_chart([fig])
    elif k == "Correlation Plots":
        st.subheader("Exploring correlation")
        options = data.columns
        w7 = st.selectbox("Assessment", options, 1)
        st.write(w7)
        f = plt.figure()
        y = st.selectbox("select columns", data.columns)
        plt.scatter(x=w7, y=y)
        plt.xlabel(w7)
        plt.ylabel(y)
        plt.title(f"{w7} vs {y}")
        # plt.show()
        st.plotly_chart(f)
    elif k == "3D plots":
        x = st.selectbox('X', data.columns, index=0)
        y = st.selectbox('Y', data.columns, index=1)
        z = st.selectbox('Z', data.columns, index=2)

        twoD = st.checkbox("Do you want to see the 2D versions?")
        if twoD:
            # fig = px.scatter(x=[data[x]], y=[data[y]])
            fig = px.scatter(
                x=data[x],
                y=data[y],
            )
            fig.update_layout(
                xaxis_title=x,
                yaxis_title=y,
            )

            st.write(fig)

        hist_data = [data[x].values, data[y].values, data[z].values]
        #x, y, z = np.random.multivariate_normal(np.array([0, 0, 0]), np.eye(3), 400).transpose()
        trace1 = go.Scatter3d(
            x=data[x],
            y=data[y],
            z=data[z],
            mode="markers",
            marker=dict(
                size=8,
                # color=df['sales'],  # set color to an array/list of desired values
                colorscale="Viridis",  # choose a colorscale
                #        opacity=0.,
            ),
        )

        data = [trace1]
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=data, layout=layout)
        st.write(fig)
    elif k == "Dist View":
        st.subheader("Combined distribution viewer (Select 3 columns at MAX)")

        x, y, z = st.multiselect("", data.columns)
        # Group data together
        hist_data = [data[x].values, data[y].values, data[z].values]

        group_labels = [x, y, z]

        # Create distplot with custom bin_size
        fig = ff.create_distplot(hist_data, group_labels, bin_size=[0.1, 0.25, 0.5])

        # Plot!
        st.plotly_chart(fig)
    elif k == "Regression Model Training":
        st.header("Modeling")
        # X, y = splitTargetClass(data)
        y = data.Total
        X = data[["Final_xx", "ContAssFinal", "Ass1", "Ass2", "Ass3", "Ass4"]].values
        # st.write(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        lrgr = LinearRegression()
        lrgr.fit(X_train, y_train)
        pred = lrgr.predict(X_test)

        mse = mean_squared_error(y_test, pred)
        rmse = sqrt(mse)

        st.markdown(f"""
        Linear Regression model trained :
    		- MSE:{mse}
    		- RMSE:{rmse}
    	""")
        st.success('Model trained successfully')
    elif k == "DO KFold":
        st.subheader("KFOLD Random sampling Evalution")
        st.empty()
        my_bar = st.progress(0)
        y = data.Total
        X = data[["Final_xx", "ContAssFinal", "Ass1", "Ass2", "Ass3", "Ass4"]].values
        # st.progress()
        kf = KFold(n_splits=10)
        # X=X.reshape(-1,1)
        mse_list = []
        rmse_list = []
        r2_list = []
        idx = 1
        fig = plt.figure()
        i = 0
        for train_index, test_index in kf.split(X):
            #	st.progress()
            my_bar.progress(idx*10)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            lrgr = LinearRegression()
            lrgr.fit(X_train, y_train)
            pred = lrgr.predict(X_test)

            mse = mean_squared_error(y_test, pred)
            rmse = sqrt(mse)
            r2 = r2_score(y_test, pred)
            mse_list.append(mse)
            rmse_list.append(rmse)
            r2_list.append(r2)
            plt.plot(pred, label=f"dataset-{idx}")
            idx += 1
        plt.legend()
        plt.xlabel("Data points")
        plt.ylabel("PRedictions")
        plt.show()
        st.plotly_chart(fig)

        res = pd.DataFrame(columns=["MSE", "RMSE", "r2_SCORE"])
        res["MSE"] = mse_list
        res["RMSE"] = rmse_list
        res["r2_SCORE"] = r2_list

        st.write(res)
        st.balloons()
    elif k == "Decision Model Training":
        st.subheader("Decision tree model training")

        # X, y = splitTargetClass(data)
        data = convert_into_categorical_values(data)
        df = LabelEncoding(data)

        y = df.Category
        X = df[["Final_xx", "ContAssFinal", "Ass1", "Ass2", "Ass3", "Ass4", "Total"]].values
        # st.progress()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        mse = mean_squared_error(y_test, pred)
        rmse = sqrt(mse)

        st.markdown(f"""
        Decision Tree model trained :
    		- MSE:{mse}
    		- RMSE:{rmse}
    	""")
        # Distribution of y test
        st.write('y actual : \n' + str(y_test.value_counts()))

        # Distribution of y predicted
        st.write('y predicted : \n' + str(pd.Series(pred).value_counts()))

        # Model Evaluation metrics ... from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
        st.write('Accuracy Score : ' + str(accuracy_score(y_test, pred)))
        #st.write('Precision Score : ' + str(precision_score(y_test, pred)))
        #st.write('Recall Score : ' + str(recall_score(y_test, pred)))
        #st.write('F1 Score : ' + str(f1_score(y_test, pred)))

        # Dummy Classifier Confusion matrix .... from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, pred)
        st.success('Confusion matrix: ')
        st.write(cm)

        st.write(plot_confusion_matrix(df, df.columns, "confusion_matrix.png"))

        fig = sns.heatmap(cm, annot=True)
        st.write(fig)

        dot_data = tree.export_graphviz(clf, out_file=None)
        st.success("Model Tree Visualization")
        st.graphviz_chart(dot_data)

    else:  # linechart:
        st.subheader("Line chart of each assessments")
        st.line_chart(data)
    return regr


if __name__ == "__main__":

    st.title("EDU-ADI:Assistive and adaptive coaching instruments")
    st.markdown("""
        The EDU-ADI tried to generate information and
        insight from historical records for coaching and
        delivering personalied feedback
    """)
    st.sidebar.title("EDUBOT- Simulation and operations on the Dataset")
    choiceOptions = ["Simulation", "Modeling: Regression"]
    choice = st.sidebar.selectbox("Choose", choiceOptions)

    if choice == "Simulation":
        eval, evalFeedbak = evalutation()
        st.info("Assessment Scores:",)
        st.write(eval)

        showFeedback = st.checkbox("Feedback Details")
        if showFeedback:
            st.info("Feedback:")
            st.write(evalFeedbak)
    else:
        feedbackClassification()
