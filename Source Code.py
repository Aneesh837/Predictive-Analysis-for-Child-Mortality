import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, matthews_corrcoef,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize
from itertools import cycle
import plotly.graph_objs as go
import plotly.subplots as sp

# Set a valid seaborn style
if 'seaborn' in plt.style.available:
    plt.style.use('seaborn')
else:
    plt.style.use('ggplot')  # Fallback style

@st.cache_data
def load_data():
    # Update this path to your dataset location
    df = pd.read_csv("child.csv")
    return df

df = load_data()

# Preprocessing
X = df.drop('fetal_health', axis=1)
y = df['fetal_health'] - 1  # Adjust target values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

@st.cache_data
def train_models():
    # Logistic Regression
    log_model = LogisticRegression(random_state=42)
    log_model.fit(X_train, y_train)
    
    # SVM
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    
    # Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    
    return log_model, svm_model, rf_model

log_model, svm_model, rf_model = train_models()

# Generate predictions
log_y_pred = log_model.predict(X_test)
svm_y_pred = svm_model.predict(X_test)
rf_y_pred = rf_model.predict(X_test)

# Calculate accuracies
log_accuracy = accuracy_score(y_test, log_y_pred)
svm_accuracy = accuracy_score(y_test, svm_y_pred)
rf_accuracy = accuracy_score(y_test, rf_y_pred)

# Helper functions for visualizations
def plot_confusion_matrix_heatmap(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return fig

def plot_multi_class_roc(model, X_test, y_test, num_classes=3):
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    y_pred_proba = model.predict_proba(X_test)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = cycle(['blue', 'red', 'green'])
    
    for i, color in zip(range(num_classes), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2, 
                label=f'Class {i} (AUC = {roc_auc[i]:.4f})')

    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Multi-Class ROC Curve')
    ax.legend(loc="lower right")
    return fig

# Streamlit UI
st.title("Child Mortality Prediction Analysis")

# EDA Section
st.header("Exploratory Data Analysis (EDA)")

if st.button("Show Data Sample"):
    st.write("First 5 rows of the dataset:")
    st.write(df.head())

if st.button("Show Data Info"):
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

if st.button("Show Null Counts"):
    st.write("Missing values per column:")
    st.write(df.isnull().sum())

# Visualization Section
st.header("Data Visualizations")

if st.button("Show Fetal Health Distribution"):
    # Create combined Plotly figure
    bar_fig = go.Bar(
        x=(df['fetal_health']-1).value_counts().index,
        y=(df['fetal_health']-1).value_counts().values,
        marker=dict(color='#66C2A5')
    )
    pie_fig = go.Pie(
        labels=(df['fetal_health']-1).value_counts().index,
        values=(df['fetal_health']-1).value_counts().values,
        hole=0.3
    )
    fig = sp.make_subplots(rows=1, cols=2,
                          subplot_titles=("Bar Plot", "Pie Chart"),
                          specs=[[{"type": "bar"}, {"type": "pie"}]])
    fig.add_trace(bar_fig, row=1, col=1)
    fig.add_trace(pie_fig, row=1, col=2)
    fig.update_layout(height=500, width=700)
    st.plotly_chart(fig)

if st.button("Show Feature Distributions (Histograms)"):
    plt.figure(figsize=(20, 50))
    num_row = 1
    for col in X.columns:
        plt.subplot(11, 2, num_row)
        plt.title(f"Distribution of {col} Data", fontsize=22)
        sns.histplot(x=df[col], kde=True, hue=df['fetal_health'], palette='bright')
        plt.xlabel(col, fontsize=20)
        plt.ylabel('Frequency', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        num_row += 1
    st.pyplot(plt.gcf())
    plt.clf()

if st.button("Show Feature Distributions (Box Plots)"):
    plt.figure(figsize=(20, 50))
    num_row = 1
    for col in X.columns:
        plt.subplot(11, 2, num_row)
        plt.title(f"Distribution of {col} Data", fontsize=22)
        sns.boxplot(y=df[col], x=df['fetal_health'], hue=df['fetal_health'])
        plt.xlabel('fetal_health', fontsize=20)
        plt.ylabel(col, fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        num_row += 1
    st.pyplot(plt.gcf())
    plt.clf()

if st.button("Show Pairplot"):
    pairplot = sns.pairplot(df, hue='fetal_health', corner=True, palette='bright', height=3)
    st.pyplot(pairplot.fig)

if st.button("Show Correlation Heatmap"):
    plt.figure(figsize=(30, 30))
    sns.heatmap(df.corr(), annot=True, cmap='viridis')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    st.pyplot(plt.gcf())
    plt.clf()

# Model Evaluation Section
st.header("Model Evaluation")

# Logistic Regression
if st.button("Show Logistic Regression Results"):
    st.subheader("Logistic Regression")
    st.write(f"Accuracy: {log_accuracy:.4f}")
    st.write("Classification Report:")
    st.text(classification_report(y_test, log_y_pred))
    
    mcc_log = matthews_corrcoef(y_test, log_y_pred)
    st.write(f"Matthews Correlation Coefficient: {mcc_log:.4f}")
    
    st.write("Confusion Matrix:")
    cm_fig = plot_confusion_matrix_heatmap(y_test, log_y_pred, "Logistic Regression")
    st.pyplot(cm_fig)
    
    st.write("ROC Curve:")
    roc_fig = plot_multi_class_roc(log_model, X_test, y_test, 3)
    st.pyplot(roc_fig)

# SVM
if st.button("Show SVM Results"):
    st.subheader("Support Vector Machine (SVM)")
    st.write(f"Accuracy: {svm_accuracy:.4f}")
    st.write("Classification Report:")
    st.text(classification_report(y_test, svm_y_pred))
    
    mcc_svm = matthews_corrcoef(y_test, svm_y_pred)
    st.write(f"Matthews Correlation Coefficient: {mcc_svm:.4f}")
    
    st.write("Confusion Matrix:")
    cm_fig = plot_confusion_matrix_heatmap(y_test, svm_y_pred, "SVM")
    st.pyplot(cm_fig)
    
    st.write("ROC Curve:")
    roc_fig = plot_multi_class_roc(svm_model, X_test, y_test, 3)
    st.pyplot(roc_fig)

# Random Forest
if st.button("Show Random Forest Results"):
    st.subheader("Random Forest")
    st.write(f"Accuracy: {rf_accuracy:.4f}")
    st.write("Classification Report:")
    st.text(classification_report(y_test, rf_y_pred))
    
    mcc_rf = matthews_corrcoef(y_test, rf_y_pred)
    st.write(f"Matthews Correlation Coefficient: {mcc_rf:.4f}")
    
    st.write("Confusion Matrix:")
    cm_fig = plot_confusion_matrix_heatmap(y_test, rf_y_pred, "Random Forest")
    st.pyplot(cm_fig)
    
    st.write("ROC Curve:")
    roc_fig = plot_multi_class_roc(rf_model, X_test, y_test, 3)
    st.pyplot(roc_fig)

# Model Accuracy
if st.button("Show Model Comparison"):
    st.subheader("Model Accuracy Comparison")
    
    models = ['Logistic Regression', 'SVM', 'Random Forest']
    accuracies = [log_accuracy, svm_accuracy, rf_accuracy]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=models, y=accuracies, palette='Set2', ax=ax)
    
    ax.set_title("Model Comparison - Accuracy")
    ax.set_xlabel("Models")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.8, 1.0)
    
    st.pyplot(fig)
    plt.clf()


