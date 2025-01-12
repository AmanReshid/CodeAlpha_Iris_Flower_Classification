import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Introduction Section
st.title("Iris Flower Classification Dashboard")
st.markdown("""

### Contents of the Dashboard
1. **Data Exploration**: Visualize feature distributions and correlations to uncover patterns and relationships within the data.
2. **Predictive Modeling**: Leverage a Random Forest Classifier to classify iris species with high accuracy.
3. **Interactive Prediction**: Input custom feature values to predict iris species in real time.
4. **Model Performance Insights**: Evaluate the model using performance metrics.

""")

# Load Iris Dataset
data = load_iris()
iris_df = pd.DataFrame(data=data.data, columns=data.feature_names)
iris_df['species'] = data.target
species_mapping = {i: name for i, name in enumerate(data.target_names)}
iris_df['species'] = iris_df['species'].map(species_mapping)

# Split Data
X = iris_df[data.feature_names]
y = iris_df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Dataset Overview
st.header("Dataset Overview")
st.markdown("""
The Iris dataset consists of 150 samples of iris flowers, categorized into three species: *Setosa*, *Versicolor*, and *Virginica*. Each sample includes four key features:
- **Sepal Length (cm)**
- **Sepal Width (cm)**
- **Petal Length (cm)**
- **Petal Width (cm)**

            """)
st.dataframe(iris_df)

# Exploratory Data Analysis
st.header("Exploratory Data Analysis")

# Feature Distributions
st.subheader("Feature Distributions")
fig, ax = plt.subplots(figsize=(10, 8))
iris_df.hist(ax=ax, bins=20)
plt.suptitle("Feature Distributions")
st.pyplot(fig)

# Pair Plot
st.subheader("Pair Plot")
st.write("Visualizing pairwise relationships between features.")
fig = sns.pairplot(iris_df, hue='species', diag_kind='kde', palette = 'Dark2')
st.pyplot(fig)

# Bar Plot
st.subheader("Bar Plot of Sepal Length by Species")
fig, ax = plt.subplots()
sns.barplot(x='species', y=data.feature_names[0], data=iris_df, ax=ax)
st.pyplot(fig)

# Swarm Plot
st.subheader("Swarm Plot of Sepal Width by Species")
fig, ax = plt.subplots()
sns.swarmplot(x='species', y=data.feature_names[1], data=iris_df, ax=ax)
st.pyplot(fig)

# Scatter Plot
st.subheader("Scatter Plot of Sepal Length vs Petal Length")
fig, ax = plt.subplots()
sns.scatterplot(x=data.feature_names[0], y=data.feature_names[2], hue='species', data=iris_df, palette='deep', ax=ax)
st.pyplot(fig)

# Correlation Heatmap
st.subheader("Feature Correlation Heatmap")
plt.figure(figsize=(10, 6))
sns.heatmap(iris_df.corr(numeric_only=True), annot=True, cmap="coolwarm")
st.pyplot(plt)

# Predict Iris Species
st.header("Predict Iris Species")
sl = st.slider("Sepal Length (cm)", float(iris_df[data.feature_names[0]].min()), float(iris_df[data.feature_names[0]].max()), float(iris_df[data.feature_names[0]].mean()))
sw = st.slider("Sepal Width (cm)", float(iris_df[data.feature_names[1]].min()), float(iris_df[data.feature_names[1]].max()), float(iris_df[data.feature_names[1]].mean()))
pl = st.slider("Petal Length (cm)", float(iris_df[data.feature_names[2]].min()), float(iris_df[data.feature_names[2]].max()), float(iris_df[data.feature_names[2]].mean()))
pw = st.slider("Petal Width (cm)", float(iris_df[data.feature_names[3]].min()), float(iris_df[data.feature_names[3]].max()), float(iris_df[data.feature_names[3]].mean()))

# Make Prediction
if st.button("Predict"):
    user_input = np.array([[sl, sw, pl, pw]])
    prediction = model.predict(user_input)
    st.success(f"The predicted species is: **{prediction[0]}**")

# Model Performance
st.header("Model Performance")
accuracy = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy**: {accuracy:.2f}")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
st.pyplot(fig)

st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)
