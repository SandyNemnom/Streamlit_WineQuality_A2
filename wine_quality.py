# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------
# Load data
@st.cache_data  # Cache the data to improve performance
def load_data_red():
    r_wine = pd.read_excel('winequality-red.xlsx')./
    return r_wine

r_wine = load_data_red()

def load_data_white():
    w_wine = pd.read_excel('winequality-white.xlsx')./
    return w_wine

w_wine = load_data_white()
# ---------------------
# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
r_wine = load_data_red()
w_wine = load_data_white()
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')
# ---------------------


# Display data
st.title('The Portuguese "Vinho Verde" wine')
st.write('This report aims to:')
st.write('1) Assess Wine Quality: Determine whether white or red wine exhibits a lower quality and explore potential improvements.')
st.write('2) Understand Wine Quality Factors: Investigate the various factors that influence the quality of wine.')

# ---------------------

st.subheader('Wine Quality per Wine Variant')

# Create a clustered quality distribution bar chart
fig, ax = plt.subplots()
width = 0.35  # Width of each bar

# Sidebar to select the dataset(s)
selected_datasets = st.multiselect("Select Dataset(s)", ["Red Wine", "White Wine"])

if selected_datasets:
    quality_counts_data = {}
    labels = []
    max_quality = 10  # Quality values range from 1 to 10
    
    for dataset_name in selected_datasets:
        if dataset_name == "Red Wine":
            quality_counts = r_wine['quality'].value_counts().sort_index()
            label = 'Red Wine'
        else:
            quality_counts = w_wine['quality'].value_counts().sort_index()
            label = 'White Wine'

        # Fill in missing quality values with zeros
        missing_values = set(range(1, max_quality + 1)) - set(quality_counts.index)
        for missing_value in missing_values:
            quality_counts[missing_value] = 0
        
        # Sort the index
        quality_counts = quality_counts.sort_index()
        
        quality_counts_data[label] = quality_counts.values
        labels.append(label)

    index = np.arange(max_quality)
    
    for i, label in enumerate(labels):
        ax.bar(
            index + i * width,
            quality_counts_data[label],
            width=width,
            label=label,
            align='center',
        )

    ax.set_xlabel('Quality')
    ax.set_ylabel('Count of Samples')
    ax.set_title('Wine Quality Distribution Comparison')
    ax.set_xticks(index + width * (len(labels) - 1) / 2)  # Center x-ticks
    ax.set_xticklabels(index + 1)  # Display quality values as x-tick labels
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("Please select at least one dataset for comparison.")
    
# ---------------------

st.subheader('Correlation Matrix for Wine Datasets')
st.write('')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Create a form for the dataset selection
with st.form("dataset_selection_form"):
    selected_dataset = st.radio("Choose a Dataset", ["Red Wine", "White Wine"])
    submit_button = st.form_submit_button("Generate Correlation Matrix")

# Only proceed if the form is submitted
if submit_button:
    # Create a correlation matrix for the selected dataset
    if selected_dataset == "Red Wine":
        dataset = r_wine
    else:
        dataset = w_wine

    # Compute the correlation matrix
    correlation_matrix = dataset.corr()

    # Display the correlation matrix using a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title(f'Correlation Matrix for {selected_dataset}')
    st.pyplot()
    
# --------------------- 

st.subheader('Scatterplot of Alcohol vs. Another Column')

# Create a form for the dataset and column selection
with st.form("scatterplot_form"):
    # Sidebar to select the dataset
    selected_dataset = st.selectbox("Select Dataset", ["Red Wine", "White Wine"])

    # Sidebar to select the column to plot against alcohol
    if selected_dataset == "Red Wine":
        selected_column = st.selectbox("Select Column", r_wine.columns)
    else:
        selected_column = st.selectbox("Select Column", w_wine.columns)

    # Submit button
    submit_button = st.form_submit_button("Generate Scatterplot")

# Only proceed if the form is submitted
if submit_button:

    if selected_dataset == "Red Wine":
        dataset = r_wine
    else:
        dataset = w_wine

    plt.figure(figsize=(10, 6))
    plt.scatter(dataset["alcohol"], dataset[selected_column], alpha=0.5)
    plt.xlabel("Alcohol Content")
    plt.ylabel(selected_column)
    plt.title(f'Scatterplot of Alcohol vs. {selected_column}')
    st.pyplot()







