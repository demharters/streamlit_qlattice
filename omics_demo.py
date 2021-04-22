import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import base64
import textwrap
import feyn
from sklearn.model_selection import train_test_split
import time
import streamlit.components.v1 as components
from PIL import Image

# Run the QLattice

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    #st.write(html, unsafe_allow_html=True)
    return(html)

def read_data(d):
    filename = "datasets/" + dataset + ".csv"
    df = pd.read_csv(filename)
    return(df)

def get_description(d):
    if d == "COVID":
        my_text = f"""
        - Data file: {d}.csv
        - Type: Proteomics (Mass Spec)
        - paper: https://www.biorxiv.org/content/10.1101/2020.11.02.365536v2
        - 305 cases, 78 controls
        - 366 proteins (Inflammation Panel)
        - train-test split: 0.66/0.33"""
    elif d == "Alzheimer":
        my_text = f"""
        - Data file: {d}.csv
        - Type: Proteomics (O-link)
        - paper: https://www.embopress.org/doi/full/10.15252/msb.20199356
        - tutorial: https://docs.abzu.ai/docs/tutorials/python/ad_omics.html
        - 49 cases, 88 controls
        - 1166 proteins
        - train-test split: 0.66/0.33"""
    elif d == "Cancer":
        my_text = f"""
        - Data file: {d}.csv
        - Type: methylation profiles of HCC (Hepatocellular Carcinoma) tumour DNA
        - paper: https://pubmed.ncbi.nlm.nih.gov/26516143/
        - 36 cases, 55 controls
        - 1712 ctDNA methylation markers
        - train-test split: 0.4/0.6"""
    elif d == "ASO_Toxicity":
        my_text = f"""
        - Data file: {d}.csv
        - Type: caspase activation
        - paper: https://doi.org/10.1016/j.omtn.2019.12.011
        - 772 sequences
        - train-test split: 0.66/0.33"""
    return(my_text)

# Header
image = Image.open("/Users/demharters/git/streamlit_feyn/images/qlattice.jpg")
st.image(image)

st.title('Omics data and the QLattice:')

# Set seed
np.random.seed(42)

# Dataset Select0r
datasets = ["COVID", "Alzheimer", "Cancer"]
dataset = st.selectbox('Select Dataset:', datasets)

# Data Loading
data = read_data(dataset)
with st.beta_expander("Data sample:"):
    st.dataframe(data)

# Dataset Descriptions
with st.beta_expander("Data description"):
    descr = get_description(dataset)
    st.markdown(descr)

# QLattice Settings
with st.beta_expander("Settings"):
    max_depth = st.slider("Number of Layers", min_value=1, max_value=6, value = 3)
    n_iter = st.slider("Number of Iterations", min_value=1, max_value=20, value = 3)

# Spacer
st.text("  ")

if st.button('Run QLattice'):
    
    # Status Box
    model_iter_holder = st.empty()
    model_iter_holder.write("Running ..")

    # Target variable & random seed
    target = 'target'
    random_seed = 8989

    # Split data
    model_iter_holder.write("Splitting Data into Train and Test ..")

    # Smaller training set for "Cancer" to avoid overfitting
    if dataset == "Cancer":
        train, test = train_test_split(data, test_size=0.6, stratify=data[target], random_state=random_seed)
    else:
        train, test = train_test_split(data, test_size=0.33, stratify=data[target], random_state=random_seed)
    time.sleep(1)
    # Run the qlattice
    model_iter_holder.write("Firing up the QLattice ..")
    ql = feyn.QLattice()
    ql.reset(random_seed)
    
    # Declaring categorical features
    stypes = {}
    for f in data.columns:
        if data[f].dtype =='object':
            stypes[f] = 'c'
    
    qg = ql.get_classifier(data.columns, target, stypes=stypes, max_depth=max_depth)
    
    #st.write("Progress")
    my_bar = st.progress(0)
    model_graph_holder = st.empty()
    
    model_iter_holder.write("Start Training ..")
    time.sleep(1)
    #n_iter = 2
    for i in range(n_iter):
        qg.fit(train, threads=8, show=None)
        model_graph_holder.markdown(render_svg(qg[0]._repr_svg_()),unsafe_allow_html=True)
        model_iter_holder.write(f"Iteration:{i+1}/{n_iter}")
        ql.update(qg.best())
        my_bar.progress(i/n_iter + 1/n_iter)
    
    model_iter_holder.write("Done")

    with st.beta_expander("Model Validation on the Training Data"):
        # Plot Summary
        components.html(qg[0].plot_summary(train)._repr_html_(), height = 300, width = 800)
    
        col1, col2 = st.beta_columns(2)
        fig_roc, ax = plt.subplots()
        qg[0].plot_roc_curve(train)
        col1.pyplot(fig_roc)
        fig_confusion, ax = plt.subplots()
        qg[0].plot_confusion_matrix(train)
        col2.pyplot(fig_confusion)

    with st.beta_expander("Model Validation on the Test Data"):
        # Plot Summary
        components.html(qg[0].plot_summary(test)._repr_html_(), height = 300, width = 800)
    
        col1, col2 = st.beta_columns(2)
        fig_roc, ax = plt.subplots()
        qg[0].plot_roc_curve(test)
        col1.pyplot(fig_roc)
        fig_confusion, ax = plt.subplots()
        qg[0].plot_confusion_matrix(test)
        col2.pyplot(fig_confusion)
