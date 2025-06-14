# Deep_Learning_MEG
UU Deep Learning (INFOMDLR) - Project 2 - Magnetoencephalography


### Group Number: 22

### Team Members:
1. **Mahshid Jafar Tajrishi - 2931788**
2. **Bar Melinarskiy - 2482975**
3. **Cis van Aken - 7295758**
4. **Simon van Klompenburg - 0903999**


## Project Overview
Magnetoencephalography (MEG) is a cutting-edge neuroimaging technique that enables the mapping of both temporal and spatial brain activity, making it a critical tool in cognitive neuroscience and clinical diagnosis. Despite its potential, MEG data analysis is challenging due to the high dimensionality and inherent noise in the signals.

This project focuses on the design and implementation of deep learning model for MEG-based task classification. The models analyze MEG recordings to predict the cognitive task a subject was performing during data acquisition. These tasks include resting state, mathematical and language processing, working memory, and motor tasks. By improving signal decoding and pattern recognition, the project aims to enhance non-invasive brain monitoring techniques.

Accurate classification of MEG tasks has significant real-world applications, such as early detection of neurological diseases (e.g., epilepsy and Alzheimer's) and monitoring treatment effectiveness. By leveraging deep learning's ability to identify complex patterns in high-dimensional data, this project contributes to more precise and scalable approaches for analyzing brain signals, surpassing the capabilities of traditional statistical methods.

One of the main aims of this project is to compare the performance btween Intra and Cross subject models.

## Project Structure
The project is organized as follows:
- ***notebook_intra.ipynb*** - Handles fine-tuning, full training, and evaluation of the Intra-Subject model.
- ***notebook_cross.ipynb*** - Handles fine-tuning, full training, and evaluation of the Cross-Subject model.
- ***figures.ipynb*** - Generates figures for all three models (Intra-Subject, Cross-Subject without Self-Attention, and Cross-Subject with Self-Attention).
- ***Model.py*** - Contains classes for the model architecture and dataset handling.
- ***utils.py*** - Includes utility functions for data preprocessing, training, evaluation, and other helper tasks.
- ***Output folder*** - Stores all generated figures and CSV results from the notebooks.
- ***Models folder*** - Contains JSON files with the chosen hyperparameters and .pth files of trained models.
- ***Data folder*** - Contains the initial datasets for training and testing (Intra and Cross), as well as preprocessed data (e.g., after downsampling and normalization).




## Prerequisites
- Python 3.12
- Required libraries: `torch`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `optuna`, `scipy`, `tqdm`, `scikit-learn`

## How to Run the Project
1. Clone the repository and navigate to the project directory.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the desired notebook to train the corresponding model and evaluate its performance.
4. Results, including evaluation metrics and plots, will be saved in the `Output/` directory.