# Breast Cancer Wisconsin (Diagnostic) Data Analysis and Classification

**What is it?**

This project analyzes the Breast Cancer Wisconsin (Diagnostic) dataset to classify tumors as either benign or malignant using a Maximum a Posteriori (MAP) classifier based on multivariate Gaussian distributions. 
It also explores the dataset visually and attempts to minimize the amount of type II errors. It is important to reduce the false negatives in a diagnosis situation.

**Project Files:**

*   **`breast_cancer_analysis.py`:** This is the main script that performs data loading, preprocessing, model training, prediction, and evaluation. It includes data visualization and confusion matrix creation, and error calculation.
*   **`breast_cancer_classifier.py`:** Contains the core classification logic using MAP. It's a more basic version compared to `breast_cancer_analysis.py`.
*   **`plots.py`:**  Contains functions for visualizing decision regions and boundaries for a simplified 2D Gaussian classification problem, not directly related to the breast cancer dataset but useful for understanding the underlying concepts.
* **`wdbc.data`**: The dataset used for the classification.

**How to Use**

1.  **Data Acquisition:** Ensure that the `wdbc.data` dataset file is located in the same directory as the Python scripts.
2.  **Run the Analysis:** Execute the `breast_cancer_analysis.py` script:
    ```bash
    python breast_cancer_analysis.py
    ```
    This will perform the analysis, train the model, make predictions, and display several plots and the confusion matrices.
3. **Running the other scripts:** Execute the `breast_cancer_classifier.py` script:
    ```bash
    python breast_cancer_classifier.py
    ```
    Execute the `plots.py` script:
     ```bash
    python plots.py
    ```
**Key Features**

*   **Data Loading and Preprocessing:** Loads the Wisconsin Breast Cancer dataset, maps the 'B' and 'M' diagnosis to 0 and 1, respectively, and splits the data into training and testing sets.
*   **Exploratory Data Analysis (EDA):**
    *   Visualizes the class distribution (benign vs. malignant).
    *   Generates a heatmap of feature correlations.
*   **MAP Classifier:**
    *   Calculates prior probabilities for each class.
    *   Estimates mean and covariance for benign and malignant classes based on training data.
    *   Uses multivariate Gaussian distributions for classification.
    *   Calculates the confusion matrix of the model.
    * Calculates the amount of type I and type II errors.
*   **Model Evaluation:**
    *   Generates a confusion matrix to evaluate the classifier's performance.
    *   Calculates and prints the count of Type I (false positive) and Type II (false negative) errors.
* **Type II error reduction**: reduces the false negatives by adjusting the threshold of the model.
*   **Visualization:**
    *   Displays the confusion matrices.
    *   Shows the probability density of one of the features.
    *  `plots.py` script: creates plots of decision regions for equal and unequal prior cases.

**Getting Started**

1.  **Clone the repository** (if you are using a repository).
2.  **Install dependencies:** Make sure you have the following Python libraries installed. You can install them using `pip`:
    ```bash
    pip install numpy pandas scikit-learn matplotlib seaborn scipy
    ```
3.  **Place `wdbc.data`:** Put the `wdbc.data` file in the project's directory.
4.  **Run `breast_cancer_analysis.py`:** Execute the main analysis script.
5. **Run `breast_cancer_classifier.py`:** Execute the basic classifier script.
6. **Run `plots.py`:** Execute the plots script.

**Notes**

*   The `reg_const` variable in `breast_cancer_analysis.py` is used for regularization when calculating the covariance matrices.
* The threshold variable is used to reduce the amount of false negatives.
*   The `plots.py` script's plots are for demonstrating decision regions and boundaries in a simplified 2D setting.
* `breast_cancer_classifier.py` does not visualize the data.
* This is a project focused on educational purposes, it is not meant for medical use.

