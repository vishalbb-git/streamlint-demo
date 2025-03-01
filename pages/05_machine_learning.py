import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import time

# Try importing seaborn, but provide a fallback
try:
    import seaborn as sns
    seaborn_available = True
except ImportError:
    seaborn_available = False
    st.warning("📊 Seaborn library is not installed. Some visualizations will be limited. To install it, run: `pip install seaborn`")
    # Define a simple replacement for sns.heatmap
    class SimpleSns:
        def heatmap(self, data, annot=True, cmap='coolwarm', ax=None):
            if ax is None:
                _, ax = plt.subplots()
            im = ax.imshow(data, cmap=cmap)
            plt.colorbar(im, ax=ax)
            
            # Add annotations
            if annot:
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        ax.text(j, i, f"{data.iloc[i, j]:.2f}", 
                                ha="center", va="center", color="black")
            
            ax.set_xticks(np.arange(len(data.columns)))
            ax.set_yticks(np.arange(len(data.index)))
            ax.set_xticklabels(data.columns)
            ax.set_yticklabels(data.index)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            return ax
            
        def kdeplot(self, data, label=None, ax=None):
            if ax is None:
                _, ax = plt.subplots()
            ax.hist(data, density=True, alpha=0.5, label=label)
            return ax
            
        def barplot(self, x, y, data, ax=None):
            if ax is None:
                _, ax = plt.subplots()
            ax.bar(data[x], data[y])
            return ax
            
        def pairplot(self, data, vars=None, hue=None):
            fig, ax = plt.subplots(len(vars), len(vars), figsize=(10, 10))
            for i, vi in enumerate(vars):
                for j, vj in enumerate(vars):
                    if i == j:  # Diagonal: histograms
                        ax[i, j].hist(data[vi], alpha=0.5)
                        ax[i, j].set_xlabel(vi)
                    else:  # Off-diagonal: scatter plots
                        ax[i, j].scatter(data[vj], data[vi], alpha=0.5)
                        ax[i, j].set_xlabel(vj)
                        ax[i, j].set_ylabel(vi)
            fig.tight_layout()
            return fig
    
    # Create a simple replacement for seaborn
    sns = SimpleSns()

st.set_page_config(page_title="Machine Learning Demo", page_icon="🤖", layout="wide")
st.title("🤖 Machine Learning Demonstration")
st.sidebar.header("Machine Learning Options")

# Step 1: Data Acquisition
st.header("1. Data Selection and Exploration")

dataset_options = {
    "Iris Dataset": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    "Wine Quality Dataset": "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/refs/heads/master/Datasets/wine.data.csv",
    "Breast Cancer Dataset": "https://raw.githubusercontent.com/selva86/datasets/master/BreastCancer.csv"
}

selected_dataset = st.selectbox("Select a dataset", list(dataset_options.keys()))

@st.cache_data
def load_data(url, dataset_name):
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        if dataset_name == "Wine Quality Dataset":
            # Wine dataset has no header
            column_names = ['Class', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 
                           'Magnesium', 'Total_phenols', 'Flavanoids', 
                           'Nonflavanoid_phenols', 'Proanthocyanins', 
                           'Color_intensity', 'Hue', 'OD280/OD315', 'Proline']
            df = pd.read_csv(io.StringIO(response.text), header=None, names=column_names)
        else:
            df = pd.read_csv(io.StringIO(response.text))
            
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# Load the data
data = load_data(dataset_options[selected_dataset], selected_dataset)

if data is not None:
    # Display dataset overview
    st.subheader("Dataset Preview")
    st.dataframe(data.head())
    
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Rows:** {data.shape[0]}")
        st.write(f"**Columns:** {data.shape[1]}")
    with col2:
        st.write("**Data Types:**")
        st.write(data.dtypes)
    
    # Data Exploration
    st.subheader("Data Exploration")
    
    if st.checkbox("Show Statistics"):
        st.write(data.describe())
        
    # Handle target column
    if selected_dataset == "Iris Dataset":
        target_column = "species"
    elif selected_dataset == "Wine Quality Dataset":
        target_column = "Class"
    else:  # Breast Cancer Dataset
        target_column = "Class"
    
    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        st.warning("Dataset contains missing values")
        st.write(missing_values[missing_values > 0])
        
        # Handle missing values
        if st.checkbox("Drop rows with missing values"):
            data = data.dropna()
            st.success(f"Dropped rows with missing values. New shape: {data.shape}")
    
    # Feature selection
    st.subheader("Feature Selection")
    
    # Get numeric columns
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if target_column in numeric_columns:
        numeric_columns.remove(target_column)
    
    # Allow user to select features
    selected_features = st.multiselect(
        "Select features for model training",
        options=numeric_columns,
        default=numeric_columns[:3] if len(numeric_columns) > 3 else numeric_columns
    )
    
    # Visualizations
    st.subheader("Data Visualization")
    
    visualization_type = st.selectbox(
        "Select Visualization Type",
        ["Pair Plot", "Correlation Heatmap", "Feature Distribution"]
    )
    
    if visualization_type == "Pair Plot" and len(selected_features) >= 2:
        if len(selected_features) <= 5:  # Limit features for pair plot to avoid overwhelm
            fig = sns.pairplot(data, vars=selected_features, hue=target_column if target_column in data.columns else None)
            st.pyplot(fig)
        else:
            st.warning("Select 5 or fewer features for pair plot visualization")
            
    elif visualization_type == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_data = data[selected_features].corr()
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
    elif visualization_type == "Feature Distribution":
        feature_to_plot = st.selectbox("Select feature to plot", selected_features)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if target_column in data.columns:
            for class_value in data[target_column].unique():
                subset = data[data[target_column] == class_value]
                sns.kdeplot(subset[feature_to_plot], label=f"{target_column}={class_value}", ax=ax)
            plt.legend()
        else:
            sns.histplot(data[feature_to_plot], kde=True, ax=ax)
            
        plt.title(f"Distribution of {feature_to_plot}")
        plt.xlabel(feature_to_plot)
        st.pyplot(fig)
    
    # Step 2: Model Selection and Training
    st.header("2. Model Selection and Training")
    
    # Prepare data for modeling
    if len(selected_features) > 0 and target_column in data.columns:
        X = data[selected_features]
        y = data[target_column]
        
        # Check if target needs encoding
        if y.dtype == 'object':
            st.info(f"Target column '{target_column}' will be label encoded")
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.write("Encoded classes:", dict(zip(le.classes_, le.transform(le.classes_))))
        
        # Train/test split
        test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        st.write(f"Training set: {X_train.shape[0]} samples")
        st.write(f"Test set: {X_test.shape[0]} samples")
        
        # Standardize features
        if st.checkbox("Standardize features", value=True):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            st.success("Features standardized")
        
        # Model selection
        model_options = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Support Vector Machine": SVC(probability=True)
        }
        
        selected_model = st.selectbox("Select a model", list(model_options.keys()))
        
        # Hyperparameters based on model
        st.subheader("Model Hyperparameters")
        
        model = None
        if selected_model == "Logistic Regression":
            C = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
            solver = st.selectbox("Solver", ["liblinear", "lbfgs", "saga"])
            max_iter = st.slider("Max Iterations", 100, 1000, 100)
            model = LogisticRegression(C=C, solver=solver, max_iter=max_iter, random_state=42)
            
        elif selected_model == "Random Forest":
            n_estimators = st.slider("Number of trees", 10, 200, 100)
            max_depth = st.slider("Max depth", 1, 20, 10)
            min_samples_split = st.slider("Min samples to split", 2, 10, 2)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                          min_samples_split=min_samples_split, random_state=42)
            
        elif selected_model == "Gradient Boosting":
            n_estimators = st.slider("Number of estimators", 50, 500, 100)
            learning_rate = st.slider("Learning rate", 0.01, 0.3, 0.1)
            max_depth = st.slider("Max depth", 1, 10, 3)
            model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                              max_depth=max_depth, random_state=42)
            
        elif selected_model == "Support Vector Machine":
            C = st.slider("Regularization (C)", 0.1, 10.0, 1.0)
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
            gamma = st.selectbox("Gamma", ["scale", "auto"])
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)
        
        # Train the model
        if st.button("Train Model"):
            with st.spinner(f"Training {selected_model}..."):
                start_time = time.time()
                model.fit(X_train, y_train)
                end_time = time.time()
                
                st.success(f"Model trained in {end_time - start_time:.2f} seconds")
                
                # Step 3: Model Evaluation
                st.header("3. Model Evaluation")
                
                # Make predictions
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                st.write(f"**Accuracy:** {accuracy:.4f}")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                st.pyplot(fig)
                
                # Classification Report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
                
                # Feature Importance (for applicable models)
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
                    plt.title('Feature Importance')
                    st.pyplot(fig)
                
                # Visualize decision boundaries (for 2D data)
                if len(selected_features) >= 2:
                    st.subheader("Decision Boundary Visualization")
                    
                    # Use PCA for dimensionality reduction if needed
                    if len(selected_features) > 2:
                        st.info("Using PCA to visualize high-dimensional data in 2D")
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X)
                        pca_feature_names = ['PCA Component 1', 'PCA Component 2']
                        X_pca_df = pd.DataFrame(X_pca, columns=pca_feature_names)
                        X_pca_df['target'] = y
                        
                        # Train a new model on PCA components for visualization
                        X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
                            X_pca, y, test_size=test_size, random_state=42)
                        
                        pca_model = model_options[selected_model]
                        pca_model.fit(X_train_pca, y_train_pca)
                        
                        # Create meshgrid for contour plot
                        h = 0.02  # step size in the mesh
                        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
                        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
                        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                           np.arange(y_min, y_max, h))
                        
                        Z = pca_model.predict(np.c_[xx.ravel(), yy.ravel()])
                        Z = Z.reshape(xx.shape)
                        
                        # Plot decision boundary and points
                        fig, ax = plt.subplots(figsize=(10, 8))
                        plt.contourf(xx, yy, Z, alpha=0.3)
                        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', alpha=0.8)
                        plt.xlabel('PCA Component 1')
                        plt.ylabel('PCA Component 2')
                        plt.title('Decision Boundary (PCA)')
                        plt.colorbar(scatter)
                        st.pyplot(fig)
                        
                        # Show variance explained by PCA
                        st.info(f"Variance explained: {pca.explained_variance_ratio_.sum():.2f}")
                    else:
                        # Directly use the first two features
                        feature1, feature2 = selected_features[:2]
                        X_2d = X[[feature1, feature2]]
                        
                        X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
                            X_2d, y, test_size=test_size, random_state=42)
                        
                        model_2d = model_options[selected_model]
                        model_2d.fit(X_train_2d, y_train_2d)
                        
                        # Create meshgrid for contour plot
                        h = 0.02  # step size in the mesh
                        x_min, x_max = X_2d[feature1].min() - 1, X_2d[feature1].max() + 1
                        y_min, y_max = X_2d[feature2].min() - 1, X_2d[feature2].max() + 1
                        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                           np.arange(y_min, y_max, h))
                        
                        Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
                        Z = Z.reshape(xx.shape)
                        
                        # Plot decision boundary and points
                        fig, ax = plt.subplots(figsize=(10, 8))
                        plt.contourf(xx, yy, Z, alpha=0.3)
                        scatter = plt.scatter(X_2d[feature1], X_2d[feature2], c=y, edgecolors='k', alpha=0.8)
                        plt.xlabel(feature1)
                        plt.ylabel(feature2)
                        plt.title('Decision Boundary')
                        plt.colorbar(scatter)
                        st.pyplot(fig)
                
                # Step 4: Make Predictions
                st.header("4. Make Predictions")
                st.subheader("Prediction on New Data")
                
                # Create input form for each feature
                st.write("Enter values for prediction:")
                
                input_data = {}
                col1, col2 = st.columns(2)
                
                for i, feature in enumerate(selected_features):
                    # Alternate between columns
                    with col1 if i % 2 == 0 else col2:
                        feature_min = float(data[feature].min())
                        feature_max = float(data[feature].max())
                        feature_mean = float(data[feature].mean())
                        input_data[feature] = st.slider(
                            f"{feature}", 
                            min_value=feature_min,
                            max_value=feature_max,
                            value=feature_mean
                        )
                
                if st.button("Predict"):
                    # Format input data for prediction
                    input_df = pd.DataFrame([input_data])
                    
                    # Apply same preprocessing as training data
                    if 'scaler' in locals():
                        input_array = scaler.transform(input_df)
                    else:
                        input_array = input_df
                    
                    # Make prediction
                    prediction = model.predict(input_array)[0]
                    
                    # Get probability if available
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(input_array)[0]
                        proba_df = pd.DataFrame({
                            'Class': range(len(proba)),
                            'Probability': proba
                        })
                        
                        # If we have original class labels, use them
                        if 'le' in locals():
                            proba_df['Class'] = le.inverse_transform(proba_df['Class'])
                    
                    # Display prediction
                    st.success(f"Predicted class: {prediction if 'le' not in locals() else le.inverse_transform([prediction])[0]}")
                    
                    # Display probabilities if available
                    if hasattr(model, 'predict_proba'):
                        st.write("Prediction Probabilities:")
                        
                        # Bar chart of probabilities
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Class', y='Probability', data=proba_df, ax=ax)
                        plt.title('Prediction Probabilities')
                        plt.ylim(0, 1)
                        plt.ylabel('Probability')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
    else:
        st.error("Please select at least one feature and ensure target column exists in the dataset")
else:
    st.error("Failed to load dataset. Please try again later.")

st.sidebar.info("This page demonstrates the machine learning workflow from data loading to model training and evaluation.")
