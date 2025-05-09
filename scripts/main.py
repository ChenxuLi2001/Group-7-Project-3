
"""
main.py

Entry point for the FSRDC Project 3 pipeline.
Executes modular steps: data processing, EDA, modeling, and clustering.

Author: Group 7 - CIT5900 Spring 2025
"""

import time
import data_processing         # Step 1: Input Processing
import EDA                     # Step 2: Exploratory Data Analysis
import Model_Analysis_PCA      # Step 3: Classification and PCA
import Clustering              # Step 4: Clustering and Topic Modeling

def step_1_data_processing():
    """Step 1: Clean and enrich raw input data from all 8 group files."""
    print("\nStep 1: Starting data processing...")
    try:
        data_processing.run_data_processing()
        print("Data processing completed.\n")
    except Exception as e:
        print(f"Error in Step 1 (Data Processing): {e}")

def step_2_EDA_analysis():
    """Step 2: Generate exploratory plots and enrich citation data."""
    print("\nStep 2: Starting EDA analysis...")
    try:
        EDA.run_EDA()
        print("EDA analysis completed.\n")
    except Exception as e:
        print(f"Error in Step 2 (EDA): {e}")

def step_3_classification_PCA():
    """Step 3: Apply classification models and PCA."""
    print("\nStep 3: Starting Classification and PCA Analysis...")
    try:
        Model_Analysis_PCA.run_Model_Analysis_PCA()
        print("Classification and PCA analysis completed.\n")
    except Exception as e:
        print(f"Error in Step 3 (Modeling/PCA): {e}")

def step_4_clustering_analysis():
    """Step 4: Perform clustering and generate visual dashboards."""
    print("\nStep 4: Starting Clustering Analysis...")
    try:
        Clustering.run_Clustering()
        print("Clustering analysis completed.\n")
    except Exception as e:
        print(f"Error in Step 4 (Clustering): {e}")

if __name__ == "__main__":
    print("===== FSRDC Project 3 Pipeline Execution =====")
    start_time = time.time()

    # Uncomment the steps below to run each module
    step_1_data_processing()
    step_2_EDA_analysis()
    step_3_classification_PCA()
    step_4_clustering_analysis()

    duration = round(time.time() - start_time, 2)
    print(f"Pipeline completed in {duration} seconds.")
