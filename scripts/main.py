# main.py
# This script controls the full pipeline execution for FSRDC project analysis.
# It allows modular execution of each step from web scraping to visualization.

# Import all processing modules
import time                   # For tracking how long the pipeline takes
import data_processing        # Step 1: Input Processing
import EDA                    # Step 2: EDA Analysis
import Model_Analysis_PCA     # Step 3: Classification Model Analysis & PCA
import Clustering             # Step 4: Clustering Technique Analysis

# STEP 1: Run the data processing code
def step_1_data_processing():
    print("\nStep 1: Starting data processing...")
    try:
        EDA.data_processing()
        print("Data processing completed.\n")
    except Exception as e:
        print(f"Error in Step 1: {e}")

# STEP 2: Run the EDA Analysis code
def step_2_EDA_analysis():
    print("\nStep 2: Starting EDA analysis...")
    try:
        EDA.run_EDA()
        print("EDA analysis completed.\n")
    except Exception as e:
        print(f"Error in Step 2: {e}")

# STEP 3: Run the Classification Model Analysis & PCA code
def step_3_classification_PCA():
    print("\nStep 3: Starting Classification Model Analysis & PCA...")
    try:
        Model_Analysis_PCA.run_Model_Analysis_PCA()
        print("Classification Model Analysis & PCA completed.\n")
    except Exception as e:
        print(f"Error in Step 3: {e}")

# STEP 4: Clean, deduplicate, and FSRDC-filter the metadata
def step_4_Clustering():
    print("\nStep 4: Processing Clustering...")
    try:
        Clustering.run_Clustering()
        print("Clustering completed.\n")
    except Exception as e:
        print(f"Error in Step 4: {e}")

# Entrypoint: Run selected steps in sequence
if __name__ == "__main__":
    print("FSRDC Metadata Pipeline - Project 3")
    start_time = time.time()  # Record start time

    # === UNCOMMENT the steps you want to run ===
    # step_1_data_processing()
    # step_2_EDA_analysis()
    # step_3_classification_PCA()
    # step_4_Clustering()

    total_time = round(time.time() - start_time, 2)
    print(f"Pipeline completed in {total_time} seconds.")