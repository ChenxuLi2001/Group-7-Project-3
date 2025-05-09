# main.py
# This script controls the full pipeline execution for FSRDC project analysis.
# It allows modular execution of each step from web scraping to visualization.

# Import all processing modules
import time                   # For tracking how long the pipeline takes
import EDA                    # Step 1: EDA Analysis
import Model_Analysis_PCA     # Step 2: Classification Model Analysis & PCA
import Clustering             # Step 3: Clustering Technique Analysis

# STEP 1: Run the EDA Analysis code
def step_1_EDA_analysis():
    print("\nStep 1: Starting EDA analysis...")
    try:
        EDA.run_EDA()
        print("EDA analysis completed.\n")
    except Exception as e:
        print(f"Error in Step 1: {e}")

# STEP 2: Run the Classification Model Analysis & PCA code
def step_2_classification_PCA():
    print("\nStep 2: Starting Classification Model Analysis & PCA...")
    try:
        Model_Analysis_PCA.run_Model_Analysis_PCA()
        print("Classification Model Analysis & PCA completed.\n")
    except Exception as e:
        print(f"Error in Step 2: {e}")

# STEP 3: Clean, deduplicate, and FSRDC-filter the metadata
def step_3_Clustering():
    print("\nStep 3: Processing Clustering...")
    try:
        Clustering.run_Clustering()
        print("Clustering completed.\n")
    except Exception as e:
        print(f"Error in Step 3: {e}")

# Entrypoint: Run selected steps in sequence
if __name__ == "__main__":
    print("FSRDC Metadata Pipeline - Project 3")
    start_time = time.time()  # Record start time

    # === UNCOMMENT the steps you want to run ===
    # step_1_EDA_analysis()
    # step_2_classification_PCA()
    # step_3_Clustering()

    total_time = round(time.time() - start_time, 2)
    print(f"Pipeline completed in {total_time} seconds.")