"""
Project 3: All-in-One Research Output Data Pipeline
"""

import os
import re
import time
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# These functions are used throughout the pipeline to normalize and clean raw text, numeric fields, and DOIs
# Helper Functions 

# This function removes HTML tags, converts text to lowercase, removes punctuation, and collapses spaces.
def normalize_text(text):
    if pd.isna(text): return ""
    text = BeautifulSoup(str(text), "html.parser").get_text().lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# This function cleans numeric strings by removing spaces and trailing zeroes often introduced by spreadsheets.
def clean_numeric_string(text):
    if pd.isna(text): return ""
    text = str(text).strip().replace(" ", "")
    text = re.sub(r'\.0$', '', text)
    text = re.sub(r'0$', '', text) if re.match(r'.*\d0$', text) else text
    return text

# This function standardizes DOI fields by removing URL prefixes and invalid characters, returning a proper DOI URL.
def clean_doi(doi):
    if pd.isna(doi): return ""
    doi = doi.lower()
    doi = doi.replace("httpsdoi", "").replace("https doi org", "").replace("doi.org", "")
    doi = re.sub(r'[^a-z0-9./]', '', doi)
    return f"https://doi.org/{doi.strip('/')}" if doi else ""

# Column Mapping 

# This dictionary maps inconsistent column names across CSV files to a unified schema.
column_mapping = {
    'project_id': 'ProjID', 'ProjectID': 'ProjID',
    'project_status': 'ProjectStatus', 'ProjectStatus': 'ProjectStatus',
    'ProjectTitle': 'ProjectTitle', 'Title': 'ProjectTitle',
    'Agency': 'ProjectRDC', 'RDC': 'ProjectRDC', 'location': 'ProjectRDC',
    'Start Year': 'ProjectYearStarted', 'ProjectStartYear': 'ProjectYearStarted',
    'End Year': 'ProjectYearEnded', 'ProjectEndYear': 'ProjectYearEnded',
    'project_pi': 'ProjectPI', 'PI': 'ProjectPI', 'researcher': 'ProjectPI', 'Researchers': 'ProjectPI',
    'title': 'OutputTitle', 'output_title': 'OutputTitle', 'Publication Title': 'OutputTitle',
    'biblio': 'OutputBiblio', 'citation': 'OutputBiblio',
    'type': 'OutputType', 'pub_type': 'OutputType',
    'status': 'OutputStatus', 'pub_status': 'OutputStatus',
    'venue': 'OutputVenue', 'publication_venue': 'OutputVenue',
    'year': 'OutputYear', 'pub_year': 'OutputYear',
    'month': 'OutputMonth', 'pub_month': 'OutputMonth',
    'volume': 'OutputVolume',
    'number': 'OutputNumber',
    'doi': 'DOI', 'DOI': 'DOI',
    'url': 'URL', 'link': 'URL',
    'abstract': 'Abstract', 'Abstract': 'Abstract'
}

# Step 1: Clean and Normalize CSV Files 

# Define the list of CSV files to be processed. Each file is expected to be named group1.csv to group8.csv.
csv_files = [f"group{i}.csv" for i in range(1, 9)]
cleaned_files = []

# Loop through each CSV file, normalize and clean the data, and save cleaned versions.
for fname in tqdm(csv_files, desc="Cleaning CSVs"):
    try:
        df = pd.read_csv(fname, dtype=str)
# Remove duplicated columns that may result from inconsistent headers or exports.
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

        new_data = {}
        for col in df.columns:
            mapped_col = column_mapping.get(col.strip())
            if mapped_col:
                new_data[mapped_col] = df[col].apply(normalize_text)
            else:
                new_data[col.strip()] = df[col]

        cleaned_df = pd.DataFrame(new_data)
        cleaned_name = f"cleaned_{fname}"
        cleaned_df.to_csv(cleaned_name, index=False)
        cleaned_files.append(cleaned_name)
        print(f"Saved: {cleaned_name}")
    except Exception as e:
        print(f"Failed to process {fname}: {e}")

# Step 2: Merge All Cleaned Files 

# Merge all individually cleaned CSV files into one DataFrame.
merged_df = pd.DataFrame()

for file in cleaned_files:
    try:
        df = pd.read_csv(file, dtype=str).fillna("")
        df["SourceFile"] = file
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    except Exception as e:
        print(f"Failed to load {file}: {e}")

merged_df.to_csv("CombinedFullRawOutputs.csv", index=False)
print("Saved merged output to CombinedFullRawOutputs.csv")

# Step 3: Deduplicate 

df = merged_df.copy()
# Remove duplicate rows based on OutputTitle and DOI, and discard entries with missing identifiers.
df = df.drop_duplicates(subset="OutputTitle", keep="first")
df = df.drop_duplicates(subset="DOI", keep="first")
if "DOI" in df.columns and "URL" in df.columns:
    df = df[~((df["DOI"].str.strip() == "") & (df["URL"].str.strip() == ""))]
else:
    print("Warning: 'DOI' and/or 'URL' column not found in merged data. Skipping DOI/URL filter.")
    print("Available columns:", df.columns.tolist())
df.to_csv("DeduplicatedFullOutputs.csv", index=False)
print("Saved DeduplicatedFullOutputs.csv")


# Step 4: Enrich from Metadata 

try:
    df = pd.read_csv("DeduplicatedFullOutputs.csv", dtype=str).fillna("")
# Load the metadata Excel sheet which contains authoritative project information.
    metadata = pd.read_excel("ProjectsAllMetadata.xlsx", sheet_name="All Metadata", dtype=str).fillna("")

# Normalizes PI names by removing HTML and punctuation to ensure consistent comparison.
    def normalize_name(name):
        if pd.isna(name): return ""
        name = BeautifulSoup(str(name), "html.parser").get_text().lower()
        name = re.sub(r'[^a-z0-9\s]', '', name)
        return re.sub(r'\s+', ' ', name).strip()

    df["norm_pi"] = df["ProjectPI"].apply(normalize_name)
    metadata["norm_pi"] = metadata["PI"].apply(normalize_name)

    # Keep only PIs that appear in ONE project
# Count occurrences of each normalized PI and keep only those associated with one project.
    pi_counts = metadata["norm_pi"].value_counts()
    valid_pis = pi_counts[pi_counts == 1].index
    metadata_valid = metadata[metadata["norm_pi"].isin(valid_pis)].copy()

    project_info = metadata_valid[[
        "norm_pi", "Proj ID", "Status", "Title", "RDC", "Start Year", "End Year", "PI"
    ]].copy()

    project_info.columns = [
        "norm_pi", "Meta_ProjID", "Meta_ProjectStatus", "Meta_ProjectTitle",
        "Meta_ProjectRDC", "Meta_ProjectYearStarted", "Meta_ProjectYearEnded", "Meta_ProjectPI"
    ]

    # Merge and fill missing values
# Enrich the main dataset by merging in project metadata for unambiguous PI matches.
    merged = pd.merge(df, project_info, how="left", on="norm_pi")

    for original, enriched in [
        ("ProjID", "Meta_ProjID"),
        ("ProjectStatus", "Meta_ProjectStatus"),
        ("ProjectTitle", "Meta_ProjectTitle"),
        ("ProjectRDC", "Meta_ProjectRDC"),
        ("ProjectYearStarted", "Meta_ProjectYearStarted"),
        ("ProjectYearEnded", "Meta_ProjectYearEnded"),
        ("ProjectPI", "Meta_ProjectPI")
    ]:
        merged[original] = merged[original].where(merged[original].str.strip() != "", merged[enriched])
        merged.drop(columns=[enriched], inplace=True)

    merged.drop(columns=["norm_pi"], inplace=True)
    merged.to_csv("FinalSafeEnrichedOutputs_by_PI.csv", index=False)
    print("Saved FinalSafeEnrichedOutputs_by_PI.csv")
except Exception as e:
    print("Metadata enrichment failed:", e)


# Step 5: Validate Using Fuzzy PI and Abstract Matching 

try:
# Load enriched dataset for validation using fuzzy PI and TF-IDF-based abstract comparison.
    df = pd.read_csv("FinalSafeEnrichedOutputs_by_PI.csv", dtype=str).fillna("")
# Load the metadata Excel sheet which contains authoritative project information.
    metadata = pd.read_excel("ProjectsAllMetadata.xlsx", sheet_name="All Metadata", dtype=str).fillna("")

    df["norm_pi"] = df["ProjectPI"].apply(normalize_text)
    df["norm_abstract"] = df["Abstract"].apply(normalize_text)
    metadata["norm_pi"] = metadata["PI"].apply(normalize_text)
    metadata["norm_abstract"] = metadata["Abstract"].apply(normalize_text)

    pi_threshold = 90
    pi_matches = []
    metadata_pis = metadata["norm_pi"].dropna().unique()

    for norm_pi in tqdm(df["norm_pi"], desc="Matching PI (fuzzy)"):
        best_score = max([fuzz.partial_ratio(norm_pi, meta_pi) for meta_pi in metadata_pis]) if norm_pi else 0
        pi_matches.append(best_score >= pi_threshold)

    df["PI_Match"] = pi_matches

    # Abstract similarity
    abstract_threshold = 0.6
    enriched_abs = df[df["norm_abstract"] != ""].copy()
    metadata_abs = metadata[metadata["norm_abstract"] != ""].copy()

# Compute TF-IDF matrix for comparing abstracts between enriched and metadata datasets.
    vectorizer = TfidfVectorizer().fit(enriched_abs["norm_abstract"].tolist() + metadata_abs["norm_abstract"].tolist())
    enriched_vectors = vectorizer.transform(enriched_abs["norm_abstract"])
    metadata_vectors = vectorizer.transform(metadata_abs["norm_abstract"])
    abs_sim_matrix = cosine_similarity(enriched_vectors, metadata_vectors)
    enriched_abs["Abstract_Match"] = abs_sim_matrix.max(axis=1) >= abstract_threshold

    # Merge back abstract match flag
    df = df.merge(enriched_abs[["Abstract", "Abstract_Match"]], on="Abstract", how="left")
    df["Abstract_Match"] = df["Abstract_Match"].fillna(False)

# Mark rows as valid if either PI or abstract match the metadata with high confidence.
    df["ValidatedFSRDC"] = df["PI_Match"] | df["Abstract_Match"]
    validated_df = df[df["ValidatedFSRDC"]].copy()
    validated_df.drop(columns=["norm_pi", "norm_abstract", "PI_Match", "Abstract_Match", "ValidatedFSRDC"], inplace=True)
    validated_df.to_csv("FinalFilteredOutputs_FuzzyPI_TFIDFAbstract.csv", index=False)

    print(f"Saved FinalFilteredOutputs_FuzzyPI_TFIDFAbstract.csv with {len(validated_df)} validated rows")
except Exception as e:
    print("Validation failed:", e)


# Step 6: Fix IDs and Clean DOIs 

try:
    df = pd.read_csv("FinalFilteredOutputs_FuzzyPI_TFIDFAbstract.csv", dtype=str).fillna("")
# Clean and fix format issues in ProjID and OutputYear fields.
    df["ProjID"] = df["ProjID"].apply(clean_numeric_string)
    df["OutputYear"] = df["OutputYear"].apply(clean_numeric_string)
    df["DOI"] = df["DOI"].apply(clean_doi)
    df.to_csv("FinalCleanedWithValidDOIs.csv", index=False)
    print("Saved FinalCleanedWithValidDOIs.csv")
except Exception as e:
    print("ID/DOI cleaning failed:", e)


# Step 7: Enrich Using CrossRef and OpenAlex APIs 

def normalize_title(title):
    return re.sub(r'\s+', ' ', BeautifulSoup(str(title), "html.parser").get_text().lower().strip())

# Query the CrossRef API using a publication title to retrieve bibliographic metadata.
def query_crossref(title):
    try:
        url = "https://api.crossref.org/works"
        params = {"query.title": title, "rows": 1}
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        items = res.json()["message"]["items"]
        return items[0] if items else None
    except:
        return None

# Query the OpenAlex API using a publication title to retrieve bibliographic metadata.
def query_openalex(title):
    try:
        url = "https://api.openalex.org/works"
        params = {"search": title, "per-page": 1}
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        items = res.json()["results"]
        return items[0] if items else None
    except:
        return None

try:
    df = pd.read_csv("FinalCleanedWithValidDOIs.csv", dtype=str).fillna("")
    df_enriched = df.copy()

# Create a dictionary mapping normalized titles to their original values for enrichment lookup.
    normalized_title_map = {
        normalize_title(title): title
        for title in df["OutputTitle"].dropna().unique()
    }

    target_cols = [
        "OutputBiblio", "OutputType", "OutputVenue", "OutputStatus",
        "OutputYear", "OutputMonth", "OutputVolume", "OutputNumber", "OutputPages"
    ]

    enriched_rows = set()

# Iterate through each unique title and enrich missing metadata using external APIs.
    for norm_title, original_title in tqdm(normalized_title_map.items(), desc="Enriching via APIs"):
        mask = df_enriched["OutputTitle"] == original_title
        sample_row = df_enriched[mask].iloc[0]

        if all(sample_row[col] != "" for col in target_cols):
            continue

        data = query_crossref(original_title) or query_openalex(original_title)
        if data is None:
            continue

        for idx in df_enriched[mask].index:
            enriched = False
            if df_enriched.at[idx, "OutputBiblio"] == "":
                df_enriched.at[idx, "OutputBiblio"] = data.get("citation", "") or data.get("display_name", "")
                enriched = True
            if df_enriched.at[idx, "OutputType"] == "":
                df_enriched.at[idx, "OutputType"] = data.get("type", "").upper()[:2]
                enriched = True
            if df_enriched.at[idx, "OutputVenue"] == "":
                venue = ""
                if "container-title" in data and isinstance(data["container-title"], list):
                    venue = data["container-title"][0]
                elif "host_venue" in data:
                    venue = data.get("host_venue", {}).get("display_name", "")
                df_enriched.at[idx, "OutputVenue"] = venue
                enriched = True
            if df_enriched.at[idx, "OutputStatus"] == "":
                df_enriched.at[idx, "OutputStatus"] = "PB" if "published" in data else "UP"
                enriched = True
            if df_enriched.at[idx, "OutputYear"] == "":
                try:
                    df_enriched.at[idx, "OutputYear"] = str(data.get("published", {}).get("date-parts", [[None]])[0][0])
                except:
                    df_enriched.at[idx, "OutputYear"] = str(data.get("publication_year", ""))
                enriched = True
            if df_enriched.at[idx, "OutputMonth"] == "":
                try:
                    df_enriched.at[idx, "OutputMonth"] = str(data.get("published", {}).get("date-parts", [[None, None]])[0][1])
                except:
                    pass
                enriched = True
            if df_enriched.at[idx, "OutputVolume"] == "":
                df_enriched.at[idx, "OutputVolume"] = data.get("volume", "")
                enriched = True
            if df_enriched.at[idx, "OutputNumber"] == "":
                df_enriched.at[idx, "OutputNumber"] = data.get("issue", "")
                enriched = True
            if df_enriched.at[idx, "OutputPages"] == "":
                df_enriched.at[idx, "OutputPages"] = data.get("page", "")
                enriched = True

            if enriched:
                enriched_rows.add(idx)

        time.sleep(0.5)  # API throttle

    df_enriched.to_csv("FinalOutputWithAPIMetadata.csv", index=False)
    print(f"Saved FinalOutputWithAPIMetadata.csv with {len(enriched_rows)} enriched rows")
except Exception as e:
    print("API enrichment failed:", e)


# Step 8: Merge with ResearchOutputs.xlsx and Deduplicate 

try:
    api_enriched_df = pd.read_csv("FinalOutputWithAPIMetadata.csv", dtype=str).fillna("")
# Load manually curated ResearchOutputs.xlsx and merge it with API-enriched outputs.
    research_outputs_df = pd.read_excel("ResearchOutputs.xlsx", dtype=str).fillna("")
    combined = pd.concat([api_enriched_df, research_outputs_df], ignore_index=True)

    def normalize_title_for_merge(title):
        if pd.isna(title): return ""
        title = str(title).lower()
        title = re.sub(r'[^a-z0-9\s]', '', title)
        return re.sub(r'\s+', ' ', title).strip()

    combined["norm_title"] = combined["OutputTitle"].apply(normalize_title_for_merge)
    combined_dedup = combined.drop_duplicates(subset="norm_title", keep="first").drop(columns=["norm_title"])
    combined_dedup.to_csv("FinalMergedResearchOutputs.csv", index=False)
    print("Saved FinalMergedResearchOutputs.csv")
except Exception as e:
    print("Merging with ResearchOutputs.xlsx failed:", e)


# Step 9: Final Format Cleanup and Export 

try:
# Final cleanup: reorder columns, fill missing values, and ensure consistent formatting.
    df = pd.read_csv("FinalMergedResearchOutputs.csv", dtype=str).fillna("")
    fallback_columns = {
        "ProjID": ["ProjectID"],
        "ProjectYearStarted": ["ProjectStartYear"],
        "ProjectYearEnded": ["ProjectEndYear"]
    }

    for target_col, fallback_list in fallback_columns.items():
        for fallback_col in fallback_list:
            if fallback_col in df.columns:
                df[target_col] = df[target_col].where(df[target_col].str.strip() != "", df[fallback_col])

    for col in ["ProjID", "ProjectYearStarted", "ProjectYearEnded"]:
        df[col] = df[col].apply(lambda x: str(int(x)) if x.strip().isdigit() else x)

    final_columns = [
        "ProjID", "ProjectStatus", "ProjectTitle", "ProjectRDC",
        "ProjectYearStarted", "ProjectYearEnded", "ProjectPI",
        "OutputTitle", "OutputBiblio", "OutputType", "OutputStatus",
        "OutputVenue", "OutputYear", "OutputMonth", "OutputVolume",
        "OutputNumber", "OutputPages"
    ]

    for col in final_columns:
        if col not in df.columns:
            df[col] = ""

    final_df = df[final_columns]
    final_df.to_csv("FinalCleanedMergedResearchOutputs.csv", index=False)
    print("Saved FinalCleanedMergedResearchOutputs.csv")
except Exception as e:
    print("Final formatting failed:", e)
