#!/usr/bin/env python3
import time
import requests
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import os

def query_openalex(title, per_page=1):
    resp = requests.get(
        "https://api.openalex.org/works",
        params={"filter": f"title.search:{title}", "per-page": per_page},
        timeout=10
    )
    if resp.status_code == 200:
        return resp.json().get("results", [])
    else:
        return []

def query_crossref(title, rows=1, session=None):
    if session is None:
        session = requests.Session()
    url = "https://api.crossref.org/works"
    params = {"query.title": title, "rows": rows}
    try:
        resp = session.get(url, params=params, timeout=(5, 30))
        resp.raise_for_status()
        return resp.json().get("message", {}).get("items", [])
    except requests.exceptions.RequestException:
        return []

def enrich_with_citations(df):
    # prepare retry‑session
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["GET"])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    enriched = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="API Queries"):
        title = getattr(row, "OutputTitle", "")
        oa = query_openalex(title)
        if oa:
            enriched.append({**row._asdict(), "cited_by": oa[0].get("cited_by_count")})
            continue

        cr = query_crossref(title, session=session)
        if cr:
            enriched.append({**row._asdict(), "cited_by": cr[0].get("is-referenced-by-count")})
            continue

        enriched.append({**row._asdict(), "cited_by": None})
        time.sleep(1.0)

    return pd.DataFrame(enriched)

def make_plots(df, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Top 10 RDCs
    rdc_counts = df['ProjectRDC'].value_counts().nlargest(10)
    plt.figure()
    rdc_counts.plot(kind='bar')
    plt.title('Top 10 RDCs by Research Output Count')
    plt.ylabel('Number of Outputs')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot1_rdc_counts.png"))
    plt.close()

    # 2. Publications per Year
    df['OutputYear'] = pd.to_numeric(df['OutputYear'], errors='coerce')
    pubs_per_year = (df
        .dropna(subset=['OutputYear'])
        .OutputYear
        .astype(int)
        .value_counts()
        .sort_index()
    )
    plt.figure()
    pubs_per_year.plot(marker='o')
    plt.title('Publications per Year')
    plt.ylabel('Number of Publications')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot2_pubs_per_year.png"))
    plt.close()

    # 3. Top 10 Authors
    authors = df['ProjectPI'].dropna().str.split(r',\s*').explode()
    top10 = authors.value_counts().nlargest(10)
    plt.figure()
    top10.plot(kind='bar')
    plt.title('Top 10 Most Prolific Authors')
    plt.ylabel('Number of Publications')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot3_top_authors.png"))
    plt.close()

    # 4. Citation Count Distribution
    cit = pd.to_numeric(df['cited_by'], errors='coerce').dropna().astype(int)
    plt.figure()
    cit.hist(bins=20)
    plt.title('Distribution of Citation Counts')
    plt.xlabel('Citations')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot4_citation_dist.png"))
    plt.close()

    # 5. Lag Years (creative insight)
    lag = df['ProjectYearEnded'] - df['ProjectYearStarted']
    plt.figure()
    lag.hist(bins=10)
    plt.title('Time from Project Start to Publication')
    plt.xlabel('Years')
    plt.ylabel('Number of Papers')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot5_lag_years.png"))
    plt.close()

def main():
    # 0. Load your data
    df = pd.read_csv("FinalOutputWithAPIMetadata_Deduplicated.csv", encoding="latin-1")

    # 1. Enrich with citation counts
    df_enriched = enrich_with_citations(df)
    df_enriched.to_csv("eda_with_api.csv", index=False)

    # 2. Make all the EDA plots
    make_plots(df_enriched)

if __name__ == "__main__":
    main()
