# -*- coding: utf-8 -*-
"""
Text Clustering with Interactive Viz

This script loads text data, preprocesses it, extracts features,
performs clustering, topic modeling, sentiment analysis,
network construction, and generates an interactive HTML dashboard.
"""
import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import networkx as nx
import umap
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources once
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Ensure the working directory is the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


### Core pipeline functions ###

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load text data from a CSV file into a pandas DataFrame.

    Parameters:
        filepath: Path to the input CSV file
    Returns:
        DataFrame containing loaded data
    """
    logger.info(f"Loading data from {filepath}")
    if not os.path.isfile(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"Cannot find file: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return pd.read_csv(filepath)


def preprocess_texts(series: pd.Series) -> list:
    """
    Preprocess text series by tokenizing, lowercasing, removing stopwords,
    and lemmatizing tokens.

    Parameters:
        series: pandas Series of raw text strings
    Returns:
        List of cleaned text strings
    """
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    cleaned = []
    for text in series.fillna(''):
        tokens = word_tokenize(text) # Tokenize raw text
        tokens = [t.lower() for t in tokens if t.isalpha() and t.lower() not in stop_words]
        tokens = [lemmatizer.lemmatize(t) for t in tokens] # Lemmatize tokens
        cleaned.append(' '.join(tokens))
    return cleaned


def vectorize(texts, method='tfidf', max_df=0.8, min_df=1, ngram=(1,1)):
    """
    Convert a list of texts into a numerical feature matrix using TF-IDF or Count.

    Parameters:
        texts: list of str, preprocessed documents
        method: 'tfidf' or 'count' selection of vectorizer
        max_df: float, ignore terms with document frequency above this threshold
        min_df: int, ignore terms with document frequency below this threshold
        ngram: tuple (min_n, max_n) range for n-grams
    Returns:
        X: sparse matrix of shape (n_samples, n_features)
        vec: fitted vectorizer instance
    """
    kwargs = dict(token_pattern=r"(?u)\b\w+\b", max_df=max_df, min_df=min_df,
                  stop_words='english', ngram_range=ngram)
    vec = TfidfVectorizer(**kwargs) if method=='tfidf' else CountVectorizer(**kwargs)
    X = vec.fit_transform(texts)
    logger.info(f"Vectorized data shape: {X.shape}")
    return X, vec


def cluster_methods(X, k, linkage='ward'):
    """
    Perform clustering using KMeans and AgglomerativeClustering.

    Parameters:
        X: feature matrix (sparse or dense)
        k: int number of clusters
        linkage: linkage criterion for agglomerative clustering
    Returns:
        km_lbl: array of KMeans labels
        agg_lbl: array of Agglomerative clustering labels
    """
    km_lbl = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    agg_lbl = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(X.toarray())
    return km_lbl, agg_lbl


def find_optimal_k(X, kmin=2, kmax=10) -> int:
    """
    Determine the optimal number of clusters via silhouette score analysis.

    Parameters:
        X: feature matrix
        kmin: minimum number of clusters to try
        kmax: maximum number of clusters to try
    Returns:
        best_k: number of clusters with highest silhouette score
    """
    best_k, best_s = kmin, -1
    for kk in range(kmin, kmax+1):
        labels = KMeans(n_clusters=kk, random_state=42, n_init=10).fit_predict(X)
        s = silhouette_score(X, labels)
        logger.info(f"k={kk}, silhouette={s:.4f}")
        if s > best_s:
            best_k, best_s = kk, s
    return best_k


def reduce_dim(X, method='pca', dim=2):
    """
    Reduce high-dimensional data to lower dimensions for visualization.

    Parameters:
        X: feature matrix (sparse or dense)
        method: 'pca' or 'umap' technique
        dim: target number of dimensions (usually 2)
    Returns:
        Xr: array of reduced data shape (n_samples, dim)
    """
    if method == 'umap':
        Xr = umap.UMAP(n_components=dim, random_state=42).fit_transform(X)
    else:
        Xr = PCA(n_components=dim, random_state=42).fit_transform(X.toarray())
    logger.info(f"Reduced data shape: {Xr.shape}")
    return Xr


def topic_modeling(texts, n_topics=5):
    """
    Extract topics from documents using Latent Dirichlet Allocation (LDA).

    Parameters:
        texts: list of str, preprocessed documents
        n_topics: int, number of topics to extract
    Returns:
        topics: list of lists containing top terms for each topic
    """
    vec = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
    dtm = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    terms = vec.get_feature_names_out()
    topics = [[terms[idx] for idx in comp.argsort()[-10:][::-1]] for comp in lda.components_]
    return topics


def sentiment_scores(texts, lang='en') -> list:
    """
    Compute sentiment scores for each document using VADER for English texts.

    Parameters:
        texts: list of str, preprocessed documents
        lang: language code ('en' or 'zh')
    Returns:
        List of compound sentiment scores (floats)
    """
    if lang=='en':
        sia = SentimentIntensityAnalyzer()
        return [sia.polarity_scores(t)['compound'] for t in texts]
    return [0]*len(texts)


def build_network(X, thr=0.7) -> nx.Graph:
    """
    Build an undirected graph where edges link documents above a cosine similarity threshold.

    Parameters:
        X: feature matrix
        thr: float threshold for cosine similarity
    Returns:
        G: networkx Graph with weighted edges
    """
    sim = cosine_similarity(X)
    G = nx.Graph()
    n = sim.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if sim[i,j]>=thr:
                G.add_edge(i, j, weight=float(sim[i,j]))
    return G

### Utility for word frequency by year ###

def word_freq(texts, years):
    """
    Compute word frequency per year.

    Parameters:
        texts: list of str, cleaned text documents
        years: iterable of year values corresponding to each text
    Returns:
        dict[int, Counter]: mapping each valid year to a Counter of word frequencies
    """
    freq = {}
    for txt, yr in zip(texts, years):
        if pd.isna(yr): # Skip missing years
            continue
        try:
            year = int(yr) # Convert to integer year
        except ValueError:
            continue
        words = txt.split()
        freq.setdefault(year, Counter()).update(words)
    return freq

### Visualization functions ###

def viz_clusters(Xr, labels, texts=None):
    """
    Create a scatter plot of clustered points using Plotly Express.

    Parameters:
        Xr: array of reduced data (n_samples, 2)
        labels: cluster labels for each point
        texts: optional list of text strings for hover info
    Returns:
        Plotly Figure object for cluster scatter
    """
    df = pd.DataFrame({'x': Xr[:, 0], 'y': Xr[:, 1], 'cluster': labels.astype(str)})
    if texts:
        df['text'] = texts
    return px.scatter(
        df, x='x', y='y', color='cluster', hover_data=['text'] if texts else None,
        title='Cluster Scatter'
    )


def viz_topics(topics):
    """
    Create a bar chart visualizing top terms for each LDA topic.

    Parameters:
        topics: list of lists containing top terms per topic
    Returns:
        Plotly Figure object for topic terms bar chart
    """
    df = pd.DataFrame({f'Topic {i+1}': t for i, t in enumerate(topics)})
    melt = df.melt(var_name='Topic', value_name='Term')
    return px.bar(melt, x='Term', color='Topic', title='LDA Topic Terms')


def viz_sentiment(scores):
    """
    Create a histogram of sentiment scores.

    Parameters:
        scores: list of sentiment compound scores
    Returns:
        Plotly Figure object for sentiment distribution histogram
    """
    return px.histogram(x=scores, nbins=30, title='Sentiment Score Distribution')


def author_clusters(df, author_col, text_col, k) -> pd.Series:
    """
    Cluster authors by aggregating their documents and applying KMeans.

    Parameters:
        df: DataFrame containing author and text columns
        author_col: name of column with author identifiers
        text_col: name of column with text data
        k: int number of clusters to form
    Returns:
        pandas Series mapping each author to a cluster label
    """
    agg = df.groupby(author_col)[text_col].apply(lambda s: ' '.join(s.dropna()))
    Xa, _ = vectorize(agg.tolist())
    lbl = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Xa)
    return pd.Series(lbl, index=agg.index)


def viz_author(auth_lbl):
    """
    Create a bar chart showing number of authors per cluster.

    Parameters:
        auth_lbl: pandas Series of author cluster labels
    Returns:
        Plotly Figure object for authors per cluster
    """
    df = auth_lbl.value_counts().reset_index()
    df.columns = ['Cluster', 'Count']
    return px.bar(df, x='Cluster', y='Count', title='Authors per Cluster')


def viz_multi_year_freq(texts, df, year_cols):
    """
    Generate an interactive line plot of word frequencies over multiple year columns.

    Parameters:
        texts: list of str, preprocessed documents
        df: DataFrame containing year columns
        year_cols: list of column names to include in frequency analysis
    Returns:
        Plotly Figure object with update menu for each year column
    """
    # compute freqs per column
    freqs = {col: word_freq(texts, df[col]) for col in year_cols if col in df.columns}
    # filter out unrealistic years (< 1000)
    for col in list(freqs.keys()):
        cleaned = {yr: cnt for yr, cnt in freqs[col].items() if yr >= 1000}
        if cleaned:
            freqs[col] = cleaned
        else:
            # remove column if no valid years remain
            del freqs[col]
    
    # union of top words across all
    top_words = set()
    for f in freqs.values(): top_words |= set(w for yr in f for w, _ in f[yr].most_common(5))
    
    # prepare multi-trace figure with update buttons
    fig = go.Figure()
    buttons = []
    for i, col in enumerate(freqs):
        f = freqs[col]
        years = sorted(f.keys())
        df_all = pd.DataFrame({
            'Year': [str(y) for y in years for _ in top_words],
            'Word': [w for _ in years for w in sorted(top_words)],
            'Count': [f[y].get(w, 0) for y in years for w in sorted(top_words)]
        })
        for word in sorted(top_words):
            fig.add_trace(go.Scatter(x=[str(y) for y in years], y=[f[y].get(word,0) for y in years], mode='lines', name=word, visible=(i==0)))
        buttons.append(dict(label=col, method='update', args=[{'visible': [j//len(years)==i for j in range(len(top_words)*len(freqs))]}, {'title': f"Word Frequency by {col}"}]))
    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=0.1, y=1.15, direction='right')],
        title=f"Word Frequency by {year_cols[0]}", xaxis_title='Year', yaxis_title='Count'
    )
    return fig


def viz_network(G):
    """
    Create a histogram of network degree distribution.

    Parameters:
        G: networkx Graph object
    Returns:
        Plotly Figure object for degree distribution
    """
    deg = [d for _, d in G.degree()]
    return px.histogram(x=deg, nbins=20, title='Network Degree Distribution')

### Main ###

def main(args):
    """
    Execute full analysis pipeline and save visualizations to HTML dashboard.

    Parameters:
        args: argparse.Namespace with input, parameters, and output settings
    """
    df = load_data(args.input)
    texts = preprocess_texts(df.get(args.text_col))
    X, _ = vectorize(
        texts,
        method=args.vectorizer,
        max_df=args.max_df,
        min_df=args.min_df,
        ngram=(args.ngram, args.ngram)
    )

    # Determine and log optimal k if not provided
    opt_k = find_optimal_k(X, args.k_min, args.k_max)
    logger.info(f"Optimal k by silhouette analysis: {opt_k}")
    k = args.k if args.k else opt_k

    # Perform clustering and dimensionality reduction
    km_lbl, agg_lbl = cluster_methods(X, k, linkage=args.linkage)
    Xr = reduce_dim(X, method=args.reducer, dim=args.dim)

    # Generate figures
    figs = []
    figs.append(viz_clusters(Xr, km_lbl, texts if args.hover else None))
    agg_fig = viz_clusters(Xr, agg_lbl, texts if args.hover else None)
    agg_fig.update_layout(title=f"Agglomerative Clustering ({args.linkage}, k={k})")
    figs.append(agg_fig)
    figs.append(viz_topics(topic_modeling(texts, args.n_topics)))
    figs.append(viz_sentiment(sentiment_scores(texts, args.sent_lang)))
    figs.append(viz_author(author_clusters(df, args.author_col, args.text_col, k)))

    # Multi-year word frequency analysis
    year_cols = ['OutputYear', 'ProjectYearStarted', 'ProjectYearEnded']
    figs.append(viz_multi_year_freq(texts, df, year_cols))

    # Network analysis
    G = build_network(X, thr=args.threshold)
    figs.append(viz_network(G))

    # Write all figures into a single HTML dashboard
    with open(args.output, 'w', encoding='utf-8') as out:
        out.write('<html><head><meta charset="utf-8"></head><body>\n')
        for i, fig in enumerate(figs, 1):
            out.write(fig.to_html(full_html=False, include_plotlyjs='cdn', div_id=f'figure_{i}'))
            out.write('<hr>\n')
        out.write('</body></html>\n')
    logger.info(f"All visualizations saved to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enhanced Text Clustering with Interactive Viz")
    parser.add_argument('--input', default='FinalCleanedMergedResearchOutputs.csv')
    parser.add_argument('--text_col', default='OutputTitle')
    parser.add_argument('--vectorizer', choices=['tfidf', 'count'], default='tfidf')
    parser.add_argument('--max_df', type=float, default=0.8)
    parser.add_argument('--min_df', type=int, default=1)
    parser.add_argument('--ngram', type=int, default=1)
    parser.add_argument('--k', type=int)
    parser.add_argument('--k_min', type=int, default=2)
    parser.add_argument('--k_max', type=int, default=10)
    parser.add_argument('--linkage', choices=['ward', 'complete', 'average'], default='ward')
    parser.add_argument('--reducer', choices=['pca', 'umap'], default='pca')
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--hover', action='store_true')
    parser.add_argument('--n_topics', type=int, default=5)
    parser.add_argument('--sent_lang', choices=['en', 'zh'], default='en')
    parser.add_argument('--author_col', default='ProjectPI')
    parser.add_argument('--year_col', default='OutputYear', help='Year column name for word freq')
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--output', default='figures.html')
    args = parser.parse_args()
    main(args)
