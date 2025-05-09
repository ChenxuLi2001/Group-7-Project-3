
## 📊 Visualizations

### PCA Scree Plot (Top 5 Components)  
![PCA Scree Plot](visualizations/Picture1.png)  
*Figure 1. Variance explained by top 5 PCA components.*

Although the first two components captured most of the variance, many encoded features had limited interpretability. ProjectStatus was the main driver in PC1, while OutputYear dominated PC2.

---

### Clustering Scatter Plot (UMAP)  
![Clustering Scatter](visualizations/Picture2.png)  
*Figure 2. UMAP projection of clustered research outputs.*

K-Means clustering revealed ten thematic document groups. Some clusters were tightly bound (indicating strong topic cohesion), while others overlapped, suggesting related subfields.

---

### Publications Over Time  
![Publications](visualizations/Picture3.png)  
*Figure 3. Number of FSRDC-related publications per year.*

Publication activity has grown significantly since 2000, peaking between 2020–2022. This trend reflects the increasing use of FSRDC data in academic research.

---

### Citation Distribution  
![Citation Distribution](visualizations/Picture4.png)  
*Figure 4. Distribution of citation counts across outputs.*

Most outputs have modest citation impact, but a few highly cited papers drive the right-skewed tail — potentially landmark or review studies.

---

### Output Count Histogram  
![Output Count Histogram](visualizations/Picture5.png)  
*Figure 5. Distribution of number of outputs per project.*

Many projects yield a single publication, while a smaller number produce multiple outputs, reflecting different research scales or data reuse.

---

### Confusion Matrix (Classification Model)  
![Confusion Matrix](visualizations/Picture6.png)  
*Figure 6. Confusion matrix for the model predicting output status.*

The classifier distinguishes published vs. unpublished well, but struggles with forthcoming outputs, likely due to class imbalance despite weighted training.

---

### PCA Scree Plot (Full)  
![PCA Scree Full](visualizations/Picture7.png)  
*Figure 7. Full PCA component variance plot.*

When using metadata like ProjectStart/End and OutputYear, two components explained most variance, with OutputYear and PageCount as top drivers.

---

### Network Degree Distribution  
![Network Degree](visualizations/Picture8.png)  
*Figure 8. Degree distribution in the document similarity graph.*

Most documents have 1–2 highly similar peers, confirming sparse overlap between research outputs. A few "hub" documents exist but are rare.

---

### Agglomerative Clustering  
![Agglomerative Clustering](visualizations/Picture9.png)  
*Figure 9. Hierarchical clustering via Ward linkage.*

Similar to K-Means, but with differences in cluster size. Some groups merged or split differently, confirming both the robustness and limits of thematic partitioning.

---

### LDA Topic Terms  
![LDA Topics](visualizations/Picture10.png)  
*Figure 10. Top keywords in each of the 5 LDA topics.*

The model uncovered coherent themes: census/survey data, public health, economic activity, and demographics. Minimal overlap among topics indicates strong distinctiveness.

---

### Sentiment Score Distribution  
![Sentiment Score](visualizations/Picture11.png)  
*Figure 11. Sentiment polarity of research abstracts.*

Abstracts are overwhelmingly neutral, with a slight skew toward positive. Very few documents show negative sentiment — consistent with academic tone.

---

### Authors Per Cluster  
![Authors per Cluster](visualizations/Picture12.png)  
*Figure 12. Author distribution across thematic clusters.*

One dominant cluster contains nearly half of all authors, showing shared research focus. Other clusters represent more niche academic communities.

---

### Word Frequency by Year  
![Word Frequency](visualizations/Picture13.png)  
*Figure 13. Temporal frequency of key terms in abstracts.*

Trends show increasing use of “census,” “health,” and “data” since 2010, while “manufacturing” declines — reflecting evolving research priorities.
