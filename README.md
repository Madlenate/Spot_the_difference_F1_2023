[README (1).md](https://github.com/user-attachments/files/31400216/README.1.md)
# Clustering Formula 1 Lap Data Using K-Means, DBSCAN, and Hierarchical Clustering

This project explores whether lap-level driving patterns in Formula 1 can be used to discover natural groupings among teams and laps. Using 2023 F1 qualifying session data from the [OpenF1 API](https://openf1.org/docs/), we engineered lap-level features and applied three unsupervised clustering methods to see whether the resulting clusters align with actual team identity — or reveal other underlying patterns in driving behavior.

**Authors:** Winnie Trinh ([witrinh@calpoly.edu](mailto:witrinh@calpoly.edu)), Nathan Madlansacay ([nmadlans@calpoly.edu](mailto:nmadlans@calpoly.edu))
Dept. of Computer Science / Dept. of Statistics, California Polytechnic State University, San Luis Obispo


LINK TOO Streamlit APP: [Spot The Difference F1 Analysis](https://madlenate-spot-the-difference-f1-2023-app-unwcdf.streamlit.app/)

---

## Overview

Formula 1 generates rich, lap-by-lap telemetry that reflects driver behavior, car performance, and track conditions. This project asks: **does that data contain enough structure for unsupervised clustering to recover meaningful groupings — and do those groupings correspond to real F1 teams?**

We built a custom dataset from OpenF1's 2023 qualifying session data, engineered features describing throttle, braking, gear usage, RPM, speed, and DRS behavior, and then applied K-means, DBSCAN, and hierarchical agglomerative clustering to the standardized feature set.

## Research Questions

1. Can Formula 1 data naturally cluster laps into meaningful groups?
2. Do the discovered clusters correspond to actual Formula 1 teams?
3. Which clustering method performs best on this dataset?
4. Can DBSCAN identify unusual laps or outliers in the data?
5. What do the clustering results suggest about the relationship between teams and lap behavior?

## Dataset

- **Source:** [OpenF1 API](https://openf1.org/docs/) — 2023 Formula 1 qualifying sessions
- **Raw data collected:** session metadata, driver info, lap records, stint data, weather data, and car telemetry samples
- **Filtering:**
  - Rainy sessions excluded (wet conditions distort comparisons)
  - Pit-out laps and incomplete/unrealistic laps removed
  - Only laps within **107%** of the session's best lap time retained
- **Processing:** Raw data saved to parquet and joined using [DuckDB](https://duckdb.org/docs/current/); car samples matched to laps by session key, driver number, and timestamp range
- **Final size:** 1,502 labeled lap records
- **Outputs:**
  - `lap_features_labeled.csv` — full feature set with team/driver labels
  - `f1_laps_scaled.csv` — standardized features (via `StandardScaler`) used for clustering, with true team labels stored separately

### Engineered Features

Average/std throttle, % time at full/zero throttle, % time braking, average gear, unique gears used, average/std/max RPM, max/average/min speed, speed std, % time in speed ranges, % time with DRS active, and number of gear changes.

## Methods

| Method | Description | Key Parameters |
|---|---|---|
| **K-Means** | Partitions data into a fixed number of clusters by iteratively assigning points to the nearest centroid | `k = 10` (matching the 10 F1 teams) |
| **DBSCAN** | Density-based clustering that groups points in dense regions and flags outliers as noise | `epsilon = 2.0`, `min_points = 10` |
| **Hierarchical (Agglomerative)** | Builds a nested cluster hierarchy via bottom-up merging | Average linkage, dendrogram cut at threshold `5.0` |

All three methods were run on the **full standardized feature set**, not on PCA-reduced data. PCA was used only afterward, to project results into two dimensions for visualization.

## Evaluation Metrics

- **Silhouette Score** — cluster compactness/separation (higher is better)
- **Rand Index** — agreement between predicted clusters and true team labels (higher is better)
- **Radius / Intercluster Distance Ratio** — compactness vs. separation (lower is better)
- **Outlier Count** — number of points DBSCAN labeled as noise (`-1`)

## Results

| Method | Silhouette | Rand Index | Radius Ratio | Outliers |
|---|---|---|---|---|
| **DBSCAN** | **0.479** | **0.848** | 1.117 | 42 |
| Hierarchical | 0.432 | 0.775 | **0.722** | N/A |
| K-Means | 0.384 | 0.789 | 2.016 | N/A |

**DBSCAN performed best overall**, achieving the highest silhouette score and Rand Index, and identifying 42 laps as outliers with unusual driving patterns. K-means, forced into exactly 10 clusters to match the number of teams, underperformed — suggesting the data doesn't separate into evenly-shaped, compact groups. Hierarchical clustering produced the most compact/separated clusters (lowest radius ratio) but matched team labels least closely.

PCA visualizations (52.5% variance explained across the first two components) showed that true team labels overlap substantially in 2D space rather than forming distinct groups — reinforcing that team identity alone does not fully explain lap-level variation.

<img width="1552" height="1131" alt="kmeans_pca" src="https://github.com/user-attachments/assets/f391e64c-ee0e-42df-a148-edcae614a608" />

<img width="1844" height="1131" alt="true_team_pca" src="https://github.com/user-attachments/assets/9fc9ac01-f6d4-4f0c-b85c-b079578030f1" />


Comparing the two plots side by side makes the core finding visually clear: the K-means clusters (Fig. 1) are fairly clean and well-separated, but they don't line up with how the true teams (Fig. 2) are distributed across the same PCA space — the team-colored plot shows heavy overlap rather than 10 distinct blobs.

## Discussion

The results show that F1 lap data **does contain meaningful structure**, but that structure only **partially** aligns with team identity. This suggests that while car/team characteristics matter, other factors — **driver style, car setup, track conditions, and lap-specific circumstances** — also meaningfully shape lap behavior. DBSCAN's density-based approach outperformed the others, likely because real lap clusters are not compact or evenly sized, which disadvantages K-means in particular.

## Conclusion & Future Work

Formula 1 lap data contains discoverable structure, but clusters do not map cleanly onto the 10 team labels — team identity is one factor among several shaping lap behavior. Future work could extend this analysis by incorporating:

- Race session data (not just qualifying)
- Multiple seasons
- Tire compound/degradation information
- Sector-level timing
- Track-specific features

## Tech Stack

- **Data collection & storage:** OpenF1 API, Parquet
- **Data wrangling & joins:** [DuckDB](https://duckdb.org/docs/current/)
- **Feature engineering & cleaning:** [pandas](https://pandas.pydata.org/docs/), [NumPy](https://numpy.org/doc/stable/)
- **Clustering & preprocessing:** [scikit-learn](https://scikit-learn.org/stable/) (K-means, DBSCAN, Agglomerative Clustering, StandardScaler, PCA)
- **Visualization:** [Matplotlib](https://matplotlib.org/stable/)

## References

1. OpenF1 API Documentation — https://openf1.org/docs/
2. pandas Documentation — https://pandas.pydata.org/docs/
3. NumPy Documentation — https://numpy.org/doc/stable/
4. DuckDB Documentation — https://duckdb.org/docs/current/
5. Matplotlib Documentation — https://matplotlib.org/stable/
6. scikit-learn Documentation — https://scikit-learn.org/stable/
7. Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise.* KDD-96.
8. Lloyd, S. (1982). *Least squares quantization in PCM.* IEEE Transactions on Information Theory, 28(2), 129–137.
9. Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.

---

*This project was completed for CSC 466 (Knowledge Discovery from Data) at California Polytechnic State University, San Luis Obispo.*
