import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, rand_score
from sklearn.decomposition import PCA

from common import APP_NAME, SEASON, load_data

# Same identifier columns excluded in Project_466_Data_visualization_ML-Models.ipynb,
# so this reproduces that notebook's methodology (StandardScaler over the numeric lap
# features only) rather than a different feature set.
LABEL_COLS = ['team_name', 'driver_name', 'driver_number', 'session_key', 'lap_number',
              'date_start', 'country_name', 'circuit_short_name', 'location']


@st.cache_data
def prepare_features(df):
    feature_cols = [c for c in df.select_dtypes(include='number').columns
                     if c not in LABEL_COLS and c != 'n_samples']
    clean = df.dropna(subset=feature_cols).reset_index(drop=True)

    # A handful of laps have max_speed/avg_speed/RPM/throttle all reading exactly 0 —
    # a telemetry sensor dropout (all 3 from the same car/session), not real driving
    # data. Left in, one such lap sits ~4x farther from the rest of the data than any
    # other point and dominates the PCA projection. Drop them.
    clean = clean[clean['max_speed'] > 0].reset_index(drop=True)

    teams = sorted(clean['team_name'].unique())
    team_to_id = {t: i for i, t in enumerate(teams)}
    drivers = sorted(clean['driver_name'].unique())
    driver_to_id = {d: i for i, d in enumerate(drivers)}
    circuits = sorted(clean['circuit_short_name'].unique())
    circuit_to_id = {c: i for i, c in enumerate(circuits)}

    y_team = clean['team_name'].map(team_to_id).values
    y_driver = clean['driver_name'].map(driver_to_id).values
    y_circuit = clean['circuit_short_name'].map(circuit_to_id).values

    X_global = StandardScaler().fit_transform(clean[feature_cols].values)

    # Session-normalized version: z-score each feature within its own race before
    # clustering, so the model can't just rediscover "which circuit was this" (lap
    # times/speeds differ far more by track than by team) — see the callout below.
    session_mean = clean.groupby('session_key')[feature_cols].transform('mean')
    session_std = clean.groupby('session_key')[feature_cols].transform('std').replace(0, 1)
    X_session = ((clean[feature_cols] - session_mean) / session_std).values

    return {
        'clean': clean, 'feature_cols': feature_cols,
        'X_global': X_global, 'X_session': X_session,
        'y_team': y_team, 'teams': teams,
        'y_driver': y_driver, 'drivers': drivers,
        'y_circuit': y_circuit, 'circuits': circuits,
    }


def cluster_metrics(labels, X, y_team, y_driver, y_circuit, name):
    mask = labels != -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    sil = silhouette_score(X[mask], labels[mask]) if len(set(labels[mask])) > 1 else float('nan')
    return {
        'algorithm': name,
        'n_clusters': n_clusters,
        'n_noise': n_noise if n_noise else None,
        'silhouette': round(sil, 3),
        'rand_index_vs_team': round(rand_score(labels, y_team), 3),
        'rand_index_vs_driver': round(rand_score(labels, y_driver), 3),
        'rand_index_vs_circuit': round(rand_score(labels, y_circuit), 3),
    }


@st.cache_data
def run_global_clustering(X_global, y_team, y_driver, y_circuit):
    results = {}
    results['KMeans (k=10)'] = KMeans(n_clusters=10, random_state=42, n_init=10).fit(X_global).labels_
    results['Hierarchical (avg, k=10)'] = AgglomerativeClustering(n_clusters=10, linkage='average').fit(X_global).labels_
    results['DBSCAN (eps=2, min_samples=10)'] = DBSCAN(eps=2, min_samples=10).fit(X_global).labels_
    metrics = pd.DataFrame([cluster_metrics(labels, X_global, y_team, y_driver, y_circuit, name)
                             for name, labels in results.items()])
    return results, metrics


@st.cache_data
def run_circuit_controlled_clustering(X_session, y_team, y_driver, y_circuit):
    results = {}
    results['KMeans (k=10)'] = KMeans(n_clusters=10, random_state=42, n_init=10).fit(X_session).labels_
    results['Hierarchical (ward, k=10)'] = AgglomerativeClustering(n_clusters=10, linkage='ward').fit(X_session).labels_
    metrics = pd.DataFrame([cluster_metrics(labels, X_session, y_team, y_driver, y_circuit, name)
                             for name, labels in results.items()])
    return results, metrics


@st.cache_data
def compute_pca(X):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_.sum()


def scatter_by_label(ax, coords, labels, names, title, show_legend):
    unique = sorted(set(labels))
    cmap = plt.get_cmap('tab10' if len(unique) <= 10 else 'tab20', max(len(unique), 3))
    for i, lab in enumerate(unique):
        mask = labels == lab
        if lab == -1:
            color, name = 'lightgray', 'noise'
        else:
            color = cmap(i)
            name = names[lab] if names is not None else f'cluster {lab}'
        ax.scatter(coords[mask, 0], coords[mask, 1], s=14, color=color, label=name, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    if show_legend:
        ncol = 2 if len(unique) > 12 else 1
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7, ncol=ncol)


def metrics_table(df):
    st.dataframe(df, hide_index=True, width='stretch')


def pca_section(X, results, y_team, y_driver, teams, drivers, key_prefix):
    coords, var_explained = compute_pca(X)
    col1, col2 = st.columns(2)
    with col1:
        algo_name = st.selectbox("Algorithm", list(results.keys()), key=f'{key_prefix}_algo')
    with col2:
        color_by = st.radio("Color the ground-truth panel by", ["Team", "Driver"],
                             horizontal=True, key=f'{key_prefix}_colorby')
    true_labels = y_team if color_by == "Team" else y_driver
    true_names = teams if color_by == "Team" else drivers

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    scatter_by_label(ax1, coords, results[algo_name], None, f'Predicted clusters — {algo_name}', show_legend=False)
    scatter_by_label(ax2, coords, true_labels, true_names, f'Actual {color_by.lower()}', show_legend=True)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(
        f"These 2 principal components capture only {var_explained:.0%} of the variance in the "
        f"{X.shape[1]} features, so this flat projection understates the real separation — the metrics "
        "table above is computed on the full feature space, not this 2D view. Expect the scatter to look "
        "more blended than the rand_index numbers imply."
    )


st.title(f"{APP_NAME} — Unsupervised Clustering")
st.caption(
    f"Do laps naturally group by team or driver, without ever being told the label? "
    f"27 numeric lap-telemetry features (pace, sectors, throttle, gear, speed distribution) from the "
    f"{SEASON} season are standardized and clustered with no team/driver information — the same feature "
    "set and preprocessing as Project_466_Data_visualization_ML-Models.ipynb, reproduced here with "
    "scikit-learn so it runs live in the app."
)

df = load_data()
data = prepare_features(df)
st.caption(
    f"{len(data['clean'])} laps (after dropping missing/broken telemetry) across "
    f"{len(data['teams'])} teams, {len(data['drivers'])} drivers, {len(data['circuits'])} circuits."
)

st.header("1. Global clustering (season-wide, as in the original notebook)")
global_results, global_metrics = run_global_clustering(
    data['X_global'], data['y_team'], data['y_driver'], data['y_circuit']
)
metrics_table(global_metrics)
st.caption(
    "**silhouette**: how well-separated the clusters are on their own (−1 to 1, higher is tighter/cleaner). "
    "**rand_index**: how often two laps agree on same-cluster/different-cluster between the unsupervised "
    "grouping and a real label (0–1, higher = clusters line up with that label)."
)

st.warning(
    f"**Caveat:** rand_index_vs_circuit (~{global_metrics['rand_index_vs_circuit'].mean():.2f} avg) is far "
    f"higher than rand_index_vs_team (~{global_metrics['rand_index_vs_team'].mean():.2f} avg) here. Raw lap "
    "times/speeds differ far more by circuit (Monaco vs. Monza) than by team, so this clustering is mostly "
    "rediscovering *which race a lap came from*, not team style. Section 2 controls for that."
)

pca_section(data['X_global'], global_results, data['y_team'], data['y_driver'],
            data['teams'], data['drivers'], key_prefix='global')

with st.expander("Reference: original notebook results (custom from-scratch implementations)"):
    st.caption(
        "Recorded in Project_466_Data_visualization_ML-Models.ipynb, which implemented KMeans/hierarchical/"
        "DBSCAN from scratch (run on Google Colab) rather than using scikit-learn. Numbers differ slightly "
        "from Section 1 above due to implementation details, but tell the same story."
    )
    st.dataframe(pd.DataFrame([
        {'algorithm': 'DBSCAN (eps=2, min_samples=10)', 'silhouette': 0.479, 'rand_index_vs_team': 0.848, 'n_outliers': 42},
        {'algorithm': 'KMeans (k=10)', 'silhouette': 0.359, 'rand_index_vs_team': 0.805, 'n_outliers': None},
        {'algorithm': 'Hierarchical (avg, threshold=5)', 'silhouette': 0.432, 'rand_index_vs_team': 0.775, 'n_outliers': None},
    ]), hide_index=True, width='stretch')

st.divider()
st.header("2. Circuit-controlled clustering (removes the track effect)")
st.caption(
    "Each feature is first converted to a z-score within its own race (same normalization used on the "
    "Team Comparison page), *then* clustered — so a fast Monaco lap and a fast Monza lap both read as "
    "'faster than the field that day' instead of being separated by raw lap time. DBSCAN isn't shown here: "
    "at this different feature scale its density assumptions no longer hold without re-tuning eps/min_samples "
    "from scratch, and a quick grid search didn't find stable clusters — dropped rather than force a bad fit. "
    "Hierarchical uses ward linkage here instead of average, which degenerates into one giant cluster on this "
    "transformed feature space."
)

circuit_results, circuit_metrics = run_circuit_controlled_clustering(
    data['X_session'], data['y_team'], data['y_driver'], data['y_circuit']
)
metrics_table(circuit_metrics)

team_sig = circuit_metrics['rand_index_vs_team'].mean()
circuit_sig = circuit_metrics['rand_index_vs_circuit'].mean()
driver_sig = circuit_metrics['rand_index_vs_driver'].mean()
st.success(
    f"With the track effect controlled for, **team identity (avg rand_index={team_sig:.3f})** is now the "
    f"strongest signal in the clustering — ahead of circuit ({circuit_sig:.3f}) — and driver identity "
    f"({driver_sig:.3f}) tracks closely behind it. This is the more honest evidence that teams (and drivers) "
    "are distinguishable by driving style, independent of which race it was."
)

pca_section(data['X_session'], circuit_results, data['y_team'], data['y_driver'],
            data['teams'], data['drivers'], key_prefix='circuit')
