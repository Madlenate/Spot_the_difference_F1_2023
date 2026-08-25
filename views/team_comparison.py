import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from common import APP_NAME, SEASON, load_data

# Features compared, and how each is read: positive = "better/more" than the
# session field average, after lap_duration is flipped (lower time = better).
FEATURE_COLS = ['lap_duration', 'max_speed', 'avg_speed', 'avg_throttle',
                 'pct_full_throttle', 'pct_braking', 'n_gear_changes', 'pct_drs_active']
FEATURE_LABELS = {
    'lap_duration': 'pace',
    'max_speed': 'top_speed',
    'n_gear_changes': 'gear_changes',
}

SCORECARD_AGG = dict(
    n_laps=('lap_duration', 'count'),
    best_lap=('lap_duration', 'min'),
    avg_lap=('lap_duration', 'mean'),
    lap_consistency=('lap_duration', 'std'),
    avg_sector_1=('duration_sector_1', 'mean'),
    avg_sector_2=('duration_sector_2', 'mean'),
    avg_sector_3=('duration_sector_3', 'mean'),
    top_speed=('max_speed', 'max'),
    avg_speed=('avg_speed', 'mean'),
    speed_trap=('st_speed', 'max'),
    avg_throttle=('avg_throttle', 'mean'),
    pct_full_throttle=('pct_full_throttle', 'mean'),
    pct_braking=('pct_braking', 'mean'),
    avg_gear_changes=('n_gear_changes', 'mean'),
    pct_drs_active=('pct_drs_active', 'mean'),
)


@st.cache_data
def compute_team_profile(df):
    """Each feature as a z-score within its own race, averaged across the season per team."""
    session_mean = df.groupby('session_key')[FEATURE_COLS].transform('mean')
    session_std = df.groupby('session_key')[FEATURE_COLS].transform('std')
    z = (df[FEATURE_COLS] - session_mean) / session_std
    z['lap_duration'] = -z['lap_duration']
    z['team_name'] = df['team_name']
    profile = z.groupby('team_name')[FEATURE_COLS].mean().round(2)
    return profile.rename(columns=FEATURE_LABELS)


@st.cache_data
def compute_scorecard(df, group_cols):
    sc = df.groupby(list(group_cols)).agg(**SCORECARD_AGG).reset_index()
    sc['lap_consistency'] = sc['lap_consistency'].fillna(0)
    round_cols = sc.select_dtypes('float').columns
    sc[round_cols] = sc[round_cols].round(2)
    return sc


def vertical_scorecard(row, value_label):
    """One metric per row instead of one column per metric, so team and driver
    cards line up on the same metric rows and are easy to scan/compare."""
    return row.to_frame(name=value_label).rename_axis('metric')


def plot_diff_bar(team_profile, team_a, team_b):
    diff = (team_profile.loc[team_a] - team_profile.loc[team_b]).sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ['#1E5BFF' if v > 0 else '#FF3B30' for v in diff]
    ax.barh(diff.index, diff.values, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel(f'{team_a} minus {team_b}  (z-score diff, vs field average)')
    ax.set_title('Where they differ')
    fig.tight_layout()
    return fig


def plot_radar(team_profile, team_a, team_b):
    categories = list(team_profile.columns)
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    shift = 2  # keeps every radius >= 0 so matplotlib's polar axes don't distort negative z-scores

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for team, color in [(team_a, '#1E5BFF'), (team_b, '#FF3B30')]:
        values = (team_profile.loc[team, categories] + shift).tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=team)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.plot(angles, [shift] * len(angles), color='gray', linestyle='--', linewidth=1, label='field average')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticklabels([])
    ax.set_title('Profile shape', y=1.12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15))
    fig.tight_layout()
    return fig


@st.cache_data
def race_options(df):
    """Session key -> circuit label, ordered by date so races read as a calendar."""
    races = df.groupby(['session_key', 'circuit_short_name'])['date_start'].min().reset_index()
    races = races.sort_values('date_start')
    return list(zip(races['circuit_short_name'], races['session_key']))


df = load_data()

st.title(f"{APP_NAME} - {SEASON} Season")
st.caption(
    f"Lap telemetry from the {SEASON} F1 season. Compare two teams' driving style and pace, "
    "normalized against each race's field average."
)

scope = st.segmented_control(
    "Scope", ["Overall (Season)", "By Race"], default="Overall (Season)", required=True
)

if scope == "By Race":
    races = race_options(df)
    race_labels = [label for label, _ in races]
    race_label = st.selectbox("Race", race_labels)
    session_key = dict(races)[race_label]
    scoped_df = df[df['session_key'] == session_key]
    scope_caption = race_label
else:
    scoped_df = df
    scope_caption = f"full {SEASON} season"

team_profile = compute_team_profile(scoped_df)
team_scorecard = compute_scorecard(scoped_df, ['team_name'])
driver_scorecard = compute_scorecard(scoped_df, ['driver_name', 'team_name'])

teams = sorted(df['team_name'].unique())
default_a = teams.index('Red Bull Racing') if 'Red Bull Racing' in teams else 0
default_b = teams.index('Mercedes') if 'Mercedes' in teams else min(1, len(teams) - 1)

col_a, col_b = st.columns(2)
with col_a:
    team_a = st.selectbox("Team A", teams, index=default_a)
with col_b:
    team_b = st.selectbox("Team B", teams, index=default_b)

if team_a == team_b:
    st.warning("Pick two different teams to compare.")
    st.stop()

missing = [t for t in (team_a, team_b) if t not in team_profile.index]
if missing:
    st.warning(f"No lap data for {', '.join(missing)} in {scope_caption}.")
    st.stop()

st.header(f"{team_a} vs {team_b}")
st.caption(scope_caption)

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.pyplot(plot_diff_bar(team_profile, team_a, team_b))
with chart_col2:
    st.pyplot(plot_radar(team_profile, team_a, team_b))

st.divider()
st.header("Team & Driver Scorecards")

score_col1, score_col2 = st.columns(2)
for col, team in [(score_col1, team_a), (score_col2, team_b)]:
    with col:
        st.subheader(team)
        t_row = team_scorecard[team_scorecard['team_name'] == team].drop(columns='team_name').iloc[0]
        st.dataframe(vertical_scorecard(t_row, 'team'), width='stretch')

        st.markdown(f"**{team} drivers**")
        d_rows = (
            driver_scorecard[driver_scorecard['team_name'] == team]
            .drop(columns='team_name')
            .sort_values('avg_lap')
        )
        for _, drow in d_rows.iterrows():
            st.markdown(f"*{drow['driver_name']}*")
            driver_stats = drow.drop('driver_name')
            st.dataframe(vertical_scorecard(driver_stats, 'value'), width='stretch')
