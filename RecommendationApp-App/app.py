
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re
from typing import List
from fuzzywuzzy import process
import unidecode
import streamlit as st 
from streamlit_searchbox import st_searchbox
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from kalgo import KMeans
import matplotlib.pyplot as plt
import plotly.express as px



st.set_page_config(page_title="Player Scouting Recommendation System", page_icon="🏏", layout="wide")
st.markdown("<h1 style='text-align: center;'>Player Scouting Recommendation System</h1>", unsafe_allow_html=True)

if 'selected_player' not in st.session_state:
    st.session_state.selected_player = None

df_player = pd.read_csv("RecommendationApp-App/IPL_22_23_BBB.csv")

# Load additional CSV files
df_2024 = pd.read_csv("/RecommendationApp-App/playerperformance/player_rankings_2024.csv")
df_2024['Year'] = 2024
df_2023 = pd.read_csv("/RecommendationApp-App/playerperformance/player_rankings_2023.csv")
df_2023['Year'] = 2023
df_2022 = pd.read_csv("/RecommendationApp-App/playerperformance/player_rankings_2022.csv")
df_2022['Year'] = 2022

# Combine the dataframes
df_rankings = pd.concat([df_2024, df_2023, df_2022])

def remove_accents(text: str) -> str:
    return unidecode.unidecode(text)

def search_csv(searchterm: str) -> List[str]:
    if searchterm:
        normalized_searchterm = remove_accents(searchterm.lower())
        df_player['NormalizedStriker'] = df_player['striker_name'].apply(lambda x: remove_accents(x.lower()))
        df_player['NormalizedBowler'] = df_player['kadamba_bowler_name'].apply(lambda x: remove_accents(x.lower()))
        filtered_df = df_player[(df_player['NormalizedStriker'].str.contains(normalized_searchterm, case=False, na=False)) | (df_player['NormalizedBowler'].str.contains(normalized_searchterm, case=False, na=False))]
        suggestions = set(filtered_df['striker_name'].tolist() + filtered_df['non_striker_name'].tolist() + filtered_df['kadamba_bowler_name'].tolist())
        sorted_suggestions = sorted(suggestions, key=lambda x: process.extractOne(searchterm, [x])[1], reverse=True)  
        return sorted_suggestions
    else:
        return []  

selected_value = st_searchbox(
    search_csv,
    key="csv_searchbox",
    placeholder="🔍 Search a Cricket Player "
)




if selected_value:
    st.session_state.selected_player = selected_value

if st.session_state.selected_player:
    striker_df = df_player[df_player['striker_name'] == st.session_state.selected_player]
    bowler_df = df_player[df_player['kadamba_bowler_name'] == st.session_state.selected_player]



######################


    phase_order=['Powerplay','Middle','Death']


    # Display player profile based on role
    if not striker_df.empty and not bowler_df.empty:  # All-rounder
        # Display overall strike rate and game phases as striker
        striker_strike_rate_df = striker_df.groupby('striker_name').agg({'runs': 'sum', 'striker_name': 'count'})
        striker_strike_rate_df.rename(columns={'runs': 'total_runs', 'striker_name': 'balls_faced'}, inplace=True)
        striker_strike_rate_df['strike_rate'] = (striker_strike_rate_df['total_runs'] / striker_strike_rate_df['balls_faced']) * 100

        selected_player_strike_rate = striker_strike_rate_df.loc[selected_value, 'strike_rate'] if selected_value in striker_strike_rate_df.index else None

        striker_game_phases = striker_df['game_phase_cl'].unique()
        striker_phase_strike_rates = {}
        for phase in striker_game_phases:
            phase_df = striker_df[striker_df['game_phase_cl'] == phase]
            phase_runs = phase_df['runs'].sum()
            phase_balls = phase_df.shape[0]
            phase_strike_rate = (phase_runs / phase_balls) * 100
            striker_phase_strike_rates[phase] = round(phase_strike_rate, 2)  # Round to two decimal places

        # Display overall economy rate and game phases as bowler
        bowler_runs_df = bowler_df.groupby('kadamba_bowler_name')['runs'].sum()
        bowler_balls_df = bowler_df.groupby('kadamba_bowler_name').size()
        bowler_economy_rate_df = (bowler_runs_df / bowler_balls_df) * 6

        selected_player_economy_rate = bowler_economy_rate_df.get(selected_value, None)

        bowler_game_phases = bowler_df['game_phase_cl'].unique()
        bowler_phase_economy_rates = {}
        for phase in bowler_game_phases:
            phase_df = bowler_df[bowler_df['game_phase_cl'] == phase]
            phase_runs = phase_df['runs'].sum()
            phase_balls = phase_df.shape[0]
            phase_economy_rate = (phase_runs / phase_balls) * 6
            bowler_phase_economy_rates[phase] = round(phase_economy_rate, 2)  # Round to two decimal places

        # Display player name, player role, overall strike rate, and overall economy rate
        st.subheader(f"Player Profile: {st.session_state.selected_player}")
        st.write(f"**Player Role:** All-Rounder")
        if selected_player_strike_rate is not None:
            st.write(f"**Overall Strike Rate:** {selected_player_strike_rate:.2f}")
        if selected_player_economy_rate is not None:
            st.write(f"**Overall Economy Rate:** {selected_player_economy_rate:.2f}")

  

        # Display game phases as striker and bowler using columns
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Game Phases as Striker:")
            striker_phase_data = {'Game Phase': [], 'Strike Rate': []}
            for phase, strike_rate in striker_phase_strike_rates.items():
                striker_phase_data['Game Phase'].append(phase)
                striker_phase_data['Strike Rate'].append(strike_rate)
            striker_phase_df = pd.DataFrame(striker_phase_data)
            striker_phase_df['Game Phase'] = pd.Categorical(striker_phase_df['Game Phase'], categories=phase_order, ordered=True)
            striker_phase_df = striker_phase_df.sort_values('Game Phase')
            st.write(striker_phase_df)
        with col2:
            st.subheader("Game Phases as Bowler:")
            bowler_phase_data = {'Game Phase': [], 'Economy Rate': []}
            for phase, economy_rate in bowler_phase_economy_rates.items():
                bowler_phase_data['Game Phase'].append(phase)
                bowler_phase_data['Economy Rate'].append(economy_rate)
            bowler_phase_df = pd.DataFrame(bowler_phase_data)
            bowler_phase_df['Game Phase'] = pd.Categorical(bowler_phase_df['Game Phase'], categories=phase_order, ordered=True)
            bowler_phase_df = bowler_phase_df.sort_values('Game Phase')
            st.write(bowler_phase_df)

        
        # Display player performance
        player_rankings_df = df_rankings[df_rankings['Player'] == st.session_state.selected_player]

        # Remove the 'Player' column
        player_rankings_df = player_rankings_df.drop(columns=['Player'])

        # Convert the "Year" column to strings without commas
        player_rankings_df['Year'] = player_rankings_df['Year'].astype(str).str.replace(',', '')

        st.subheader("Player Performance:")
        st.dataframe(player_rankings_df.set_index('Year'))

    elif not striker_df.empty:  # Batter only
        # Display overall strike rate and game phases as striker
        striker_strike_rate_df = striker_df.groupby('striker_name').agg({'runs': 'sum', 'striker_name': 'count'})
        striker_strike_rate_df.rename(columns={'runs': 'total_runs', 'striker_name': 'balls_faced'}, inplace=True)
        striker_strike_rate_df['strike_rate'] = (striker_strike_rate_df['total_runs'] / striker_strike_rate_df['balls_faced']) * 100

        selected_player_strike_rate = striker_strike_rate_df.loc[selected_value, 'strike_rate'] if selected_value in striker_strike_rate_df.index else None

        striker_game_phases = striker_df['game_phase_cl'].unique()
        striker_phase_strike_rates = {}
        for phase in striker_game_phases:
            phase_df = striker_df[striker_df['game_phase_cl'] == phase]
            phase_runs = phase_df['runs'].sum()
            phase_balls = phase_df.shape[0]
            phase_strike_rate = (phase_runs / phase_balls) * 100
            striker_phase_strike_rates[phase] = round(phase_strike_rate, 2)  # Round to two decimal places

        # Display player name, player role, and overall strike rate
        st.subheader(f"Player Profile: {st.session_state.selected_player}")
        st.write(f"**Player Role:** Batter")
        if selected_player_strike_rate is not None:
            st.write(f"**Overall Strike Rate:** {selected_player_strike_rate:.2f}")

        # Display game phases as striker
        st.subheader("Game Phases as Striker:")
        striker_phase_data = {'Game Phase': [], 'Strike Rate': []}
        for phase, strike_rate in striker_phase_strike_rates.items():
            striker_phase_data['Game Phase'].append(phase)
            striker_phase_data['Strike Rate'].append(strike_rate)
        striker_phase_df = pd.DataFrame(striker_phase_data)
        striker_phase_df['Game Phase'] = pd.Categorical(striker_phase_df['Game Phase'], categories=phase_order, ordered=True)
        striker_phase_df = striker_phase_df.sort_values('Game Phase')
        st.write(striker_phase_df)

        # Display player performance
        player_rankings_df = df_rankings[df_rankings['Player'] == st.session_state.selected_player]

        # Remove the 'Player' column
        player_rankings_df = player_rankings_df.drop(columns=['Player'])

        # Convert the "Year" column to strings without commas
        player_rankings_df['Year'] = player_rankings_df['Year'].astype(str).str.replace(',', '')

        st.subheader("Player Performance:")
        st.dataframe(player_rankings_df.set_index('Year'))

    elif not bowler_df.empty:  # Bowler only
        # Display overall economy rate and game phases as bowler
        bowler_runs_df = bowler_df.groupby('kadamba_bowler_name')['runs'].sum()
        bowler_balls_df = bowler_df.groupby('kadamba_bowler_name').size()
        bowler_economy_rate_df = (bowler_runs_df / bowler_balls_df) * 6

        selected_player_economy_rate = bowler_economy_rate_df.get(selected_value, None)

        bowler_game_phases = bowler_df['game_phase_cl'].unique()
        bowler_phase_economy_rates = {}
        for phase in bowler_game_phases:
            phase_df = bowler_df[bowler_df['game_phase_cl'] == phase]
            phase_runs = phase_df['runs'].sum()
            phase_balls = phase_df.shape[0]
            phase_economy_rate = (phase_runs / phase_balls) * 6
            bowler_phase_economy_rates[phase] = round(phase_economy_rate, 2)  # Round to two decimal places

        # Display player name, player role, and overall economy rate
        st.subheader(f"Player Profile: {st.session_state.selected_player}")
        st.write(f"**Player Role:** Bowler")
        if selected_player_economy_rate is not None:
            st.write(f"**Overall Economy Rate:** {selected_player_economy_rate:.2f}")

        

       # Display game phases as bowler
        st.subheader("Game Phases as Bowler:")
        bowler_phase_data = {'Game Phase': [], 'Economy Rate': []}
        for phase, economy_rate in bowler_phase_economy_rates.items():
            bowler_phase_data['Game Phase'].append(phase)
            bowler_phase_data['Economy Rate'].append(economy_rate)
        bowler_phase_df = pd.DataFrame(bowler_phase_data)
        bowler_phase_df['Game Phase'] = pd.Categorical(bowler_phase_df['Game Phase'], categories=phase_order, ordered=True)
        bowler_phase_df = bowler_phase_df.sort_values('Game Phase')
        st.write(bowler_phase_df)


#######################


        # Display player performance
        player_rankings_df = df_rankings[df_rankings['Player'] == st.session_state.selected_player]

        # Remove the 'Player' column
        player_rankings_df = player_rankings_df.drop(columns=['Player'])

        # Convert the "Year" column to strings without commas
        player_rankings_df['Year'] = player_rankings_df['Year'].astype(str).str.replace(',', '')

        st.subheader("Player Performance:")
        st.dataframe(player_rankings_df.set_index('Year'))

    else:
        st.write("Player not found or does not have data available.")

# Extract unique player names from df_player dataframe
unique_players = pd.concat([df_player['striker_name'], df_player['kadamba_bowler_name']]).unique()

# Calculate overall strike rate and overall economy rate for each player
player_stats = {'Striker name': [], 'Bowler name': [], 'Overall Strike Rate': [], 'Overall Economy Rate': []}
for player in unique_players:
    striker_df = df_player[df_player['striker_name'] == player]
    bowler_df = df_player[df_player['kadamba_bowler_name'] == player]

    total_runs_striker = striker_df['runs'].sum()
    total_balls_striker = striker_df.shape[0]
    overall_strike_rate = (total_runs_striker / total_balls_striker) * 100 if total_balls_striker > 0 else None

    total_runs_bowler = bowler_df['runs'].sum()
    total_balls_bowler = bowler_df.shape[0]
    overall_economy_rate = (total_runs_bowler / total_balls_bowler) * 6 if total_balls_bowler > 0 else None

    player_stats['Striker name'].append(player)
    player_stats['Bowler name'].append(player)
    player_stats['Overall Strike Rate'].append(overall_strike_rate)
    player_stats['Overall Economy Rate'].append(overall_economy_rate)



# Create a dataframe
overall_stats_df = pd.DataFrame(player_stats)

# Create bowler_stat_df
bowler_stat_df = overall_stats_df.dropna(subset=['Overall Economy Rate']).rename(columns={'Bowler name': 'Bowler name', 'Overall Economy Rate': 'Economy Rate'})
bowler_stat_df = bowler_stat_df[['Bowler name', 'Economy Rate']]

# Create batter_stat_df
batter_stat_df = overall_stats_df.dropna(subset=['Overall Strike Rate']).rename(columns={'Striker name': 'Batter name', 'Overall Strike Rate': 'Strike Rate'})
batter_stat_df = batter_stat_df[['Batter name', 'Strike Rate']]

# Create allrounder_stat_df
allrounder_stat_df = overall_stats_df.dropna(subset=['Overall Strike Rate', 'Overall Economy Rate']).rename(columns={'Striker name': 'All-Rounder name', 'Overall Strike Rate': 'Strike Rate', 'Overall Economy Rate': 'Economy Rate'})
allrounder_stat_df = allrounder_stat_df[['All-Rounder name', 'Strike Rate', 'Economy Rate']]




# Pivot the df_rankings DataFrame for RAA
RAA_df = df_rankings.pivot(index='Player', columns='Year', values='RAA').reset_index()
RAA_df.columns = ['Player', '2022', '2023', '2024']
RAA_df = RAA_df.fillna(0)

# Pivot the df_rankings DataFrame for EFscore
Ef_df = df_rankings.pivot(index='Player', columns='Year', values='EFscore').reset_index()
Ef_df.columns = ['Player', '2022', '2023', '2024']
Ef_df = Ef_df.fillna(0)


#st.dataframe(RAA_df)

# Check if a player is selected
if st.session_state.selected_player:
    selected_player = st.session_state.selected_player

    if selected_player in RAA_df['Player'].values:
        # Filter RAA_df for the selected player
        selected_player_raa_df = RAA_df[RAA_df['Player'] == selected_player]

        # Create the line graph for the selected player's RAA
        fig_raa = go.Figure()
        fig_raa.add_trace(go.Scatter(x=selected_player_raa_df.columns[1:], y=selected_player_raa_df.values[0][1:], mode='lines+markers', name=selected_player))
        fig_raa.update_layout(title=f'RAA Values Over Years ', xaxis_title='Year', yaxis_title='Run Above Average')

    if selected_player in Ef_df['Player'].values:
        # Filter Ef_df for the selected player
        selected_player_ef_df = Ef_df[Ef_df['Player'] == selected_player]

        # Create the line graph for the selected player's EFscore
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(x=selected_player_ef_df.columns[1:], y=selected_player_ef_df.values[0][1:], mode='lines+markers', name=selected_player))
        fig_ef.update_layout(title=f'EF Score Values Over Years ', xaxis_title='Year', yaxis_title='EF Score')

    if selected_player in RAA_df['Player'].values:
    # Display the plots side by side
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_raa)
        with col2:
            st.plotly_chart(fig_ef)




def perform_clustering(df, role):
    X = df.iloc[:, 1:].values  # Assuming the first column is the player name

    km = KMeans(n_clusters=15, max_iter=100)
    cluster_labels = km.fit_predict(X)

    df['Cluster'] = cluster_labels
    return df


bowler_stat_df_clustered = perform_clustering(bowler_stat_df, 'Bowler')
batter_stat_df_clustered = perform_clustering(batter_stat_df, 'Batter')
allrounder_stat_df_clustered = perform_clustering(allrounder_stat_df, 'All-Rounder')

def display_similar_players(selected_player, df, role):
    similar_players_df = df[df['Cluster'] == df[df[f'{role} name'] == selected_player]['Cluster'].values[0]]

    st.subheader(f"Similar {role}s:")
    if similar_players_df.empty:
        st.write("No similar players found.")
    else:
        if role == 'Bowler':
            similar_players_df = similar_players_df[['Bowler name', 'Economy Rate']]
            st.dataframe(similar_players_df)
        elif role == 'Batter':
            similar_players_df = similar_players_df[['Batter name', 'Strike Rate']]
            st.dataframe(similar_players_df)
        elif role == 'All-Rounder':
            similar_players_df = similar_players_df[['All-Rounder name', 'Strike Rate', 'Economy Rate']]
            st.dataframe(similar_players_df)



def get_similar_players(selected_player, df, role):
    similar_players_df = df[df['Cluster'] == df[df[f'{role} name'] == selected_player]['Cluster'].values[0]]
    return similar_players_df


def plot_line_graph(similar_players_df, role):
    if role == 'Bowler':
        title = ""
        x_label = "Player"
        y_label = "Economy Rate"
        x_data = similar_players_df['Bowler name']
        y_data = similar_players_df['Economy Rate']
    elif role == 'Batter':
        title = ""
        x_label = "Player"
        y_label = "Strike Rate"
        x_data = similar_players_df['Batter name']
        y_data = similar_players_df['Strike Rate']

    fig = px.line(x=x_data, y=y_data, title=title, labels={x_data.name: x_label, y_data.name: y_label})
    fig.update_yaxes(title_text=y_label)  # Update y-axis label
    st.plotly_chart(fig)

    
# Display similar players based on selected player's role
if st.session_state.selected_player:
    if not bowler_stat_df_clustered.empty and st.session_state.selected_player in bowler_stat_df_clustered['Bowler name'].values:
        similar_bowlers_df = get_similar_players(st.session_state.selected_player, bowler_stat_df_clustered, 'Bowler')
        similar_bowlers_df=similar_bowlers_df.sort_values(by='Economy Rate', ascending=False)
        similar_bowlers_df = similar_bowlers_df[similar_bowlers_df['Bowler name'] != st.session_state.selected_player]
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Bowlers : Economy Rate")
            st.dataframe(similar_bowlers_df.drop(columns=['Cluster']))
        with col2:
            st.subheader("Bowlers : Economy Rate Graph:")
            plot_line_graph(similar_bowlers_df, 'Bowler')
    if not batter_stat_df_clustered.empty and st.session_state.selected_player in batter_stat_df_clustered['Batter name'].values:
        similar_batters_df = get_similar_players(st.session_state.selected_player, batter_stat_df_clustered, 'Batter')
        similar_batters_df=similar_batters_df.sort_values(by='Strike Rate', ascending=False)
        similar_batters_df = similar_batters_df[similar_batters_df['Batter name'] != st.session_state.selected_player]
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Batters : Strike Rate")
            st.dataframe(similar_batters_df.drop(columns=['Cluster']))
        with col4:
            st.subheader("Batters : Strike Rate Graph")
            plot_line_graph(similar_batters_df, 'Batter')
    

def plot_all_rounders_line_graph(similar_players_df):
    fig = px.line()

    # Add Strike Rate traces
    fig.add_scatter(x=similar_players_df['All-Rounder name'], y=similar_players_df['Strike Rate'], mode='lines+markers', name='Strike Rate')

    # Add Economy Rate traces
    fig.add_scatter(x=similar_players_df['All-Rounder name'], y=similar_players_df['Economy Rate'], mode='lines+markers', name='Economy Rate', yaxis='y2')

    # Create axis titles
    fig.update_layout(
        title='',
        xaxis=dict(title='Player'),
        yaxis=dict(title='Strike Rate'),
        yaxis2=dict(title='Economy Rate', overlaying='y', side='right')
    )

    st.plotly_chart(fig)

# Display similar all-rounders table and graph
if not allrounder_stat_df_clustered.empty and st.session_state.selected_player in allrounder_stat_df_clustered['All-Rounder name'].values:
    similar_all_rounders_df = get_similar_players(st.session_state.selected_player, allrounder_stat_df_clustered, 'All-Rounder')
    similar_all_rounders_df =similar_all_rounders_df.sort_values(by=['Strike Rate', 'Economy Rate'], ascending=False)
    similar_all_rounders_df = similar_all_rounders_df[similar_all_rounders_df['All-Rounder name'] != st.session_state.selected_player]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("All-Rounders : Strike Rate & Economy Rate")
        st.dataframe(similar_all_rounders_df.drop(columns=['Cluster']))
    with col2:
        st.subheader("All-Rounders : Strike Rate & Economy Rate Graph")
        plot_all_rounders_line_graph(similar_all_rounders_df)

# Load the CSV file
player_rankings_2024 = pd.read_csv("/RecommendationApp-App/playerperformance/player_rankings_2024.csv")

# Function to extract additional information for common names
def extract_additional_info(df, role):
    common_names = df[df[f'{role} name'].isin(player_rankings_2024['Player'])][f'{role} name']
    additional_info = player_rankings_2024[player_rankings_2024['Player'].isin(common_names)][['Player', 'RAA', 'Wins', 'EFscore']]
    return additional_info

# Extract additional information for bowlers
bowler_additional_info = extract_additional_info(bowler_stat_df, 'Bowler')
bowler_additional_info.rename(columns={'Player': 'Bowler name'}, inplace=True)
# Merge with bowler_stat_df to include economy rate
bowler_stat_df_merged = pd.merge(bowler_stat_df, bowler_additional_info, how='inner', left_on='Bowler name', right_on='Bowler name')

# Extract additional information for batters
batter_additional_info = extract_additional_info(batter_stat_df, 'Batter')
batter_additional_info.rename(columns={'Player': 'Batter name'}, inplace=True)
# Merge with batter_stat_df to include strike rate
batter_stat_df_merged = pd.merge(batter_stat_df, batter_additional_info, how='inner', left_on='Batter name', right_on='Batter name')

# Extract additional information for all-rounders
allrounder_additional_info = extract_additional_info(allrounder_stat_df, 'All-Rounder')
allrounder_additional_info.rename(columns={'Player': 'All-Rounder name'}, inplace=True)
# Merge with allrounder_stat_df to include strike rate and economy rate
allrounder_stat_df_merged = pd.merge(allrounder_stat_df, allrounder_additional_info, how='inner', left_on='All-Rounder name', right_on='All-Rounder name')



bowler_stat_df_merged_clustered = perform_clustering(bowler_stat_df_merged, 'Bowler')
batter_stat_df_merged_clustered = perform_clustering(batter_stat_df_merged, 'Batter')
allrounder_stat_df_merged_clustered = perform_clustering(allrounder_stat_df_merged, 'All-Rounder')



############################# Radar Analytics ##########################################


def bar_analytics_plot(similar_players_df, player_role):
    # Define categories and player column based on role
    if player_role == 'Bowler':
        categories = ['RAA', 'Wins', 'EFscore', 'Economy Rate']
        player_column = 'Bowler name'
    elif player_role == 'Batter':
        categories = ['RAA', 'Wins', 'EFscore', 'Strike Rate']
        player_column = 'Batter name'
    elif player_role == 'All-Rounder':
        categories = ['RAA', 'Wins', 'EFscore', 'Strike Rate', 'Economy Rate']
        player_column = 'All-Rounder name'
    else:
        raise ValueError(f"")

    # Creating grouped bar chart for each player in similar_players_df
    fig = go.Figure()

    for category in categories:
        fig.add_trace(go.Bar(
            x=similar_players_df[player_column],
            y=similar_players_df[category],
            name=category
        ))

    fig.update_layout(
        barmode='group',
        xaxis_title=player_column,
        yaxis_title='Values',
        title=f'',
        showlegend=True
    )

    return fig

# Check if a player is selected
if st.session_state.selected_player:
    if not bowler_stat_df_merged_clustered.empty and st.session_state.selected_player in bowler_stat_df_merged_clustered['Bowler name'].values:
        similar_bowlers_df = get_similar_players(st.session_state.selected_player, bowler_stat_df_merged_clustered, 'Bowler')
        similar_bowlers_df = similar_bowlers_df[similar_bowlers_df['Bowler name'] != st.session_state.selected_player]
        similar_bowlers_df_sorted = similar_bowlers_df.sort_values(by=['Economy Rate', 'RAA', 'Wins', 'EFscore'], ascending=[True, False, False, False])
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Bowlers :  Economy Rate , RAA , Wins , EFscore")
            st.dataframe(similar_bowlers_df_sorted.drop(columns=['Cluster']))
        with col2:
            st.subheader(" Bowlers : Analytics")
            fig = bar_analytics_plot(similar_bowlers_df_sorted, 'Bowler')
            st.plotly_chart(fig, use_container_width=True)
    if not batter_stat_df_merged_clustered.empty and st.session_state.selected_player in batter_stat_df_merged_clustered['Batter name'].values:
        similar_batters_df = get_similar_players(st.session_state.selected_player, batter_stat_df_merged_clustered, 'Batter')
        similar_batters_df = similar_batters_df[similar_batters_df['Batter name'] != st.session_state.selected_player]
        similar_batters_df_sorted = similar_batters_df.sort_values(by=['Strike Rate', 'RAA', 'Wins', 'EFscore'], ascending=False)
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Batters :  Strike Rate , RAA , Wins , EFscore")
            st.dataframe(similar_batters_df_sorted.drop(columns=['Cluster']))
        with col4:
            st.subheader(" Batters: Analytics")
            fig = bar_analytics_plot(similar_batters_df_sorted, 'Batter')
            st.plotly_chart(fig, use_container_width=True)
    if not allrounder_stat_df_merged_clustered.empty and st.session_state.selected_player in allrounder_stat_df_merged_clustered['All-Rounder name'].values:
        similar_all_rounders_df = get_similar_players(st.session_state.selected_player, allrounder_stat_df_merged_clustered, 'All-Rounder')
        similar_all_rounders_df = similar_all_rounders_df[similar_all_rounders_df['All-Rounder name'] != st.session_state.selected_player]
        similar_all_rounders_df_sorted = similar_all_rounders_df.sort_values(by=['Strike Rate', 'Economy Rate', 'RAA', 'Wins', 'EFscore'], ascending=[False, True, False, False, False])
        col5, col6 = st.columns(2)
        with col5:
            st.subheader("All-Rounders : Economy Rate, Strike Rate , RAA , Wins , EFscore")
            st.dataframe(similar_all_rounders_df_sorted.drop(columns=['Cluster']))
        with col6:
            st.subheader(" All-Rounders: Analytics")
            fig = bar_analytics_plot(similar_all_rounders_df_sorted, 'All-Rounder')
            st.plotly_chart(fig, use_container_width=True)

################



def player_performance_over_years(df_rankings, selected_player):
    # Filter the dataframe for the selected player
    player_data = df_rankings[df_rankings['Player'] == selected_player]
    
    # Ensure the data is sorted by year
    player_data = player_data.sort_values('Year')
    
    # Extract years, RAA, Wins, and EFscore
    years = player_data['Year'].tolist()
    raa = player_data['RAA'].tolist()
    wins = player_data['Wins'].tolist()
    efscore = player_data['EFscore'].tolist()
    
    
    
    # Create the line graph
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=years, y=raa, mode='lines+markers', name='RAA'))
    fig.add_trace(go.Scatter(x=years, y=wins, mode='lines+markers', name='Wins'))
    fig.add_trace(go.Scatter(x=years, y=efscore, mode='lines+markers', name='EFscore'))

    fig.update_layout(
        title=f'',
        xaxis_title='Year',
        yaxis_title='Values',
        showlegend=True
    )

    st.plotly_chart(fig)

# Usage example for displaying the player's performance over years
if st.session_state.selected_player:
    st.subheader(f"{st.session_state.selected_player}'s Performance Over Years:")
    player_performance_over_years(df_rankings, st.session_state.selected_player)




#########################################################

# # Display the three dataframes
# st.subheader("Statistics:")
# col1, col2, col3 = st.columns(3)

# with col1:
#     st.subheader("Bowler Statistics:")
#     st.dataframe(bowler_stat_df.drop(columns=['Cluster']))  # Remove the 'Cluster' column for bowler statistics

# with col2:
#     st.subheader("Batter Statistics:")
#     st.dataframe(batter_stat_df.drop(columns=['Cluster']))

# with col3:
#     st.subheader("All-Rounder Statistics:")
#     st.dataframe(allrounder_stat_df.drop(columns=['Cluster']))
    

# # Display the merged dataframes
# st.subheader("Merged Statistics:")
# col1, col2, col3 = st.columns(3)

# with col1:
#     st.subheader("Bowler Statistics:")
#     st.dataframe(bowler_stat_df_merged.drop(columns=['Cluster']))  # Remove the 'Cluster' column for bowler statistics

# with col2:
#     st.subheader("Batter Statistics:")
#     st.dataframe(batter_stat_df_merged.drop(columns=['Cluster']))

# with col3:
#     st.subheader("All-Rounder Statistics:")
#     st.dataframe(allrounder_stat_df_merged.drop(columns=['Cluster']))


####################################################################



