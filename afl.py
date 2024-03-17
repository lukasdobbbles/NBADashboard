import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
import numpy as np
import gspread
import pytz
import gc
from datetime import datetime, time, timedelta
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="AFL Dashboard", layout="wide")

# Authenticate with Google Sheets API
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
gc = gspread.authorize(credentials)

@st.cache_data(ttl=86400, hash_funcs={pd.DataFrame: lambda x: None})
def load_data():
    worksheet = gc.open("afl stats").sheet1  # Use the appropriate worksheet
    data = worksheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])
    df["Round"] = pd.to_numeric(df["Round"])
    return df

# Load initial data from Google Sheets using the cached function
df = load_data()

# Define unique seasons
unique_seasons = ['2023', '2024']

allowed_columns = ["Disposals", "Fantasy","CBA Pct","KI","Kicks","Handballs", "Marks", "Goals", "Behinds", "Shots","Hitouts", "Tackles", "CP", "UP","Clearances", "FF", "FA", "TOG" ]

tab_names = ["Player Dashboard", "Hit Rates", "Trending (coming soon)","DVP","Team Dashboard","CBAs","Kick Ins","Team Stats"]
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(tab_names)
######################################FILTERS######################################

with tab1:
    # Multi-select filter for SEASON_YEAR with valid default values
    st.sidebar.markdown("## Dashboard Filters")

    # Get unique seasons from the DataFrame
    unique_seasons = df["Season"].unique()

    selected_years = st.sidebar.multiselect(
        "Season",
        options=unique_seasons,
        default=['2024'],  # Set the default value to the first season
        key="season_year_multiselect"
    )

    # Ensure at least one season is selected
    if not selected_years:
        selected_years = ['2024']  # Set the default value to the first season

    col1, col2, col3 = st.columns(3)

    # Player Name selection
    with col1:
        name = st.selectbox(
            "Select Player",
            options=df["Player"].unique(),
        )

    # Filter players from the same team
    team_players = df[df["Team"] == df[df["Player"] == name]["Team"].values[0]]["Player"].unique()

    # "Without" Filter (Default: No Players Selected)
    without_names = st.sidebar.multiselect(
        "Without Player/s",
        options=team_players,
        default=[],
    )

    # "With" Filter (Default: No Players Selected)
    with_names = st.sidebar.multiselect(
        "With Player/s",
        options=team_players,
        default=[],
    )

    # Multi-select filter for VENUE
    selected_venues = st.sidebar.multiselect(
        "Venue",
        options=df["Venue"].unique(),
        default=[],
    )

     # Filter the data based on the selected player
    df_selection = df[df["Player"] == name]

    # Filter the data based on the selected years in the "Select Season Years" filter
    if selected_years:
        df_selection = df_selection[df_selection["Season"].isin(selected_years)]

    # Convert TOG/CBA columns to numeric
    df_selection["TOG"] = pd.to_numeric(df_selection["TOG"], errors="coerce")
    df_selection["CBA Pct"] = pd.to_numeric(df_selection["CBA Pct"], errors="coerce")

    # Filter the data based on the selected Venue
    if selected_venues:
        df_selection = df_selection[df_selection["Venue"].isin(selected_venues)]

    # Exclude rounds where players in "Without" filter played
    for without_name in without_names:
        rounds_to_exclude = df[df["Player"] == without_name]["Round"].unique()
        df_selection = df_selection[~df_selection["Round"].isin(rounds_to_exclude)]

    # Filter the data based on the selected players in the "With" filter
    for with_name in with_names:
        rounds_to_include = df[df["Player"] == with_name]["Round"].unique()
        df_selection = df_selection[df_selection["Round"].isin(rounds_to_include)]

    # Get unique opponent team names from the dataset
    opponent_teams = df["Opponent"].unique()

    # Add the "Opponent" filter as a dropdown selector in the sidebar
    selected_opponent = st.sidebar.selectbox("Opponent", [""] + sorted(list(opponent_teams)))

    # Filter the data based on the selected opponent
    if selected_opponent:
        df_selection["Opponent"] = df_selection["Opponent"]
        df_selection = df_selection[df_selection["Opponent"] == selected_opponent]

    # Select a stat column
    with col2:
        selected_column = st.selectbox("Select a stat", allowed_columns, key="select_stat")

    # Player Line Input
    with col3:
        line_value = st.number_input("Player Line", step=0.5, key="player_line")
        
    # Filter the data based on checkbox selections
    win_state = st.sidebar.checkbox("Win")
    loss_state = st.sidebar.checkbox("Loss")
    home_state = st.sidebar.checkbox("Home")
    away_state = st.sidebar.checkbox("Away")

    # Filter the data based on checkbox selections
    if win_state:
        df_selection = df_selection[df_selection["WL"] == "Win"]

    if loss_state:
        df_selection = df_selection[df_selection["WL"] == "Loss"]

    if home_state:
        df_selection = df_selection[df_selection["H/A"] == "Home"]

    if away_state:
        df_selection = df_selection[df_selection["H/A"] == "Away"]

    # Filter the data based on the selected TOG range
    selected_tog_range = st.sidebar.slider("Time on Ground (TOG)", 0, 100, (0, 100))
    df_selection = df_selection[(df_selection["TOG"] >= selected_tog_range[0]) & (df_selection["TOG"] <= selected_tog_range[1])]

    # Filter the data based on the selected CBA range
    cba_range = st.sidebar.slider("CBA PCT", 0, 100, (0, 100))
    df_selection = df_selection[(df_selection["CBA Pct"] >= cba_range[0]) & (df_selection["CBA Pct"] <= cba_range[1])]

    # Add a slider for selecting the number of x values to display on the chart
    num_x_values = st.sidebar.slider("Games Displayed", 1, len(df_selection),10)

    ############################Player vs Line Calcs########################################    
    @st.cache_data
    def calculate_statistics(data, selected_column, line_value, game_span):
        # Sort the data
        data_recent = data.iloc[-game_span:]
        data_recent[selected_column] = data_recent[selected_column].astype(float)

        # Calculate the stats for the specified column
        above_line = len(data_recent[data_recent[selected_column] > line_value])
        below_line = len(data_recent[data_recent[selected_column] < line_value])
        total_values = len(data_recent)

        return above_line, below_line, total_values

    last_3_above, last_3_below, last_3_total = calculate_statistics(df_selection, selected_column, line_value, 3)
    last_5_above, last_5_below, last_5_total = calculate_statistics(df_selection, selected_column, line_value, 5)
    last_10_above, last_10_below, last_10_total = calculate_statistics(df_selection, selected_column, line_value, 10)
    overall_above, overall_below, overall_total = calculate_statistics(df_selection, selected_column, line_value, df_selection.shape[0])
    last_3_percentage = round((last_3_above / last_3_total) * 100, 2) if last_3_total != 0 else 0
    last_5_percentage = round((last_5_above / last_5_total) * 100, 2) if last_5_total != 0 else 0
    last_10_percentage = round((last_10_above / last_10_total) * 100, 2) if last_10_total != 0 else 0
    overall_percentage = round((overall_above / overall_total) * 100, 2) if overall_total != 0 else 0

    # Define the data for each game span
    game_spans = [
        {"span": "Last 3", "above": last_3_above, "below": last_3_below},
        {"span": "L5", "above": last_5_above, "below": last_5_below},
        {"span": "L10", "above": last_10_above, "below": last_10_below},
        {"span": "Season", "above": overall_above, "below": overall_below},
    ]

    ######chart######
    # Select the columns you want in the tooltip
    tooltip_columns = ["Opponent","Venue","WL","Disposals", "Fantasy","CBA Pct","KI","Kicks","Handballs", "Marks", "Goals", "Behinds", "Shots","Hitouts", "Tackles", "CP", "UP","Clearances",  "FF", "FA", "TOG", ]

    # Remove the first 3 letters from the "MATCHUP" column
    df_selection["Opponent"] = df_selection["Opponent"].str[3:]

    # Create the Altair chart
    player_vs_line_text = (
        f"<b>{line_value} {selected_column} Hit Rate:</b> "
        + ' | '.join([
            f'{span["span"]}: {span["above"]}-{span["below"]} ({round(span["above"] / span["total"] * 100) if span["total"] != 0 else 0}%)'
            for span in [
                {"span": "Last 3", "above": last_3_above, "below": last_3_below, "total": last_3_total},
                {"span": "L5", "above": last_5_above, "below": last_5_below, "total": last_5_total},
                {"span": "L10", "above": last_10_above, "below": last_10_below, "total": last_10_total},
                {"span": "Season", "above": overall_above, "below": overall_below, "total": overall_total},
            ]
        ])
    )

    implied_odds_text = (
        "<b>Implied Odds:</b> "
        + (f'Last 3: &#36;{1 / (last_3_percentage / 100):.2f} | ' if last_3_percentage != 0 else '')
        + (f'L5: &#36;{1 / (last_5_percentage / 100):.2f} | ' if last_5_percentage != 0 else '')
        + (f'L10: &#36;{1 / (last_10_percentage / 100):.2f} | ' if last_10_percentage != 0 else '')
        + (f'Season: &#36;{1 / (overall_percentage / 100):.2f}' if overall_percentage != 0 else '')
    )

    st.write(player_vs_line_text, unsafe_allow_html=True)
    st.write(implied_odds_text, unsafe_allow_html=True)

# Create the color condition
    color_condition = alt.condition(alt.datum[selected_column] > line_value, alt.value("green"), alt.value("red"))

    # Create a new column with the clock emoji conditionally
    df_selection["clock_emoji"] = np.where(df_selection["TOG"] < 50, "⏰", "")

    # Merge Season and Round values, make it an integer, and pad single-digit rounds
    df_selection["Season_Round"] = (df_selection["Season"].astype(str) + " - R" + df_selection["Round"].astype(str).str.zfill(2)).astype(str)
    df_selection = df_selection.sort_values(by="Season_Round", ascending=False)

    # Create the bar chart
    chart = alt.Chart(df_selection.head(num_x_values)).mark_bar().encode(
        x=alt.X("Season_Round:O", axis=alt.Axis(title="", labelAngle=0)),
        y=alt.Y(f"{selected_column}:Q", axis=alt.Axis(title=None)),
        tooltip=tooltip_columns,
        color=color_condition,
    ).properties(width=800, height=400)

    # Add bars with conditional color
    colored_bars = chart.mark_bar().encode(color=color_condition)

    # Add labels dynamically at the top of each bar
    label = chart.mark_text(align='center', baseline='bottom', dy=-5, fontSize=12).encode(
            text=f"{selected_column}:Q"
    )

    # Add clock emoji above the top label
    clock_emoji_label = chart.mark_text(align='center', baseline='top', dy=5, fontSize=12).encode(
        y=alt.Y(f"{selected_column}:Q", stack="zero"),
        text="clock_emoji:N"
    )

    # Add red rule
    rule = alt.Chart(pd.DataFrame({'player_line': [line_value]})).mark_rule(color='red').encode(y='player_line:Q')

    # Combine using layer
    combined_chart = alt.layer(chart, colored_bars, label, clock_emoji_label, rule).properties(width=800, height=400)
    st.altair_chart(combined_chart, use_container_width=True)

    ######form and splits#######
    # Function to calculate mean stats for a given dataset and columns
    @st.cache_data
    def calculate_mean_stats(data, columns):
        numeric_data = data[columns].apply(pd.to_numeric, errors='coerce')
        return numeric_data.mean()

    # Function to format values with one decimal place
    def format_with_one_decimal(val):
        return f"{val:.1f}"

    # Define the columns to calculate mean for
    columns_to_mean = ["Disposals", "Fantasy","CBA Pct","KI","TOG","Kicks","Handballs", "Marks", "Goals", "Behinds","Shots","Hitouts", "Tackles", "CP", "UP","Clearances"]

    # Filter the data based on the player's name and selected years
    df_selection = df[df["Player"] == name]
    if selected_years:
        df_selection = df_selection[df_selection["Season"].isin(selected_years)]

    # Calculate mean stats for different game spans
    game_spans = [3, 5, 10, len(df_selection)]
    mean_stats = [calculate_mean_stats(df_selection.tail(span), columns_to_mean) for span in game_spans]
    
    # Create a DataFrame to store the results
    results_df = pd.DataFrame(mean_stats, columns=columns_to_mean, index=["L3", "L5", "L10", "Season"])

    # Display the results
    col1,col2=st.columns(2)
    with col1:
        st.write(f"<div style='text-align: center;'><h3>🏉 Player Form</h3></div>", unsafe_allow_html=True)
        formatted_results_df = results_df.applymap(format_with_one_decimal)
        st.write(formatted_results_df)

    # Filter data for home and away games
    home_games = df_selection[df_selection["H/A"] == "Home"]
    away_games = df_selection[df_selection["H/A"] == "Away"]

    # Calculate mean stats for home and away games
    home_stats = calculate_mean_stats(home_games, columns_to_mean)
    away_stats = calculate_mean_stats(away_games, columns_to_mean)

    # Filter data for win and loss games
    win_games = df_selection[df_selection["WL"] == "Win"]
    loss_games = df_selection[df_selection["WL"] == "Loss"]

    # Calculate mean stats for win and loss games
    win_stats = calculate_mean_stats(win_games, columns_to_mean)
    loss_stats = calculate_mean_stats(loss_games, columns_to_mean)

    # Filter data for different venues
    venue_stats = {}
    for venue in df_selection["Venue"].unique():
        venue_games = df_selection[df_selection["Venue"] == venue]
        venue_stats[venue] = calculate_mean_stats(venue_games, columns_to_mean)
        
    # Calculate mean stats for all games
    all_games_stats = calculate_mean_stats(df_selection, columns_to_mean)

    # Create a DataFrame to store the results
    results_df = pd.DataFrame([home_stats, away_stats, win_stats, loss_stats, all_games_stats] + list(venue_stats.values()), 
                              index=["Home", "Away", "Win", "Loss", "All Games"] + list(venue_stats.keys()))

    # Display the results
    formatted_results_df = results_df.applymap(format_with_one_decimal)
    formatted_results_df = formatted_results_df.drop("All Games", errors='ignore')

    with col2:
        st.write(f"<div style='text-align: center;'><h3>🏉 Season Splits</h3></div>", unsafe_allow_html=True)
        st.write(formatted_results_df, unsafe_allow_html=True)

    # Add a title for the player log table with the selected player's name
    st.markdown(f"### {name} Game Log")

    # Convert "Season" to string
    df_selection["Season"] = df_selection["Season"].astype(str)

    # Define the columns to display in the player log table
    log_columns = ["Season", "Round", "Venue", "Opponent", "WL", "Fantasy","Disposals","CBA Pct","KI", "Kicks", "Handballs", "Marks", "Goals", "Behinds", "Shots","Hitouts", "Tackles", "CP", "UP", "Clearances", "TOG"]

    # Display the player log table as a DataFrame (reversed order)
    st.dataframe(df_selection[log_columns][::-1], hide_index=True)

####hit rates
with tab2:
    # Define function to load matchups data
    @st.cache_data
    def load_matchups_data():
        matchups_worksheet = gc.open("afl stats").worksheet("Matchups")  # Use the appropriate worksheet name
        matchups_data = matchups_worksheet.get_all_values()
        matchups_df = pd.DataFrame(matchups_data[1:], columns=matchups_data[0])
        return matchups_df

    # Load matchups data
    matchups_df = load_matchups_data()

    # Extract unique seasons from the main data
    unique_seasons = df['Season'].unique().tolist()

    # Display a multi-select widget for selecting seasons
    selected_years_hit_rates = st.multiselect(
        "Select Season",
        options=unique_seasons,
        default=['2024'],  # Set the default value to the first season
        key="selected_years_hit_rates"
    )

    # Ensure at least one season is selected
    if not selected_years_hit_rates:
        selected_years_hit_rates = ['2024']  # Set the default value to the first season

    # Filter the main DataFrame based on the selected seasons
    df_hit_rates = df[df['Season'].isin(selected_years_hit_rates)]

    # Button for TOG filter (>50)
    tog_filter_button_hit_rates = st.checkbox(" >50% TOG")

    # Filter the DataFrame if the button is selected
    if tog_filter_button_hit_rates:
        df_hit_rates["TOG"] = pd.to_numeric(df_hit_rates["TOG"], errors="coerce")
        df_hit_rates = df_hit_rates[df_hit_rates["TOG"] > 49]

    # Define the content for the "Goals Cheat Sheet" section
    def goals_cheat_sheet():
        # Iterate over matchups and create expanders using the filtered DataFrame
        for index, row in matchups_df.iterrows():
            home_team, away_team = row['Home'], row['Away']

            # Skip matchups with blank home or away teams
            if home_team and away_team:
                with st.expander(f"{home_team} vs {away_team}"):
                    # Filter the data based on the selected matchup and seasons
                    filtered_df = df_hit_rates[(df_hit_rates["Team"] == home_team) | (df_hit_rates["Team"] == away_team)]
                    
                    # Get unique players in the filtered data
                    unique_players = filtered_df["Player"].unique()

                    # Group the filtered data by player and round
                    goals_df = filtered_df.pivot_table(index='Player', columns='Round', values='Goals', fill_value=None, aggfunc='sum')

                    # Add Team column
                    goals_df['Team'] = filtered_df.drop_duplicates('Player').set_index('Player')['Team']

                    # Get the original order of rounds
                    original_round_order = filtered_df['Round'].unique()

                    # Define the criteria for hit rates (1-6+ AGS)
                    criteria = range(1, 7)

                    # Calculate hit rates for each criterion and add them as new columns
                    for c in criteria:
                        hit_rate_column = f'{c}+ %'
                        total_rounds_with_criteria = (goals_df.apply(lambda x: (pd.to_numeric(x, errors='coerce').fillna(0).astype(int) >= c).sum(), axis=1))

                        # Calculate the number of rounds played by each player
                        rounds_played = filtered_df.groupby('Player')['Round'].nunique().reindex(goals_df.index).fillna(0)

                        # Calculate hit rate considering all rounds played
                        hit_rate = (total_rounds_with_criteria / rounds_played * 100).astype(int)
                        goals_df[hit_rate_column] = hit_rate.replace(np.nan, '').astype(str) + '%'

                    # Convert the 'Goals' column to numeric values
                    filtered_df['Goals'] = pd.to_numeric(filtered_df['Goals'], errors='coerce').fillna(0)

                    # Calculate average goals per player
                    avg_goals_per_player = filtered_df.groupby('Player')['Goals'].mean().reindex(goals_df.index).fillna(0)
                    goals_df['Avg'] = avg_goals_per_player.map(lambda x: '{:.1f}'.format(x))  # Display 'Avg' column with 1 decimal place

                    # Replace NaN and invalid values with blank values
                    goals_df = goals_df.replace({pd.NA: '', 'nan%': ''})

                    # Remove duplicate columns, if any
                    goals_df = goals_df.loc[:,~goals_df.columns.duplicated()]

                    # Define hit rate columns
                    hit_rate_columns = [f'{c}+ %' for c in range(1, 7)]

                    # Remove percentage sign from the hit rate columns
                    for col in hit_rate_columns:
                        goals_df[col] = goals_df[col].str.rstrip('%')

                    # Convert the hit rate columns to numeric values
                    goals_df[hit_rate_columns] = goals_df[hit_rate_columns].apply(pd.to_numeric, errors='coerce')

                    # Apply background gradient to the "Avg" and "Hit Rates" columns in the DataFrame, reversing the original order
                    st.dataframe(goals_df[['Team'] + ['Avg'] + [f'{c}+ %' for c in range(1, 7)] + [*original_round_order[::-1]]])

    # Define the content for the "Disposals Cheat Sheet" section
    def disposals_cheat_sheet():
        # Iterate over matchups and create expanders
        for index, row in matchups_df.iterrows():
            home_team, away_team = row['Home'], row['Away']

            # Skip matchups with blank home or away teams
            if home_team and away_team:
                with st.expander(f"{home_team} vs {away_team}"):
                    # Filter the data based on the selected matchup
                    filtered_df = df_hit_rates[(df_hit_rates["Team"] == home_team) | (df_hit_rates["Team"] == away_team)]
                    # Get unique players in the filtered data
                    unique_players = filtered_df["Player"].unique()

                    # Group the filtered data by player and round
                    disposals_df = filtered_df.pivot_table(index='Player', columns='Round', values='Disposals', fill_value=None, aggfunc='sum')

                    # Add Team column
                    disposals_df['Team'] = filtered_df.drop_duplicates('Player').set_index('Player')['Team']

                        # Get the original order of rounds
                    original_round_order = filtered_df['Round'].unique()

                    # Define the criteria for hit rates
                    criteria = [15 + 5 * i for i in range(6)]

                    # Calculate hit rates for each criterion and add them as new columns
                    for c in criteria:
                        hit_rate_column = f'{c}+ %'
                        total_rounds_with_criteria = (disposals_df.apply(lambda x: (pd.to_numeric(x, errors='coerce').fillna(0).astype(int) >= c).sum(), axis=1))

                        # Calculate the number of rounds played by each player
                        rounds_played = filtered_df.groupby('Player')['Round'].nunique().reindex(disposals_df.index).fillna(0)

                        # Calculate hit rate considering all rounds played
                        hit_rate = (total_rounds_with_criteria / rounds_played * 100).astype(int)
                        disposals_df[hit_rate_column] = hit_rate.replace(np.nan, '').astype(str) + '%'

                    # Convert the 'Disposals' column to numeric values
                    filtered_df['Disposals'] = pd.to_numeric(filtered_df['Disposals'], errors='coerce').fillna(0)

                    # Calculate average fantasy points per player
                    avg_disposals_per_player = filtered_df.groupby('Player')['Disposals'].mean().reindex(disposals_df.index).fillna(0)
                    disposals_df['Avg'] = avg_disposals_per_player.round(1)

                    # Replace NaN and invalid values with blank values
                    disposals_df = disposals_df.replace({pd.NA: '', 'nan%': ''})

                    # Reorder columns to reverse the order of the rounds
                    columns_order = ['Team'] + ['Avg'] + [f'{c}+ %' for c in criteria] + [*original_round_order[::-1]]
                    disposals_df = disposals_df[columns_order]

                    # Remove duplicate columns, if any
                    disposals_df = disposals_df.loc[:,~disposals_df.columns.duplicated()]

                    # Define hit rate columns
                    hit_rate_columns = [f'{c}+ %' for c in criteria]

                    # Remove percentage sign from the hit rate columns
                    for col in hit_rate_columns:
                        disposals_df[col] = disposals_df[col].str.rstrip('%')

                    # Convert the hit rate columns to numeric values
                    disposals_df[hit_rate_columns] = disposals_df[hit_rate_columns].apply(pd.to_numeric, errors='coerce')

                    # Apply background gradient to the hit rate columns in the DataFrame
                    st.dataframe(disposals_df)

    # Define the content for the "Fantasy Cheat Sheet" section
    def fantasy_cheat_sheet():
        # Iterate over matchups and create expanders
        for index, row in matchups_df.iterrows():
            home_team, away_team = row['Home'], row['Away']

            # Skip matchups with blank home or away teams
            if home_team and away_team:
                with st.expander(f"{home_team} vs {away_team}"):
                    # Filter the data based on the selected matchup
                    filtered_df = df_hit_rates[(df_hit_rates["Team"] == home_team) | (df_hit_rates["Team"] == away_team)]
 
                    # Get unique players in the filtered data
                    unique_players = filtered_df["Player"].unique()

                    # Group the filtered data by player and round
                    fantasy_df = filtered_df.pivot_table(index='Player', columns='Round', values='Fantasy', fill_value=None, aggfunc='sum')

                    # Add Team column
                    fantasy_df['Team'] = filtered_df.drop_duplicates('Player').set_index('Player')['Team']

                    # Get the original order of rounds
                    original_round_order = filtered_df['Round'].unique()

                    # Define the criteria for hit rates
                    criteria = [70, 80, 90, 100, 110, 120]

                    # Calculate hit rates for each criterion and add them as new columns
                    for c in criteria:
                        hit_rate_column = f'{c}+ %'
                        total_rounds_with_criteria = (fantasy_df.apply(lambda x: (pd.to_numeric(x, errors='coerce').fillna(0).astype(int) >= c).sum(), axis=1))

                        # Calculate the number of rounds played by each player
                        rounds_played = filtered_df.groupby('Player')['Round'].nunique().reindex(fantasy_df.index).fillna(0)

                        # Calculate hit rate considering all rounds played
                        hit_rate = (total_rounds_with_criteria / rounds_played * 100).astype(int)
                        fantasy_df[hit_rate_column] = hit_rate.replace(np.nan, '').astype(str) + '%'

                    # Convert the 'Fantasy' column to numeric values
                    filtered_df['Fantasy'] = pd.to_numeric(filtered_df['Fantasy'], errors='coerce').fillna(0)

                    # Calculate average fantasy points per player
                    avg_fantasy_per_player = filtered_df.groupby('Player')['Fantasy'].mean().reindex(fantasy_df.index).fillna(0)
                    fantasy_df['Avg'] = avg_fantasy_per_player.round(1)

                    # Replace NaN and invalid values with blank values
                    fantasy_df = fantasy_df.replace({pd.NA: '', 'nan%': ''})

                    # Reorder columns to reverse the order of the rounds
                    columns_order = ['Team'] + ['Avg'] + [f'{c}+ %' for c in criteria] + [*original_round_order[::-1]]
                    fantasy_df = fantasy_df[columns_order]

                    # Remove duplicate columns, if any
                    fantasy_df = fantasy_df.loc[:,~fantasy_df.columns.duplicated()]

                    # Define hit rate columns
                    hit_rate_columns = [f'{c}+ %' for c in criteria]

                    # Remove percentage sign from the hit rate columns
                    for col in hit_rate_columns:
                        fantasy_df[col] = fantasy_df[col].str.rstrip('%')

                    # Convert the hit rate columns to numeric values
                    fantasy_df[hit_rate_columns] = fantasy_df[hit_rate_columns].apply(pd.to_numeric, errors='coerce')

                    # Apply background gradient to the hit rate columns in the DataFrame
                    st.dataframe(fantasy_df)

    # Display radio buttons for each section in the same row
    section = st.radio("", ["Goals", "Disposals", "Fantasy"],horizontal=True)

    # Display the selected section
    if section == "Goals":
        goals_cheat_sheet()
    elif section == "Disposals":
        disposals_cheat_sheet()
    elif section == "Fantasy":
        fantasy_cheat_sheet()

######hot/cold players######
with tab3:
    col1, col2, col3, col4, col5 = st.columns([0.2,0.2,0.2,0.2,0.15])

    with col2:
        # Create a dropdown to select the stat column
        selected_column = st.selectbox("Select Stat:", columns_to_mean)

    with col3:
        # Add a slider for setting the minimum average criteria
        min_avg_criteria = st.number_input(f"Min {selected_column} Avg", min_value=0, max_value=120, value=0)

    with col4:
        # Create buttons for different game spans
        selected_span = st.radio("Game Span:", ["Last 3", "Last 5", "Last 10"],horizontal=True)

    with col5:
        # Add checkbox for TOG filter
        tog_filter = st.checkbox("TOG >50%", key="tog_filter_tab5")

    with col1:
        # Create a dropdown to select a specific team
        selected_team = st.selectbox("Select Team:", ["All"] + sorted(df["Team"].unique()))

    # Filter the data for the 2023/2024 season
    df_2023_2024 = df[df["Season"] == "2024"].sort_values(by="Round", ascending=False)

    # Apply TOG filter if toggled
    if tog_filter:
        # Convert the "TOG" column to numeric values
        df_2023_2024["TOG"] = pd.to_numeric(df_2023_2024["TOG"], errors='coerce').fillna(0)
        df_2023_2024 = df_2023_2024[df_2023_2024["TOG"] >= 50]
        
    # Filter the data for the selected team
    if selected_team != "All":
        df_2023_2024 = df_2023_2024[df_2023_2024["Team"] == selected_team]

    # Convert selected columns to numeric and fill missing values with zeros
    df_2023_2024[columns_to_mean] = df_2023_2024[columns_to_mean].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Calculate the mean for the specified columns grouped by player name
    mean_values = df_2023_2024.groupby("Player")[columns_to_mean].mean().reset_index().round(1)

    # Map the user's selection to the corresponding number of games
    game_span_mapping = {"Last 3": 3, "Last 5": 5, "Last 10": 10}

    # Calculate mean stats for the selected game span
    selected_span_games = game_span_mapping[selected_span]

    # Filter the DataFrame to keep only the selected number of games for each player
    df_selected_span = df_2023_2024.groupby("Player").head(selected_span_games)

    # Calculate the mean for the specified columns grouped by player name
    mean_stats = df_selected_span.groupby("Player")[columns_to_mean].mean().reset_index().round(1)

    # Convert selected columns to numeric and fill missing values with zeros
    df_2023_2024[columns_to_mean] = df_2023_2024[columns_to_mean].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Calculate the season averages
    season_avg = df_2023_2024.groupby("Player")[columns_to_mean].mean().reset_index().round(1)

    # Calculate the game span averages
    game_span_avg = df_2023_2024.groupby("Player").head(selected_span_games).groupby("Player")[columns_to_mean].mean().reset_index().round(1)

    # Merge the two DataFrames on the player name
    comparison_df = pd.merge(season_avg, game_span_avg, on="Player", suffixes=('_Season', f'_Last_{selected_span}'))

    # Calculate the percentage difference between season and game span averages for the selected column
    selected_column_diff = f'{selected_column}_Diff%'
    comparison_df[selected_column_diff] = ((comparison_df[f'{selected_column}_Last_{selected_span}'] - comparison_df[f'{selected_column}_Season']) / comparison_df[f'{selected_column}_Season']) * 100
    comparison_df[selected_column_diff] = comparison_df[selected_column_diff].apply(lambda x: int(x) if x.is_integer() else round(x, 1))

    col1,col2=st.columns(2)
    last_n_column_name = f'{selected_column}_Last_{selected_span}'

    # Filter players with % differences
    positive_diff_df = comparison_df[comparison_df[selected_column_diff] > 0].rename(columns={'Player': 'Player', f'{selected_column}_Season': 'Average', last_n_column_name: selected_span, selected_column_diff: 'Differential %'})
    positive_diff_df = positive_diff_df.sort_values(by='Differential %', ascending=False)

    negative_diff_df = comparison_df[comparison_df[selected_column_diff] < 0].rename(columns={'Player': 'Player', f'{selected_column}_Season': 'Average', last_n_column_name: selected_span, selected_column_diff: 'Differential %'})
    negative_diff_df = negative_diff_df.sort_values(by='Differential %', ascending=True)

    # Filter players based on the minimum average criteria
    positive_diff_df_filtered = positive_diff_df[positive_diff_df['Average'] >= min_avg_criteria]
    negative_diff_df_filtered = negative_diff_df[negative_diff_df['Average'] >= min_avg_criteria]

    with col1:
        st.write(f"<h4><b>🔥 Hot Players</b></h4>", unsafe_allow_html=True)
        st.dataframe(positive_diff_df_filtered[['Player', selected_span, 'Average', 'Differential %']], hide_index=True)

    with col2:
        st.write(f"<h4><b>🥶 Cold Players</b></h4>", unsafe_allow_html=True)
        st.dataframe(negative_diff_df_filtered[['Player', selected_span, 'Average', 'Differential %']], hide_index=True)

##DVP###
with tab4:
    st.title(f"Season DVP Rankings (TOG Weighted)")
    # Function to load player positions
    @st.cache_data
    def load_player_positions():
        players_worksheet = gc.open("afl stats").worksheet("players")  # Adjust "afl stats" to your Google Sheet name if different
        players_data = players_worksheet.get_all_values()
        players_df = pd.DataFrame(players_data[1:], columns=players_data[0])
        return players_df[['Player', 'Position']]  # Adjust columns as per your sheet

    # Load data from Google Sheets
    players_df = load_player_positions()
    game_data_df = load_data()

    # Initialize an empty DataFrame to store position-wise statistics for disposals
    all_positions_stats = pd.DataFrame()

    c1,c2=st.columns(2)
    # Select the statistic for all positions
    with c1:
        selected_statistic = st.selectbox("Stat", ["Disposals", "Fantasy", "Goals"])
    with c2:
        selected_years = st.multiselect(
            "Season",
            options=unique_seasons,
            default=['2024'],  # Set the default value to the first season
            key="8"
        )

    # Iterate over each unique position type
    for position_type in sorted(players_df["Position"].unique()):
            # Filter game_data_df based on selected years
        game_data_df = game_data_df[game_data_df['Season'].isin(selected_years)]

        # Filter game data for the current position type
        filtered_players_names = players_df[players_df["Position"] == position_type]["Player"].tolist()
        position_filtered_game_data = game_data_df[game_data_df["Player"].isin(filtered_players_names)]

        # Normalize selected statistic by TOG (Time on Ground)
        position_filtered_game_data[selected_statistic] = pd.to_numeric(position_filtered_game_data[selected_statistic], errors='coerce')
        position_filtered_game_data["TOG"] = pd.to_numeric(position_filtered_game_data["TOG"], errors='coerce')

        position_filtered_game_data[selected_statistic] = position_filtered_game_data[selected_statistic] * (100 / position_filtered_game_data["TOG"])

        # Group by Opponent, then calculate mean for the normalized selected statistic
        opponent_avg_stats = position_filtered_game_data.groupby('Opponent')[selected_statistic].mean().reset_index()

        # Add position type column
        opponent_avg_stats["Position"] = position_type

        # Append the position-wise selected statistic to the overall DataFrame
        all_positions_stats = pd.concat([all_positions_stats, opponent_avg_stats])

    # Also, calculate statistics for all players irrespective of position
    all_players_data = game_data_df.copy()
    all_players_data[selected_statistic] = pd.to_numeric(all_players_data[selected_statistic], errors='coerce')
    all_players_data["TOG"] = pd.to_numeric(all_players_data["TOG"], errors='coerce')
    all_players_data[selected_statistic] = all_players_data[selected_statistic] * (100 / all_players_data["TOG"])
    all_players_stats = all_players_data.groupby('Opponent')[selected_statistic].mean().reset_index()
    all_players_stats["Position"] = "All"
    all_positions_stats = pd.concat([all_positions_stats, all_players_stats])

    # Pivot the DataFrame to have positions as columns
    pivoted_df = all_positions_stats.pivot(index='Opponent', columns='Position', values=selected_statistic)

    # Function to determine background color
    def background_color(val):
        if val.name not in ["Opponent"]:
            top3 = val.nlargest(3).index
            bottom3 = val.nsmallest(3).index
            return ['background-color: lightgreen' if idx in top3 else 'background-color: lightcoral' if idx in bottom3 else '' for idx in val.index]
        else:
            return [''] * len(val)

    # Apply background color function and display the DataFrame
    styled_df = pivoted_df.style.apply(background_color, axis=0).format("{:.1f}")
    st.write(styled_df)

    c1, c2 = st.columns(2)
    with c2:
        selected_team = st.selectbox("Opponent", options=sorted(game_data_df["Opponent"].unique()))
    with c1:
        selected_position = st.selectbox("Position", options=sorted(players_df["Position"].unique()))
        
    # Ensure numerical columns are of the correct type
    numerical_cols = ["Disposals", "Goals", "Fantasy", "Kicks", "Handballs", "Marks"]
    for col in numerical_cols:
        game_data_df[col] = pd.to_numeric(game_data_df[col], errors='coerce')

    # Filter using the selected years
    filtered_players_names = players_df[players_df["Position"] == selected_position]["Player"].tolist()
    filtered_game_data = game_data_df[
        (game_data_df["Player"].isin(filtered_players_names)) & 
        (game_data_df["Opponent"] == selected_team) & 
        (game_data_df["Season"].isin(selected_years))
    ]
    
    # Columns to display
    columns_to_display = ["Round", "Player", "Team", "Disposals", "Fantasy", "Goals", "Behinds", "Kicks", "Handballs", "Marks", "CBA Pct", "KI", "TOG"]
    st.dataframe(filtered_game_data[columns_to_display].iloc[::-1], hide_index=True)

########team dashboard"
with tab5:
    # filters
    c1, c2, c3, c4, c5,c6 = st.columns(6)
    with c2:
        selected_team_tab7 = st.selectbox("Team", options=sorted(df["Team"].unique()), key="team_select_tab7")
    with c6:
        home_away_filter = st.radio("Home/Away", ["All", "Home", "Away"], key="home_away_radio_tab7")
    with c5:
        win_loss_filter = st.radio("Win/Loss", ["All", "Win", "Loss"], key="win_loss_radio_tab7")
    with c3:
        selected_venue_tab7 = st.multiselect("Venue", options=sorted(df["Venue"].unique()), key="venue_select_tab7")
    with c1:
        selected_season_tab7 = st.multiselect("Season", options=sorted(df["Season"].unique()), default=["2024"], key="season_select_tab7")
    with c4:
        without_names = st.multiselect("Without", options=sorted(df[df["Team"] == selected_team_tab7]["Player"].unique()), key="without_players_tab7")

    # Filter the DataFrame based on the selections
    df_filtered_tab7 = df[(df["Team"] == selected_team_tab7) & (df["Season"].isin(selected_season_tab7))]

    # Apply Home/Away filter
    if home_away_filter != "All":
        df_filtered_tab7 = df_filtered_tab7[df_filtered_tab7["H/A"] == home_away_filter]

    # Apply Win/Loss filter
    if win_loss_filter != "All":
        df_filtered_tab7 = df_filtered_tab7[df_filtered_tab7["WL"] == win_loss_filter]

    # Apply Venue filter
    if selected_venue_tab7:
        df_filtered_tab7 = df_filtered_tab7[df_filtered_tab7["Venue"].isin(selected_venue_tab7)]

    for column in allowed_columns:
        # Attempt to convert each column to numeric, coercing errors to NaN
        df_filtered_tab7[column] = pd.to_numeric(df_filtered_tab7[column], errors='coerce')

    # Apply Without Player filter    
    for without_name in without_names:
        rounds_to_exclude = df[df["Player"] == without_name]["Round"].unique()
        df_filtered_tab7 = df_filtered_tab7[~df_filtered_tab7["Round"].isin(rounds_to_exclude)]

    for column in allowed_columns:
        # Attempt to convert each column to numeric, coercing errors to NaN
        df_filtered_tab7[column] = pd.to_numeric(df_filtered_tab7[column], errors='coerce')

    # mean for the first table
    player_stats = df_filtered_tab7.groupby("Player")[allowed_columns].mean().reset_index().round(1)

    # Second table with only season and team filters, calculating player averages
    df_filtered_tab7_second = df[(df["Team"] == selected_team_tab7) & (df["Season"].isin(selected_season_tab7))]
    for column in allowed_columns:
        # Attempt to convert each column to numeric, coercing errors to NaN
        df_filtered_tab7_second[column] = pd.to_numeric(df_filtered_tab7_second[column], errors='coerce')

    # mean for the second table
    player_stats_second = df_filtered_tab7_second.groupby("Player")[allowed_columns].mean().reset_index().round(1)

     # Merge the two tables on the player column, using a left join to keep all players from the first table
    player_stats_merged = player_stats.merge(player_stats_second, on="Player", suffixes=('_Adj', '_Avg'), how="left")

    # Replace NaN values in the second table with "N/A"
    for col in allowed_columns:
        player_stats_merged[col+'_Avg'].fillna("N/A", inplace=True)

    # Interleave the statistics from both tables
    interleaved_columns = ['Player']
    for col in allowed_columns:
        interleaved_columns.append(f"{col} (Adj)")
        interleaved_columns.append(f"{col} (Avg)")

    # Your data manipulation code
    interleaved_stats_df = pd.DataFrame(columns=interleaved_columns)

    for index, row in player_stats_merged.iterrows():
        interleaved_row = [row['Player']]
        for col in allowed_columns:
            interleaved_row.extend([row[f'{col}_Adj'], row[f'{col}_Avg']])
        interleaved_stats_df.loc[index] = interleaved_row

    interleaved_stats_df.set_index("Player", inplace=True)

    # Custom formatting function
    def format_without_trailing_zeros(value):
        if pd.notnull(value) and isinstance(value, (int, float)):
            return f"{int(value)}" if value == int(value) else f"{value:.1f}"
        return str(value) if pd.notnull(value) else ""

    # Apply color coding to the interleaved DataFrame
    interleaved_stats_with_style = interleaved_stats_df.copy()

    def color_cells(row):
        styles = [''] * len(row)
        for col in allowed_columns:
            col_filtered = f"{col} (Adj)"
            col_avg = f"{col} (Avg)"

            if row[col_filtered] != 'N/A' and row[col_avg] != 'N/A':
                if float(row[col_filtered]) > float(row[col_avg]):
                    styles[interleaved_stats_with_style.columns.get_loc(col_filtered)] = 'background-color: lightgreen'
                elif row[col_filtered] < row[col_avg]:
                    styles[interleaved_stats_with_style.columns.get_loc(col_filtered)] = 'background-color: lightcoral'
        return styles

    # Apply formatting and styling
    styled_interleaved_stats = interleaved_stats_with_style.style.format(format_without_trailing_zeros).apply(color_cells, axis=1)

    # Display the styled DataFrame
    st.dataframe(styled_interleaved_stats, hide_index=False)
# CBA Chart
with tab6:
    st.title("CBA Breakdown (%)")
    c1, c2 = st.columns(2)

    with c1:
        # Team filter
        selected_team_tab8 = st.selectbox("Select a Team", options=sorted(df["Team"].unique()))

    with c2:
        # Season filter
        selected_season_tab8 = st.selectbox("Select a Season", options=sorted(df["Season"].unique()), key="season_se", index=df["Season"].unique().tolist().index("2024"))

    # Filter data based on selected team and season
    filtered_data = df[(df["Team"] == selected_team_tab8) & (df["Season"] == selected_season_tab8)]

    # Check if the DataFrame is not empty before proceeding
    if not filtered_data.empty:
        # Convert "CBA Pct" column to numeric
        filtered_data["CBA Pct"] = pd.to_numeric(filtered_data["CBA Pct"], errors="coerce")

        # Calculate average CBA Pct for each player and round to integers
        avg_cba = filtered_data.groupby('Player')['CBA Pct'].mean().round(0).astype(int)

        # Pivot the data to have players as rows and rounds as columns, with CBA Pct as cell values
        cba_pivot = filtered_data.pivot(index='Player', columns='Round', values='CBA Pct')

        # Add a column for average CBA Pct as the first column after the index column
        cba_pivot.insert(0, 'Avg', avg_cba)

        # Apply background gradient to the DataFrame
        styled_cba_pivot = cba_pivot.style.background_gradient(cmap='RdYlGn', axis=None)

        # Apply formatting to retain desired decimal places
        styled_cba_pivot = styled_cba_pivot.format("{:.0f}")

        # Display the styled DataFrame
        st.dataframe(styled_cba_pivot)
    else:
        # Display a message or do nothing if the DataFrame is empty
        st.write("DataFrame is empty. No data to display.")

# KI Chart
with tab7:
    st.title("KI Breakdown")
    c1, c2 = st.columns(2)

    with c1:
        # Team filter
        selected_team_tab8 = st.selectbox("Select a Team", options=sorted(df["Team"].unique()), key="team_select")

    with c2:
        # Season filter
        selected_season_tab8 = st.selectbox("Select a Season", options=sorted(df["Season"].unique()), key="season_select", index=df["Season"].unique().tolist().index("2024"))

    # Filter data based on selected team and season
    filtered_data = df[(df["Team"] == selected_team_tab8) & (df["Season"] == selected_season_tab8)]

    # Check if the DataFrame is not empty before proceeding
    if not filtered_data.empty:
        # Convert "KI" column to numeric
        filtered_data["KI"] = pd.to_numeric(filtered_data["KI"], errors="coerce")

        # Calculate average KI for each player and round to 1 decimal place
        avg_ki = filtered_data.groupby('Player')['KI'].mean().round(1)

        # Pivot the data to have players as rows and rounds as columns, with KI as cell values
        ki_pivot = filtered_data.pivot(index='Player', columns='Round', values='KI')

        # Add a column for average KI as the first column after the index column
        ki_pivot.insert(0, 'Avg', avg_ki)

        # Apply background gradient to the DataFrame
        styled_ki_pivot = ki_pivot.style.background_gradient(cmap='RdYlGn', axis=None)

        # Set formatting for gaverage column (1 decimal) and KI values columns (0 decimals)
        styled_ki_pivot = styled_ki_pivot.format({'Avg': '{:.1f}', **{col: '{:.0f}' for col in ki_pivot.columns if col != 'Avg'}})

        # Display the styled DataFrame
        st.dataframe(styled_ki_pivot)
    else:
        # Display a message or do nothing if the DataFrame is empty
        st.write("DataFrame is empty. No data to display.")

with tab8:
    # Select numeric columns
    numeric_columns = ["Disposals", "Fantasy", "UP", "CP", "Marks"]

    # Convert selected columns to numeric and fill missing values with zeros
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Season filter
    selected_season = st.selectbox("Select Season", options=sorted(df["Season"].unique()), index=len(df["Season"].unique()) - 1)

    # Filter DataFrame based on the selected season
    df_season = df[df["Season"] == selected_season]

    # Initialize empty DataFrames to store team statistics
    team_stats_for = pd.DataFrame(columns=["Team"] + numeric_columns)
    team_stats_against = pd.DataFrame(columns=["Team"] + numeric_columns)

    # Iterate over each team
    for team in sorted(df_season["Team"].unique()):
        # Filter data for the current team
        team_data = df_season[df_season["Team"] == team]

        # Calculate total statistics for the current team
        total_stats_for = team_data[numeric_columns].sum()
        total_stats_against = df_season[df_season["Opponent"].str[3:] == team][numeric_columns].sum()

        # Calculate the total number of unique rounds played by the team
        unique_rounds_played = team_data["Round"].nunique()

        # Calculate the average statistics for and against the current team
        avg_stats_for = total_stats_for / unique_rounds_played
        avg_stats_against = total_stats_against / unique_rounds_played

        # Assign team name to the statistics
        avg_stats_for["Team"] = team
        avg_stats_against["Team"] = team

        # Concatenate the calculated statistics to the DataFrame
        team_stats_for = pd.concat([team_stats_for, avg_stats_for.to_frame().T], ignore_index=True)
        team_stats_against = pd.concat([team_stats_against, avg_stats_against.to_frame().T], ignore_index=True)

    # Set team name as index
    team_stats_for.set_index("Team", inplace=True)
    team_stats_against.set_index("Team", inplace=True)

    # Apply background gradient to each statistic separately in the table displaying team stats for
    styled_team_stats_for = team_stats_for.style.background_gradient(subset=numeric_columns, cmap='RdYlGn').format("{:.1f}")

    # Apply background gradient to each statistic separately in the table displaying team stats against
    styled_team_stats_against = team_stats_against.style.background_gradient(subset=numeric_columns, cmap='RdYlGn').format("{:.1f}")

    # Display the styled DataFrames
    c1, c2 = st.columns(2)
    with c1:
        st.title("For")
        st.write(styled_team_stats_for)
    with c2:
        st.title("Against")
        st.write(styled_team_stats_against)












