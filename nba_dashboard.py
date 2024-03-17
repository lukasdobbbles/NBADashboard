import pandas as pd
import streamlit as st
import altair as alt
import gspread
import gc
from oauth2client.service_account import ServiceAccountCredentials
import os
import dotenv
dotenv.load_dotenv()
import ast

# Set page configuration
st.set_page_config(page_title="NBA Dashboard", layout="wide")

# Authenticate with Google Sheets API
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_dict(ast.literal_eval(os.environ["credentials"]), scope)
gc = gspread.authorize(credentials)

# Define a function to load data from Google Sheets
@st.cache_data(ttl=3600, hash_funcs={pd.DataFrame: lambda x: None})
def load_data():
    worksheet = gc.open("NBA Scores scrapper").worksheet("logs")
    data = worksheet.get_all_values()
    df = pd.DataFrame(data[1:], columns=data[0])
    return df

# Load initial data from Google Sheets using the cached function
df = load_data()

# Convert and sort 'GAME_DATE'
df['GAME_DATE'] = df['GAME_DATE'].str.replace(' 0:00:00', 'T00:00:00')
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
df = df.sort_values(by='GAME_DATE')

# Define the allowed columns for filtering and plotting
allowed_columns = ["PTS","REB","AST","STL","BLK","MIN","PRA","PA","PR","RA", "FGA","FG3A","FG3M","FTA","TOV","STOCKS","DD2","TD3","REB_CHANCES","POT_AST"]

# Define a function to calculate the derived columns
def calculate_derived_columns(data):
    data["PRA"] = data["PTS"] + data["REB"] + data["AST"]
    data["PA"] = data["PTS"] + data["AST"]
    data["PR"] = data["PTS"] + data["REB"]
    data["RA"] = data["REB"] + data["AST"]
    data["STOCKS"] = data["STL"] + data["BLK"]

# Convert selected columns to integers
int_columns = ["PTS", "REB", "AST", "STL", "BLK"]
df[int_columns] = df[int_columns].apply(pd.to_numeric, errors='coerce')

# Call the function to calculate the derived columns
calculate_derived_columns(df)

tab_names = ["Player Dashboard", "Game Logs", "DVP", "Team Dashboard", "Trending","Play Types x Shooting Zones"]
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_names)

######################################FILTERS######################################
with tab1:
    # Multi-select filter for SEASON_YEAR
    st.sidebar.markdown("## Dashboard Filters")
    selected_years = st.sidebar.multiselect(
        "Season",
        options=df["SEASON_YEAR"].unique(),
        default=["2023-24"],  # Set "2023-24" as the default season
        key="season_year_multiselect"  # Add a unique key here
    )
    
    # Ensure at least one season is selected
    if not selected_years:
        selected_years = ["2023-24"]
        
    col1, col2, col3 = st.columns(3)
    
    # Player Name selection
    with col1:
        name = st.selectbox(
            "Player Name",
            options=df["PLAYER_NAME"].unique(),
        )

    # Retrieve the corresponding player ID for the selected player name
    player_id = df[df["PLAYER_NAME"] == name]["PLAYER_ID"].iloc[0]

    # Retrieve all team abbreviations for the selected player across all seasons
    all_team_abbreviations = df[df["PLAYER_ID"] == player_id]["TEAM_ABBREVIATION"].unique()

    # Get all unique player names from all teams associated with the selected player
    all_player_names = df[df["TEAM_ABBREVIATION"].isin(all_team_abbreviations)]["PLAYER_NAME"].unique()

    # "Without" Filter (Default: No Players Selected)
    without_names = st.sidebar.multiselect(
        "Without Player/s",
        options=all_player_names,
        default=[],
    )

    # "With" Filter (Default: No Players Selected)
    with_names = st.sidebar.multiselect(
        "With Player/s",
        options=all_player_names,
        default=[],
    )

    # Filter the data based on the selected player's ID
    df_selection = df[df["PLAYER_ID"] == player_id]

    # Filter the data based on the selected years in the "Select Season Years" filter
    if selected_years:
        df_selection = df_selection[df_selection["SEASON_YEAR"].isin(selected_years)]

    # Filter the data based on the selected players in the "Without" filter
    if without_names and without_names != [name]:
        # Get the game IDs of selected players from the same team and season year
        game_ids_same_team = set(df[(df["PLAYER_NAME"].isin(without_names)) & (df["TEAM_ABBREVIATION"].isin(all_team_abbreviations)) & (df["SEASON_YEAR"].isin(selected_years))]["GAME_ID"].unique())

        # Exclude games with intersecting game IDs from the selection for the selected player
        df_selection = df_selection[~df_selection["GAME_ID"].isin(game_ids_same_team)]

    # Filter the data based on the selected players in the "With" filter
    if with_names and any(all_team_abbreviations):
        common_game_ids = None
        for other_name in with_names:
            other_player_id = df[df["PLAYER_NAME"] == other_name]["PLAYER_ID"].iloc[0]
            other_player_game_ids = set(df[(df["PLAYER_ID"] == other_player_id) & (df["TEAM_ABBREVIATION"].isin(all_team_abbreviations)) & (df["SEASON_YEAR"].isin(selected_years))]["GAME_ID"].unique())
            common_game_ids = common_game_ids.intersection(other_player_game_ids) if common_game_ids is not None else other_player_game_ids

        if common_game_ids:
            df_selection = df_selection[df_selection["GAME_ID"].isin(common_game_ids)]
        else:
            df_selection = pd.DataFrame(columns=df.columns)

    # Sort the data by "GAME_DATE" in descending order after filtering
    df_selection = df_selection.sort_values(by="GAME_DATE", ascending=False)

    # Get unique opponent team names from the dataset
    opponent_teams = df["MATCHUP"].str[-3:].unique()

    # Add the "Opponent" filter as a dropdown selector in the sidebar
    selected_opponent = st.sidebar.selectbox("Opponent", [""] + sorted(list(opponent_teams)))

    # Filter the data based on the selected opponent
    if selected_opponent:
        # Extract the last 3 letters of the "MATCHUP" column to determine the opponent
        df_selection["Opponent"] = df_selection["MATCHUP"].str[-3:]
        df_selection = df_selection[df_selection["Opponent"].str.upper() == selected_opponent.upper()]

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
        df_selection = df_selection[df_selection["WL"] == "W"]

    if loss_state:
        df_selection = df_selection[df_selection["WL"] == "L"]

    if home_state:
        df_selection = df_selection[~df_selection["MATCHUP"].str.contains("@")]

    if away_state:
        df_selection = df_selection[df_selection["MATCHUP"].str.contains("@")]

    # Convert and sort 'GAME_DATE'
    df_selection['GAME_DATE'] = pd.to_datetime(df_selection['GAME_DATE'])
    df_selection = df_selection.sort_values(by='GAME_DATE')

    # Calculate the difference in days between consecutive game dates
    df_selection['Days_Between_Games'] = df_selection['GAME_DATE'].diff().dt.days

    # Define a function to categorize rest days
    @st.cache_data
    def categorize_rest_days(days_diff):
        if days_diff == 1:
            return 0
        elif days_diff == 2:
            return 1
        elif days_diff >= 3:
            return 2  # 3 or more days difference is 2 or more rest days
        else:
            return None  # Not a rest day

    # Apply the categorize_rest_days function to create a new column 'Rest_Days'
    df_selection['Rest_Days'] = df_selection['Days_Between_Games'].apply(categorize_rest_days)

    # Create checkboxes for selecting rest days
    zero_rest_days = st.sidebar.checkbox("Back to Back")
    one_rest_day = st.sidebar.checkbox("1 Rest Day")
    two_or_more_rest_days = st.sidebar.checkbox("2+ Rest Days")

    # Filter the data based on the selected checkboxes
    if zero_rest_days or one_rest_day or two_or_more_rest_days:
        if not zero_rest_days:
            df_selection = df_selection[df_selection['Rest_Days'] != 0]
        if not one_rest_day:
            df_selection = df_selection[df_selection['Rest_Days'] != 1]
        if not two_or_more_rest_days:
            df_selection = df_selection[df_selection['Rest_Days'] != 2]

    # Add a slider for selecting the number of x values to display on the chart
    num_x_values = st.sidebar.slider("Games Displayed", 1, len(df_selection),10)

    #minutes filter
    minutes_range = st.sidebar.slider("Minutes Played", 0, 48, (0, 48))
    df_selection['MIN'] = df_selection['MIN'].astype(float)
    df_selection = df_selection[(df_selection["MIN"] >= minutes_range[0]) & (df_selection["MIN"] <= minutes_range[1])]

    ############################Player vs Line Calcs########################################    
    @st.cache_data
    def calculate_statistics(data, selected_column, line_value, game_span):
        # Filter out rows with empty strings in the selected column
        data = data[data[selected_column] != '']
        
        # Sort the data by "GAME_DATE" in ascending order
        data = data.sort_values(by="GAME_DATE")
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
    last_20_above, last_20_below, last_20_total = calculate_statistics(df_selection, selected_column, line_value, 20)
    overall_above, overall_below, overall_total = calculate_statistics(df_selection, selected_column, line_value, df_selection.shape[0])
    last_3_percentage = round((last_3_above / last_3_total) * 100, 2) if last_3_total != 0 else 0
    last_5_percentage = round((last_5_above / last_5_total) * 100, 2) if last_5_total != 0 else 0
    last_10_percentage = round((last_10_above / last_10_total) * 100, 2) if last_10_total != 0 else 0
    last_20_percentage = round((last_20_above / last_20_total) * 100, 2) if last_20_total != 0 else 0
    overall_percentage = round((overall_above / overall_total) * 100, 2) if overall_total != 0 else 0

        
    # Define the data for each game span
    game_spans = [
        {"span": "Last 3", "above": last_3_above, "below": last_3_below},
        {"span": "L5", "above": last_5_above, "below": last_5_below},
        {"span": "L10", "above": last_10_above, "below": last_10_below},
        {"span": "L20", "above": last_20_above, "below": last_20_below},
        {"span": "Season", "above": overall_above, "below": overall_below},
    ]

    ######chart######
    # Select the columns you want in the tooltip
    tooltip_columns = ["PTS", "REB","REB_CHANCES", "AST", "POT_AST", "TOV", "STL", "BLK", "PF", "PLUS_MINUS", alt.Tooltip("MIN:Q", format=".0f"), "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "WL"]

    # Remove the first 3 letters from the "MATCHUP" column
    df_selection["MATCHUP"] = df_selection["MATCHUP"].str[3:]

    # Convert the "GAME_DATE" column to a datetime format and format it
    df_selection['GAME_DATE'] = pd.to_datetime(df_selection['GAME_DATE'])
    df_selection['unique_matchup'] = df_selection['GAME_DATE'].dt.strftime("%b %d, %Y")  # Include the year in the formatting
    df_selection = df_selection.sort_values(by="GAME_DATE", ascending=False)

    # Create the Altair chart
    player_vs_line_text = (
        f"<b>{line_value} {selected_column} Hit Rate:</b> "
        + ' | '.join([
            f'{span["span"]}: {span["above"]}-{span["below"]} ({round(span["above"] / span["total"] * 100) if span["total"] != 0 else 0}%)'
            for span in [
                {"span": "Last 3", "above": last_3_above, "below": last_3_below, "total": last_3_total},
                {"span": "L5", "above": last_5_above, "below": last_5_below, "total": last_5_total},
                {"span": "L10", "above": last_10_above, "below": last_10_below, "total": last_10_total},
                {"span": "L20", "above": last_20_above, "below": last_20_below, "total": last_20_total},
                {"span": "Season", "above": overall_above, "below": overall_below, "total": overall_total},
            ]
        ])
    )
    
    implied_odds_text = (
        "<b>Implied Odds:</b> "
        + (f'Last 3: &#36;{1 / (last_3_percentage / 100):.2f} | ' if last_3_percentage != 0 else '')
        + (f'L5: &#36;{1 / (last_5_percentage / 100):.2f} | ' if last_5_percentage != 0 else '')
        + (f'L10: &#36;{1 / (last_10_percentage / 100):.2f} | ' if last_10_percentage != 0 else '')
        + (f'L20: &#36;{1 / (last_20_percentage / 100):.2f} | ' if last_20_percentage != 0 else '')
        + (f'Season: &#36;{1 / (overall_percentage / 100):.2f}' if overall_percentage != 0 else '')
    )

    st.write(player_vs_line_text, unsafe_allow_html=True)
    st.write(implied_odds_text, unsafe_allow_html=True)
    chart = alt.Chart(df_selection.head(num_x_values)).mark_bar().encode(
        y=alt.Y(f"{selected_column}:Q", axis=alt.Axis(title=None)),
        x=alt.X("unique_matchup:N", axis=alt.Axis(title="Most Recent Games", labelAngle=-75), sort=None),
        tooltip=tooltip_columns,
        text='MATCHUP:N'
    ).properties(width=800, height=400)

    # Add bars with conditional color and red line
    color_condition = alt.condition(alt.datum[selected_column] > line_value, alt.value("green"), alt.value("red"))
    colored_bars = chart.mark_bar().encode(color=color_condition)
    rule = alt.Chart(pd.DataFrame({'player_line': [line_value]})).mark_rule(color='red').encode(y='player_line:Q')

    # Add labels dynamically at the top and bottom of each bar
    top_label = chart.mark_text(align='center', dy=10, fontSize=12, color="white").encode(x=alt.X("unique_matchup:N", sort=None), y=alt.Y(f"{selected_column}:Q", stack="zero"), text=f"{selected_column}:Q")
    bottom_label = chart.mark_text(align='center', baseline='bottom', dy=-5, fontSize=12).encode(x=alt.X("unique_matchup:N", sort=None), y=alt.Y(f"{selected_column}:Q", stack="zero"), text='MATCHUP:N')

    # Combine
    combined_label_chart = top_label + bottom_label
    combined_chart = (chart + colored_bars + combined_label_chart + rule)
    st.altair_chart(combined_chart, use_container_width=True)

######form and splits#######
    # Function to calculate rest days and categorize them
    @st.cache_data
    def calculate_rest_days(data):
        data['GAME_DATE'] = pd.to_datetime(data['GAME_DATE'])
        data = data.sort_values(by='GAME_DATE')
        data['Days_Between_Games'] = data['GAME_DATE'].diff().dt.days

        def categorize_rest_days(days_diff):
            if days_diff == 1:
                return 0
            elif days_diff == 2:
                return 1
            elif days_diff >= 3:
                return 2
            else:
                return None

        data['Rest_Days'] = data['Days_Between_Games'].apply(categorize_rest_days)
        return data

    # Function to calculate mean stats for a given dataset and columns
    @st.cache_data
    def calculate_mean_stats(data, columns):
        return data[columns].mean()

    # Function to format values with one decimal place
    @st.cache_data
    def format_with_one_decimal(val):
        return f"{val:.1f}"

    # Define the columns to calculate mean for
    columns_to_mean = [
        "MIN", "PTS", "REB", "AST", "STL", "BLK", "FGA", "FG3A", "FTA", "TOV", "PRA", "PA", "PR", "RA", "STOCKS"
    ]

    # Filter the data based on the player's name and selected years
    df_selection = df[df["PLAYER_NAME"] == name]
    if selected_years:
        df_selection = df_selection[df_selection["SEASON_YEAR"].isin(selected_years)]

    # Convert 'GAME_DATE' to datetime and sort by it in descending order
    df_selection['GAME_DATE'] = pd.to_datetime(df_selection['GAME_DATE'])
    df_selection = df_selection.sort_values(by='GAME_DATE', ascending=False)

    # Convert selected columns to numeric and fill missing values with zeros
    for column in columns_to_mean:
        df_selection[column] = pd.to_numeric(df_selection[column], errors='coerce').fillna(0)

    # Calculate mean stats for different game spans
    game_spans = [3, 5, 10, 20, len(df_selection)]
    mean_stats = [calculate_mean_stats(df_selection.head(span), columns_to_mean) for span in game_spans]

    # Create a DataFrame to store the results
    results_df = pd.DataFrame(mean_stats, columns=columns_to_mean, index=["L3", "L5", "L10", "L20", "Season"])

    # Display the results
    col1,col2=st.columns(2)
    with col1:
        st.write(f"<div style='text-align: center;'><h3>🏀 Player Form</h3></div>", unsafe_allow_html=True)
        formatted_results_df = results_df.applymap(format_with_one_decimal)
        st.write(formatted_results_df)

    # Filter data for home and away games
    home_games = df_selection[~df_selection["MATCHUP"].str.contains("@")]
    away_games = df_selection[df_selection["MATCHUP"].str.contains("@")]

    # Calculate mean stats for home and away games
    home_stats = calculate_mean_stats(home_games, columns_to_mean)
    away_stats = calculate_mean_stats(away_games, columns_to_mean)

    # Filter data for win and loss games
    win_games = df_selection[df_selection["WL"] == "W"]
    loss_games = df_selection[df_selection["WL"] == "L"]

    # Calculate mean stats for win and loss games
    win_stats = calculate_mean_stats(win_games, columns_to_mean)
    loss_stats = calculate_mean_stats(loss_games, columns_to_mean)

    # Create a DataFrame to store the results
    results_df = pd.DataFrame([home_stats, away_stats, win_stats, loss_stats], index=["Home", "Away", "Win", "Loss"])

    # Display the results
    formatted_results_df = results_df.applymap(format_with_one_decimal)

    # Convert the DataFrame to an HTML table
    html_table = formatted_results_df.to_html(classes='table table-bordered table-hover', justify='left', border=0, index_names=False)

    # Calculate the mean stats for games with 0, 1, and 2+ days of rest
    df_selection = calculate_rest_days(df_selection)
    rest_0_stats = calculate_mean_stats(df_selection[df_selection['Rest_Days'] == 0], columns_to_mean)
    rest_1_stats = calculate_mean_stats(df_selection[df_selection['Rest_Days'] == 1], columns_to_mean)
    rest_2_stats = calculate_mean_stats(df_selection[df_selection['Rest_Days'] == 2], columns_to_mean)

    # Create a DataFrame to store the results
    rest_day_results = pd.DataFrame([rest_0_stats, rest_1_stats, rest_2_stats], index=["0 Days Rest", "1 Day Rest", "2+"])

    # Display the results
    formatted_rest_day_results = rest_day_results.applymap(format_with_one_decimal)

    # Merge all the data into one DataFrame
    merged_results = pd.concat([formatted_results_df, formatted_rest_day_results])

    with col2:
        # Display the merged table
        st.write(f"<div style='text-align: center;'><h3>🏀 Season Splits</h3></div>", unsafe_allow_html=True)
        st.write(merged_results, use_container_width=True)

##player logs###
           
    with tab2:
        st.write(f"<div style='text-align: left;'><h3><b>Player Game Logs</b></h3></div>", unsafe_allow_html=True)
        # Filter for Player Name on the Game Logs tab
        selected_player_logs = st.selectbox(
            "Select Player",
            options=sorted(df["PLAYER_NAME"].unique()),
            key="player_name_game_logs"
        )

        # Filter the game logs based on the selected player name
        df_game_logs = df[df["PLAYER_NAME"] == selected_player_logs]

        # Ensure 'GAME_DATE' is in datetime format
        df_game_logs['GAME_DATE'] = pd.to_datetime(df_game_logs['GAME_DATE'])

        # Sort the DataFrame by 'GAME_DATE' in descending order
        df_game_logs = df_game_logs.sort_values(by='GAME_DATE', ascending=False)

        # Extract only the date part from the 'GAME_DATE' column
        df_game_logs['GAME_DATE'] = df_game_logs['GAME_DATE'].dt.date

        # List of columns to hide
        columns_to_hide = ['PLAYER_ID', 'NICKNAME', 'TEAM_ID', 'TEAM_NAME', 'GAME_ID', 'TEAM_ABBREVIATION', 'PLAYER_NAME', 'SEASON_YEAR']

        # Drop the columns to hide
        df_game_logs = df_game_logs.drop(columns=columns_to_hide, axis=1)

        # Find the index of the 'POT_AST' column
        try:
            pot_ast_index = df_game_logs.columns.get_loc('POT_AST')
            # If 'POT_AST' is found, keep all columns up to and including 'POT_AST'
            df_game_logs = df_game_logs.iloc[:, :pot_ast_index + 1]
        except ValueError:
            # If 'POT_AST' is not found, do nothing (pass)
            pass


        # Function to format columns as percentages with no decimals
        def format_percentage(x):
            try:
                # Attempt to format as percentage
                return f'{float(x):.0%}' if pd.notnull(x) and str(x).replace(".", "", 1).isdigit() else x
            except (ValueError, TypeError):
                # If an error occurs, leave the value as is
                return x

          # Merge REB and REB_CHANCES into a new column 'REB'
        df_game_logs['REB'] = df_game_logs['REB'].astype(str) + ' (' + df_game_logs['REB_CHANCES'].astype(str) + ')'

        # Merge AST and POT_AST into a new column 'AST'
        df_game_logs['AST'] = df_game_logs['AST'].astype(str) + ' (' + df_game_logs['POT_AST'].astype(str) + ')'

        # Drop the original REB_CHANCES and POT_AST columns
        df_game_logs.drop(['REB_CHANCES', 'POT_AST'], axis=1, inplace=True)

        # Rearrange columns to have 'WL' first, followed by 'MIN', 'PTS', 'REB', and 'AST'
        column_order = ['GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 'REB', 'AST'] + [col for col in df_game_logs.columns if
                                                                                     col not in ['GAME_DATE', 'MATCHUP', 'WL', 'MIN',
                                                                                                 'PTS', 'REB', 'AST']]
        # Reorder columns
        df_game_logs = df_game_logs[column_order]

        # Rename columns
        df_game_logs.rename(columns={'PLAYER_NAME': 'Name', 'TEAM_ABBREVIATION': 'Team', 'REB': 'REB (Potentials)', 'AST': 'AST (Potentials)'}, inplace=True)

        # Format percentage columns conditionally
        percentage_columns = ['FG_PCT', 'FG3_PCT', 'FT_PCT']
        for col in percentage_columns:
            df_game_logs[col] = df_game_logs[col].apply(format_percentage)

        # Display the modified DataFrame as a Streamlit table
        st.dataframe(df_game_logs, use_container_width=True, hide_index=True)
        
####BOX SCORE####
        st.write(f"<div style='text-align: left;'><h3><b>Box Score Generator</b></h3></div>", unsafe_allow_html=True)
        # Change the selectbox options to use matchups from df_selection_table
        selected_matchup_index = st.selectbox('Select Matchup and Game Date (based on Selected Player above)', df_game_logs.index, format_func=lambda i: f"{df_game_logs.loc[i, 'MATCHUP']} - {df_game_logs.loc[i, 'GAME_DATE']}")

        # Extract the selected MATCHUP and GAME_ID from the original DataFrame
        selected_matchup = df.loc[selected_matchup_index, 'MATCHUP']
        selected_game_id = df.loc[selected_matchup_index, 'GAME_ID']

        # Filter the DataFrame based on the selected matchup to get the corresponding game data
        df_matchup = df[df['GAME_ID'] == selected_game_id].copy()

        # Ensure 'GAME_DATE' is in datetime format
        df_matchup['GAME_DATE'] = pd.to_datetime(df_matchup['GAME_DATE'])

        # Sort the DataFrame by 'GAME_DATE' in descending order
        df_matchup = df_matchup.sort_values(by='GAME_DATE', ascending=False)

        # Extract only the date part from the 'GAME_DATE' column
        df_matchup['GAME_DATE'] = df_matchup['GAME_DATE'].dt.date

        # List of columns to hide
        columns_to_hide_matchup = ['PLAYER_ID', 'GAME_DATE', 'NICKNAME', 'PFD', 'BLKA', 'TEAM_NAME', 'TEAM_ID', 'GAME_ID', 'MATCHUP', 'SEASON_YEAR']

        # Drop the columns to hide
        df_matchup.drop(columns=columns_to_hide_matchup, inplace=True)

        try:
            td3_index_matchup = df_matchup.columns.get_loc('POT_AST')
            df_matchup = df_matchup.iloc[:, :td3_index_matchup + 1]
        except ValueError:
            pass

        # Merge REB and REB_CHANCES into a new column 'REB'
        df_matchup['REB'] = df_matchup['REB'].astype(str) + ' (' + df_matchup['REB_CHANCES'].astype(str) + ')'

        # Merge AST and POT_AST into a new column 'AST'
        df_matchup['AST'] = df_matchup['AST'].astype(str) + ' (' + df_matchup['POT_AST'].astype(str) + ')'

        # Drop the original REB_CHANCES and POT_AST columns
        df_matchup.drop(['REB_CHANCES', 'POT_AST'], axis=1, inplace=True)

        # Rename columns
        df_matchup.rename(columns={'PLAYER_NAME': 'Name', 'TEAM_ABBREVIATION': 'Team'}, inplace=True)

        # Reorder columns, moving 'Name', 'Team', 'WL', 'MIN', 'PTS', 'REB', and 'AST' to the desired order
        column_order = ['Name', 'Team', 'WL', 'MIN', 'PTS', 'REB', 'AST'] + [col for col in df_matchup.columns if
                                                                             col not in ['Name', 'Team', 'WL', 'MIN', 'PTS', 'REB', 'AST']]
        df_matchup = df_matchup[column_order]

        # Convert 'MIN' column to integers
        df_matchup['MIN'] = df_matchup['MIN'].astype(int)

        # Sort the DataFrame by 'Team' and 'MIN' in descending order
        df_matchup = df_matchup.sort_values(by=['Team', 'MIN'], ascending=[False, False])

        # Rename columns
        df_matchup.rename(columns={'PLAYER_NAME': 'Name', 'TEAM_ABBREVIATION': 'Team', 'REB': 'REB (Potentials)', 'AST': 'AST (Potentials)'}, inplace=True)


        # Format percentage columns conditionally
        percentage_columns = ['FG_PCT', 'FG3_PCT', 'FT_PCT']
        for col in percentage_columns:
            df_matchup[col] = df_matchup[col].apply(format_percentage)

        # Display the modified DataFrame for the selected matchup as a Streamlit table
        st.dataframe(df_matchup, use_container_width=True, hide_index=True)


##DVP###
with tab3:
    # Extract the last 3 letters from the "MATCHUP" column to create a list of opponents
    opponents = df["MATCHUP"].str[-3:].unique()

    # Assuming opponents is a list that may contain empty or blank values
    opponents = [opponent for opponent in opponents if opponent.strip() != ""]

    # Sort the remaining options alphabetically
    opponents = sorted(opponents)

    @st.cache_data
    def fetch_players_data():
        # Fetch the "players" sheet from the Google Sheet
        players_sheet = gc.open("NBA Scores scrapper").worksheet("players")
        players_data = players_sheet.get_all_values()
        players_df = pd.DataFrame(players_data[1:], columns=players_data[0])
        return players_df

    @st.cache_data
    def filter_data(selected_opponent, selected_position, df, players_df):
        # Filter the data based on the selected opponent
        filtered_data = df[df["MATCHUP"].str[-3:] == selected_opponent]

        if selected_position != "All":
            # Match player names with the selected position
            matching_players = players_df[players_df["Pos"].str.contains(selected_position)]
            player_names = matching_players["Player"].tolist()

            # Filter by matching player names
            filtered_data = filtered_data[filtered_data["PLAYER_NAME"].isin(player_names)]

        return filtered_data

    players_df = fetch_players_data()

    # Check if the "Pos" column exists in players_df
    if "Pos" in players_df.columns:
        # Extract the positions from the "Pos" column
        positions = players_df["Pos"].str.split(', ').explode().unique()

    col1, col2,col3,col4 = st.columns(4)

    with col1:
        # Create a position filter for both player and opponent stats
        if "Pos" in players_df.columns:
            positions = ["Guards", "Wings", "Bigs"]
            selected_position = st.radio("Position", ["All"] + positions)

        else:
            selected_position = "All"
            
    with col3:

        # Create opp filter with alphabetical sorting and no blanks
        selected_opponent = st.selectbox(
            "Team",
            options=opponents,
            key="opponent_selection"
        )

    # Add buttons for selecting the number of last games in one row
    with col2:
        last_n_games_options = [3, 5, 10, 20, "Season"]
        selected_last_n_games = st.radio("Game Span", last_n_games_options, index=1)
    with col4:
         # Add a slider for minimum minutes threshold
        min_minutes_threshold = st.slider("Minutes Threshold", min_value=0, max_value=48, value=25)

    # Filter the data for the 2023-24 season
    df_2023_24 = df[df["SEASON_YEAR"] == "2023-24"]
    filtered_data = filter_data(selected_opponent, selected_position, df_2023_24, players_df)

    # Define the desired column order
    desired_column_order = [
        "TEAM_ABBREVIATION","PLAYER_NAME","GAME_DATE","PTS","REB","AST","MIN","FGM","FGA","FG3M","FG3A"]

    # Ensure "GAME_DATE" is in datetime format
    filtered_data["GAME_DATE"] = pd.to_datetime(filtered_data["GAME_DATE"])
    filtered_data["GAME_DATE"] = filtered_data["GAME_DATE"].dt.date

    # Reorder the columns to match the desired order
    filtered_data = filtered_data[desired_column_order]

    # Rename the "PTS" column to "Points"
    filtered_data = filtered_data.rename(columns={"TEAM_ABBREVIATION": "TEAM", "PLAYER_NAME": "NAME"})

    filtered_data["MIN"] = filtered_data["MIN"].str.replace('DNP', '0').astype(float)
    filtered_data = filtered_data[filtered_data["MIN"] >= min_minutes_threshold]

    # Group the initial data by "PLAYER_NAME"
    player_groups = df_2023_24.groupby("PLAYER_NAME")

    # List of columns for which you want to calculate means
    integer_columns = ["MIN", "PTS", "REB", "AST", "FGM", "FGA", "FG3M", "FG3A"]

    # Create an empty DataFrame to store the means
    player_means = pd.DataFrame()

    # Calculate the means for each player for each column
    for column in integer_columns:
        # Convert the column to a numeric type (float)
        df_2023_24[column] = pd.to_numeric(df_2023_24[column], errors='coerce').fillna(0)
        player_means[f"{column}_Mean"] = player_groups[column].mean().round(1)

    # Iterate through the rows of the filtered_data DataFrame
    for index, row in filtered_data.iterrows():
        # Iterate through the columns of interest
        for col in integer_columns:
            # Calculate the mean for the specific player and column
            player_mean = player_groups[col].mean().get(row["NAME"], 0)

            # Round the mean to 1 decimal place
            player_mean = round(player_mean, 1)

            # Convert the cell value to a numeric type
            cell_value = pd.to_numeric(row[col])

            # Create a new column for each stat vs. average pair
            filtered_data.loc[index, f"{col} (vs Average)"] = f"{int(cell_value)} ({player_mean})"

    # Drop the original columns from the DataFrame
    filtered_data.drop(columns=integer_columns, inplace=True)

    # Sort the filtered_data DataFrame by 'GAME_DATE' in descending order
    filtered_data = filtered_data.sort_values(by='GAME_DATE', ascending=False)

    # DVP TABLE
    # Display the radio buttons in one row
    st.write("<style>div.row-widget.stRadio > div{flex-direction:row; margin-bottom: -20px;}</style>", unsafe_allow_html=True)

    @st.cache_data
    def calculate_opponent_stats(opponent, selected_position=None, last_n_games=None):
        opponent_data = df_2023_24[df_2023_24["MATCHUP"].str.endswith(opponent)]

        if last_n_games and last_n_games != "Season":
            # Filter opponent_data based on the last N unique game dates
            unique_game_dates = opponent_data['GAME_DATE'].unique()
            last_n_dates = unique_game_dates[-last_n_games:]
            opponent_data = opponent_data[opponent_data['GAME_DATE'].isin(last_n_dates)]

        if selected_position and selected_position != "All":
            # If a position filter is applied, calculate the normalized statistics based on the selected position
            filtered_opponent_data = opponent_data[opponent_data["PLAYER_NAME"].isin(player_names)]
            total_minutes_played = filtered_opponent_data["MIN"].sum()
            if total_minutes_played == 0:
                return None

            opponent_stats = filtered_opponent_data[["PTS", "REB", "AST", "FGM", "FGA", "FG3M", "FG3A"]].sum()
            opponent_normalized_stats = opponent_stats / total_minutes_played
            opponent_normalized_stats = round(opponent_normalized_stats, 2)

            opponent_stats_df = pd.DataFrame({
                "Team": [opponent],
                "PTS/Min": [opponent_normalized_stats["PTS"]],
                "REB/Min": [opponent_normalized_stats["REB"]],
                "AST/Min": [opponent_normalized_stats["AST"]],
                "FGM/Min": [opponent_normalized_stats["FGM"]],
                "FGA/Min": [opponent_normalized_stats["FGA"]],
                "FG3M/Min": [opponent_normalized_stats["FG3M"]],
                "FG3A/Min": [opponent_normalized_stats["FG3A"]],
            })

            return opponent_stats_df

        else:
            # If no position filter or "All" is selected, calculate the traditional average statistics
            opponent_stats = opponent_data[["PTS", "REB", "AST", "FGM", "FGA", "FG3M", "FG3A"]].sum()

            unique_game_ids = set()
            for index, row in opponent_data.iterrows():
                game_id = row["GAME_ID"]
                if game_id not in unique_game_ids:
                    unique_game_ids.add(game_id)
                else:
                    continue

            num_unique_game_ids = len(unique_game_ids)
            opponent_avg_stats = opponent_stats / num_unique_game_ids
            opponent_avg_stats = round(opponent_avg_stats, 1)

            opponent_stats_df = pd.DataFrame({
                "Team": [opponent],
                "PTS": [opponent_avg_stats["PTS"]],
                "REB": [opponent_avg_stats["REB"]],
                "AST": [opponent_avg_stats["AST"]],
                "FGM": [opponent_avg_stats["FGM"]],
                "FGA": [opponent_avg_stats["FGA"]],
                "FG3M": [opponent_avg_stats["FG3M"]],
                "FG3A": [opponent_avg_stats["FG3A"]],
            })

            return opponent_stats_df

    # Calculate opponent statistics
    opponent_stats_dfs = []

    if selected_position and selected_position != "All":
        # If a position is selected, fetch the player names for the selected position
        matching_players = players_df[players_df["Pos"].str.contains(selected_position)]
        player_names = matching_players["Player"].tolist()

    for opponent in opponents:
        opponent_stats_df = calculate_opponent_stats(opponent, selected_position, selected_last_n_games)
        if opponent_stats_df is not None:
            opponent_stats_dfs.append(opponent_stats_df)

    # Concatenate all opponent DataFrames into one
    opponent_stats_df = pd.concat(opponent_stats_dfs, ignore_index=True)

    # Set "Team" as the index
    opponent_stats_df = opponent_stats_df.set_index("Team")
    opponent_stats_df = opponent_stats_df.sort_values(by="Team")

    # Color code the statistic columns
    def color_code_columns(col):
        if col.name not in ["Team"]:
            top5 = col.nlargest(5).index
            bottom5 = col.nsmallest(5).index
            return ['background-color: lightgreen' if idx in top5 else 'background-color: lightcoral' if idx in bottom5 else '' for idx in col.index]
        else:
            return [''] * len(col)

    if selected_position and selected_position != "All":
        # If a position filter is applied, display the normalized statistics
        opponent_stats_df = opponent_stats_df.style.format("{:.2f}").apply(color_code_columns, axis=0)
    else:
        # If no position filter or "All" is selected, display the traditional average statistics
        opponent_stats_df = opponent_stats_df.style.format("{:.1f}").apply(color_code_columns, axis=0)

    col1,col2=st.columns(2)
    with col1:
        if selected_position and selected_position != "All":
            # If a position is selected, display the normalized statistics
            st.write(f"<div style='text-align: left;'><h3><b>DVP ({selected_position})</b></h3></div>", unsafe_allow_html=True)
        else:
            # If no position filter or "All" is selected, display the traditional average statistics
            st.write("<div style='text-align: left;'><h3><b>Opponent Stats</b></h3></div>", unsafe_allow_html=True)
        # Display the styled DataFrame
        st.dataframe(opponent_stats_df, use_container_width=True)

    with col2:
        # Display the DataFrame with player stats and means, keeping the column headers
        st.write(f"<div style='text-align: left;'><h3><b>{selected_position} vs {selected_opponent}</b></h3></div>", unsafe_allow_html=True)
        st.dataframe(filtered_data, use_container_width=True, hide_index=True)


######team dashboard######
with tab4:
    col1, col2, col3, col4,col5 = st.columns(5)

    with col1:
        # Add the "Select Team" dropdown filter with alphabetical sorting and a default value
        selected_team = st.selectbox(
            "Team",
            options=sorted(df["TEAM_ABBREVIATION"].unique()),  # Sort the options alphabetically
            key="team_selection",
            index=sorted(df["TEAM_ABBREVIATION"].unique()).index("ATL")  # Index of "ATL" in the sorted list
        )

    with col2:
        # Multi-select filter for SEASON_YEAR
        selected_years = st.multiselect(
            "Season",
            options=df["SEASON_YEAR"].unique(),
            default=["2023-24"],  # Set "2023-24" as the default season
            key="season_year"
        )

    with col3:
        # "Without" Filter (Default: No Players Selected)
        without_names = st.multiselect(
            "Without",
            options=df[df["TEAM_ABBREVIATION"] == selected_team]["PLAYER_NAME"].unique(),
            key="without_names",
        )

    with col4:
        # "With" Filter (Default: No Players Selected)
        with_names = st.multiselect(
            "With",
            options=df[df["TEAM_ABBREVIATION"] == selected_team]["PLAYER_NAME"].unique(),
            key="with_names",
        )

    with col5:
        # "Opposition Team" Filter (Select teams based on matchup)
        selected_opposition = st.multiselect(
            "Opposition",
            options=sorted(df[df["TEAM_ABBREVIATION"] == selected_team]["MATCHUP"].str[-3:].unique()),
            key="opposition_team"
        )

    # Filter the data based on the selected team and season
    df_filtered = df[(df["TEAM_ABBREVIATION"] == selected_team) & df["SEASON_YEAR"].isin(selected_years)]

    if selected_opposition:
        df_filtered = df_filtered[df_filtered["MATCHUP"].str[-3:].isin(selected_opposition)]        
        
# Exclude games where the selected player played
    if without_names:
        df_filtered = df_filtered[~df_filtered["GAME_ID"].isin(df[df["PLAYER_NAME"].isin(without_names)]["GAME_ID"])]

# Filter the data based on the selected players in the "With" filter
    if with_names:
        # Get the list of game IDs where all selected players played
        games_with_all_selected_players = df[df["PLAYER_NAME"].isin(with_names)].groupby("GAME_ID").filter(lambda x: len(x) == len(with_names))["GAME_ID"].unique()

        # Filter the DataFrame to include only the games where all selected players played
        df_filtered = df_filtered[df_filtered["GAME_ID"].isin(games_with_all_selected_players)]
    
# Filter the data based on checkbox selections
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        win = st.checkbox("Win", key="win_checkbox")
    with col2:
        loss = st.checkbox("Loss", key="loss_checkbox")
    with col3:
        home = st.checkbox("Home", key="home_checkbox")
    with col4:
        away = st.checkbox("Away", key="away_checkbox")

    # Filter the data based on checkbox selections
    if win:
        df_filtered = df_filtered[df_filtered["WL"] == "W"]

    if loss:
        df_filtered = df_filtered[df_filtered["WL"] == "L"]

    if home:
        df_filtered = df_filtered[~df_filtered["MATCHUP"].str.contains("@")]

    if away:
        df_filtered = df_filtered[df_filtered["MATCHUP"].str.contains("@")]        

    # Convert numeric columns to numeric
    numeric_columns = ["MIN", "PTS", "REB", "AST", "FGM", "FGA", "FG3M", "FG3A"]
    df_filtered[numeric_columns] = df_filtered[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Calculate player averages for the filtered data
    @st.cache_data
    def calculate_player_averages(df):
        player_averages = df.groupby("PLAYER_NAME")[numeric_columns].mean().round(1)
        return player_averages

    player_averages = calculate_player_averages(df_filtered)

    # Create a flag to check if filter is active
    filter_without_names = without_names
    filter_with_names = with_names
    filter_selected_opposition = selected_opposition
    filter_win = win
    filter_loss = loss
    filter_home = home
    filter_away = away

    player_averages_filtered = None  # Define it before the if block

    if filter_without_names or filter_with_names or filter_selected_opposition or filter_win or filter_loss or filter_home or filter_away:
        # Filter the data based on the selected team and season
        df_filtered = df[(df["TEAM_ABBREVIATION"] == selected_team) & df["SEASON_YEAR"].isin(selected_years)]

        # Convert numeric columns to numeric
        df_filtered[numeric_columns] = df_filtered[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Calculate player averages for the filtered data
        player_averages_filtered = calculate_player_averages(df_filtered)

    # Display player averages for the selected team
    if not player_averages.empty:
        st.write("<div><h3><b>Situational Player Stats (vs Avg)</b></h3></div>", unsafe_allow_html=True)

        # Check if the filters are active and player_averages_filtered is not empty
        if (filter_without_names or filter_with_names or filter_selected_opposition or filter_win or filter_loss or filter_home or filter_away) and player_averages_filtered is not None and not player_averages_filtered.empty:
            for column in numeric_columns:
                player_averages[f"{column} "] = player_averages[column]
                player_averages[f"{column} (Avg)"] = player_averages_filtered[column]

            # Remove columns 2 to 9 from the player_averages DataFrame
            player_averages = player_averages.drop(player_averages.columns[0:8], axis=1)
        else:
            # Handle the scenario when filters are not active or player_averages_filtered is empty
            try:
                if player_averages_filtered is not None and not player_averages_filtered.empty:
                    for column in numeric_columns:
                        player_averages[f"{column} "] = player_averages[column]
                        player_averages[f"{column} (Avg)"] = player_averages_filtered[column]
            except ValueError:
                # If there's a ValueError, set player_averages to an empty DataFrame
                player_averages = pd.DataFrame(columns=[f"{col} (Avg)" for col in numeric_columns] + [f"{col} (Filtered Avg)" for col in numeric_columns])
                player_averages.index.name = "PLAYER_NAME"

        # Display the sorted DataFrame
        st.write(player_averages)

#####trending

with tab5:
    col1, col2, col3, col4 = st.columns(4)
    
    with col2:
        # Create a dropdown to select the stat column
        selected_column = st.selectbox("Select Stat:", columns_to_mean)

    with col3:
        # Add a slider for setting the minimum average criteria
        min_avg_criteria = st.slider(f"Minimum {selected_column} Avg", min_value=0, max_value=50, value=0)

    with col4:
        # Create buttons for different game spans
        selected_span = st.radio("Game Span:", ["Last 3", "Last 5", "Last 10"])

    with col1:
        selected_team = st.selectbox("Team:", sorted(['All'] + list(df['TEAM_ABBREVIATION'].unique())))

    # Filter the data for the 2023/2024 season
    df_2023_2024 = df[df["SEASON_YEAR"] == "2023-24"].sort_values(by="GAME_DATE", ascending=False)

    # Convert selected columns to numeric and fill missing values with zeros
    df_2023_2024[columns_to_mean] = df_2023_2024[columns_to_mean].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Calculate the mean for the specified columns grouped by player name
    mean_values = df_2023_2024.groupby("PLAYER_NAME")[columns_to_mean].mean().reset_index().round(1)

    # Map the user's selection to the corresponding number of games
    game_span_mapping = {"Last 3": 3, "Last 5": 5, "Last 10": 10}

    # Calculate mean stats for the selected game span
    selected_span_games = game_span_mapping[selected_span]

    # Filter the DataFrame to keep only the selected number of games for each player
    df_selected_span = df_2023_2024.groupby("PLAYER_NAME").head(selected_span_games)

    # Calculate the mean for the specified columns grouped by player name
    mean_stats = df_selected_span.groupby("PLAYER_NAME")[columns_to_mean].mean().reset_index().round(1)

    # Convert selected columns to numeric and fill missing values with zeros
    df_2023_2024[columns_to_mean] = df_2023_2024[columns_to_mean].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Calculate the season averages
    season_avg = df_2023_2024.groupby("PLAYER_NAME")[columns_to_mean].mean().reset_index().round(1)

    # Calculate the game span averages
    game_span_avg = df_2023_2024.groupby("PLAYER_NAME").head(selected_span_games).groupby("PLAYER_NAME")[columns_to_mean].mean().reset_index().round(1)

    # Merge the two DataFrames on the player name
    comparison_df = pd.merge(season_avg, game_span_avg, on="PLAYER_NAME", suffixes=('_Season', f'_Last_{selected_span}'))

    # Calculate the percentage difference between season and game span averages for the selected column
    selected_column_diff = f'{selected_column}_Diff%'
    comparison_df[selected_column_diff] = ((comparison_df[f'{selected_column}_Last_{selected_span}'] - comparison_df[f'{selected_column}_Season']) / comparison_df[f'{selected_column}_Season']) * 100
    comparison_df[selected_column_diff] = comparison_df[selected_column_diff].apply(lambda x: int(x) if x.is_integer() else round(x, 1))

    col1, col2 = st.columns(2)
    last_n_column_name = f'{selected_column}_Last_{selected_span}'

    # Sort the DataFrame by 'GAME_DATE' in descending order
    df_sorted = df.sort_values(by='GAME_DATE', ascending=False)

    # Group by 'PLAYER_NAME' and select the first row for each player (most recent game)
    most_recent_game_data = df_sorted.groupby('PLAYER_NAME').first().reset_index()

    # Merge the most recent game data's team abbreviation column from the original DataFrame into comparison_df
    comparison_df = pd.merge(comparison_df, most_recent_game_data[['PLAYER_NAME', 'TEAM_ABBREVIATION']], on="PLAYER_NAME")

    # Filter players with % differences based on the selected team abbreviation
    positive_diff_df = comparison_df[comparison_df[selected_column_diff] > 0].rename(columns={'PLAYER_NAME': 'Player', f'{selected_column}_Season': 'Average', last_n_column_name: selected_span, selected_column_diff: 'Differential %'})
    negative_diff_df = comparison_df[comparison_df[selected_column_diff] < 0].rename(columns={'PLAYER_NAME': 'Player', f'{selected_column}_Season': 'Average', last_n_column_name: selected_span, selected_column_diff: 'Differential %'})

    if selected_team != 'All':
        positive_diff_df = positive_diff_df[positive_diff_df['TEAM_ABBREVIATION'] == selected_team]
        negative_diff_df = negative_diff_df[negative_diff_df['TEAM_ABBREVIATION'] == selected_team]

    # Filter players based on the minimum average criteria
    positive_diff_df_filtered = positive_diff_df[positive_diff_df['Average'] >= min_avg_criteria].groupby('Player').first().reset_index()
    negative_diff_df_filtered = negative_diff_df[negative_diff_df['Average'] >= min_avg_criteria].groupby('Player').first().reset_index()

    with col1:
        st.write(f"<h4><b>🔥 Hot Players</b></h4>", unsafe_allow_html=True)
        st.dataframe(positive_diff_df_filtered[['Player', selected_span, 'Average', 'Differential %']], hide_index=True)

    with col2:
        st.write(f"<h4><b>🥶 Cold Players</b></h4>", unsafe_allow_html=True)
        st.dataframe(negative_diff_df_filtered[['Player', selected_span, 'Average', 'Differential %']], hide_index=True)

####playtypes x shooting zones
with tab6:
    @st.cache_data
    def fetch_playtype_data():
        worksheet = gc.open("NBA Scores scrapper").worksheet("playtype")
        data = worksheet.get_all_values()
        return pd.DataFrame(data[1:], columns=data[0])

    @st.cache_data
    def fetch_def_playtype_data():
        def_worksheet = gc.open("NBA Scores scrapper").worksheet("defplaytype")
        def_data = def_worksheet.get_all_values()
        def_playtype_df = pd.DataFrame(def_data[1:], columns=def_data[0])
        def_playtype_df['PPP'] = pd.to_numeric(def_playtype_df['PPP'], errors='coerce')
        return def_playtype_df

    @st.cache_data
    def team_opp_shooting():
        def_worksheet = gc.open("NBA Scores scrapper").worksheet("team opp shooting")
        def_data = def_worksheet.get_all_values()
        df = pd.DataFrame(def_data[1:], columns=def_data[0])
        df.columns = df.columns.str.replace('OPP_', '')
        # Drop the first column and columns U to W
        df = df.drop(df.columns[[0] + list(range(19, 22)) + list(range(12, 18))], axis=1)
        
        # Drop columns containing 'FG_PCT' in their column names
        df = df.loc[:, ~df.columns.str.contains('FG_PCT')]
        
        # Convert columns after 'TEAM_NAME' to numeric
        numeric_columns = df.columns[1:]  # Exclude the first column 'TEAM_NAME'
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        
        return df

    @st.cache_data
    def load_player_shooting():
        player_worksheet = gc.open("NBA Scores scrapper").worksheet("player shooting")
        player_data = player_worksheet.get_all_values()
        df = pd.DataFrame(player_data[1:], columns=player_data[0])
        # Convert FGA columns to numeric
        fga_columns = [col for col in df.columns if 'FGA' in col]
        df[fga_columns] = df[fga_columns].apply(pd.to_numeric, errors='coerce')
        return df

    st.title('Defense vs Play Type (points per possession)')
    col1, col2 = st.columns([0.75, 0.25])
    ########play types######
    with col2:
        selected_player = st.selectbox("Player", fetch_playtype_data()['PLAYER_NAME'].unique())
        playtype_df = fetch_playtype_data()

        # Multiply all "POSS_PCT" numbers by 100
        playtype_df[['POSS_PCT', 'PERCENTILE']] = playtype_df[['POSS_PCT', 'PERCENTILE']].apply(pd.to_numeric, errors='coerce') * 100

        # Pivot the table for PERCENTILES/POSS PCT
        combined_pivot = playtype_df.pivot_table(index='PLAYER_NAME', columns='PLAY_TYPE', values=['PTS', 'POSS_PCT', 'PERCENTILE'], aggfunc='first')

        if selected_player:
            # Filter the playtype_df for the selected player
            player_data = playtype_df[playtype_df['PLAYER_NAME'] == selected_player]

            # Add a new column 'POS_PCT' by subtracting 'POSS_PCT' from 100
            player_data['POS_PCT'] = 100 - player_data['POSS_PCT'].astype(float)

            # Define a function to handle multiple rows for a certain play type
            def add_team_abbreviation(play_type_group):
                if len(play_type_group) > 1:
                    play_type_group['PLAY_TYPE'] = play_type_group['PLAY_TYPE'] + ' (' + play_type_group['TEAM_ABBREVIATION'] + ')'
                return play_type_group

            # Apply the function to handle multiple rows for a certain play type
            player_data = player_data.groupby('PLAY_TYPE').apply(add_team_abbreviation)

            # Display a table for the selected player
            st.dataframe(player_data[['PLAY_TYPE', 'PTS', 'POSS_PCT', 'PERCENTILE']].rename(columns={'PLAY_TYPE': 'Play Type', 'PTS': 'Points','POSS_PCT': '%'}), hide_index=True)
    with col1:
        def_playtype_df = fetch_def_playtype_data()

        # Group PPP by Team and Play Type
        def_playtype_pivot = def_playtype_df.pivot_table(index='TEAM_NAME', columns='PLAY_TYPE', values='PPP', aggfunc='first')

        # Color code the statistic columns
        def color_code_columns(col):
            if col.name != "TEAM_NAME":
                top5 = col.nlargest(5).index
                bottom5 = col.nsmallest(5).index
                return ['background-color: lightgreen' if idx in top5 else 'background-color: lightcoral' if idx in bottom5 else '' for idx in col.index]
            else:
                return [''] * len(col)

        # Apply the style to the DataFrame, including the first column
        styled_def_playtype_pivot = def_playtype_pivot.style.apply(color_code_columns, axis=0, subset=pd.IndexSlice[:, :])
        # Display the styled DataFrame using st.dataframe
        st.dataframe(styled_def_playtype_pivot.format("{:.2f}"))

    st.title('Defense vs Shooting Zones (FGA allowed)')
    col1,col2=st.columns([0.75,0.25])

    ##########shooting zones#######
    with col1:
        team_opp_shooting_df = team_opp_shooting()

        # Drop the FGM columns
        team_opp_shooting_df = team_opp_shooting_df.drop(columns=team_opp_shooting_df.filter(like='FGM').columns)

        # Define the color coding function
        def color_code(col):
            if col.name != "TEAM_NAME":
                top5 = col.nlargest(5).index
                bottom5 = col.nsmallest(5).index
                return ['background-color: lightgreen' if idx in top5 else 'background-color: lightcoral' if idx in bottom5 else '' for idx in col.index]
            else:
                return [''] * len(col)

        # Set 'TEAM_NAME' as the index column
        team_opp_shooting_df.set_index('TEAM_NAME', inplace=True)

        # Rename the columns in the DataFrame and replace underscores with spaces
        team_opp_shooting_df.columns = team_opp_shooting_df.columns.str.replace('FGA_', '').str.replace('_', ' ')

        # Apply the style to the DataFrame
        styled_team_opp_shooting_df = team_opp_shooting_df.style.apply(color_code, axis=0, subset=pd.IndexSlice[:, :])

        # Convert numeric columns to the desired format
        numeric_columns = team_opp_shooting_df.select_dtypes(include='number').columns
        styled_team_opp_shooting_df = styled_team_opp_shooting_df.format({col: '{:.1f}' for col in numeric_columns})

        # Display the styled DataFrame without the index column
        st.write(styled_team_opp_shooting_df, hide_index=True)

    with col2:
        # Load player shooting data
        player_shooting_df = load_player_shooting()

        # Filter the DataFrame based on selected player
        filtered_df = player_shooting_df[player_shooting_df['PLAYER'] == selected_player]

        # Drop the specified columns
        filtered_df = filtered_df.drop(columns=['FGA (LEFT CORNER 3)', 'FGA (RIGHT CORNER 3)'])

        # Extract FGA values for the selected player
        fga_values = filtered_df.filter(like='FGA').rename(columns=lambda x: 'FGA' if x == '0' else x.replace('FGA', '').replace('(', '').replace(')', '')).squeeze()
        fga_values.name = 'FGA'

        # FGA %
        total_fga = fga_values.sum()
        fga_percentages = (fga_values / total_fga) * 100
        fga_percentages.name = '%'

        # Merge FGA values and FGA percentages into a single DataFrame and round
        fga_data = pd.concat([fga_values, fga_percentages], axis=1)
        fga_data['%'] = fga_data['%'].round(0)

        # Label the first column as "Shooting Zone"
        fga_data.index.name = 'Shooting Zone'

        st.write(fga_data)


