import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image

background_image = Image.open('Kabaddi.png')

# Function to plot the cumulative points chart
def team_points(df):
    team1_points = df['team1 Points'].sum()
    team2_points = df['Team2 Points'].sum()
    return team1_points, team2_points

def plot_cumulative_points(df):
    cumulative_team1_points = df['team1 Points'].cumsum()
    cumulative_team2_points = df['Team2 Points'].cumsum()
    points = list(range(1, len(df) + 1))

    trace_team1 = go.Scatter(
        x=points,
        y=cumulative_team1_points,
        mode='lines+markers',
        name='Ahmednagar Cumulative Points'
        
    )

    trace_team2 = go.Scatter(
        x=points,
        y=cumulative_team2_points,
        mode='lines+markers',
        name='Palghar Cumulative Points',
        line=dict(color='orange')  # Change the color to yellow
    )

    layout = go.Layout(
        title='Points scored by Ahmednagar and Palghar',
        xaxis=dict(title='No. of Raids'),
        yaxis=dict(title='Points'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        width = 1200
        
    )

    fig = go.Figure(data=[trace_team1, trace_team2], layout=layout)

    st.plotly_chart(fig)

#  Function to analyze Kabaddi data for Team 1
def analyze_kabaddi_data_team1(df):
    df.loc[df['Tackle Points'] == 'None', 'Tackle Points'] = np.nan
    df.loc[df['Defence player No.'] == 'None','Defence player No.']= np.nan
    df.rename({'Bonus Point scored':'Bonus Points','Points By Raider':'Touch Points'},axis=1)

    team1_total_points = df['team1 Points'].sum()
    team1_bonus_points = df[df['RD'] == 'R']['Bonus Points'].sum()
    team1_touch_points = df[df['RD'] == 'R']['Touch Points'].sum()
    team1_tackle_points = df[df['RD'] == 'D']['Tackle Points'].sum()
    team1_All_Out = df['Team 1 All Out'].sum()
    team1_raids = (df['RD'] == 'R').sum()
    mask_team1 = (df['RD'] == 'R') & ((df['Bonus Points'] == 1) | (df['Touch Points'].between(1, 7)))
    successful_raids_team1 = mask_team1.sum()
    unsuccessful_raids_team1 = team1_raids - successful_raids_team1
    raid_strike_rate1 = (successful_raids_team1 / team1_raids) * 100

    analysis_results = {
        "Unsuccessful Raids": unsuccessful_raids_team1,
        "Successful Raids": successful_raids_team1,
        "All Out": team1_All_Out,
        "Tackle Points": team1_tackle_points,
        "Touch Points" : team1_touch_points,
        "Bonus Points": team1_bonus_points,
        "Total Points": team1_total_points,
        "Total Raids": team1_raids  
    }

    return analysis_results

# Function to analyze Kabaddi data for Team 2
def analyze_kabaddi_data_team2(df):
    df.loc[df['Tackle Points'] == 'None', 'Tackle Points'] = np.nan
    df.loc[df['Defence player No.'] == 'None','Defence player No.']= np.nan
    df.rename({'Bonus Point scored':'Bonus Points','Points By Raider':'Touch Points'},axis=1)
    team2_total_points = df['Team2 Points'].sum()
    team2_bonus_points = df[df['RD'] == 'D']['Bonus Points'].sum()
    team2_touch_points = df[df['RD'] == 'D']['Touch Points'].sum()
    team2_tackle_points = df[df['RD'] == 'R']['Tackle Points'].sum()
    team2_All_Out = df['Team 2 All out'].sum()
    team2_raids = (df['RD'] == 'D').sum()
    mask_team2 = (df['RD'] == 'D') & ((df['Bonus Points'] == 1) | (df['Touch Points'].between(1, 7)))
    successful_raids_team2 = mask_team2.sum()
    unsuccessful_raids_team2 = team2_raids - successful_raids_team2
    raid_strike_rate2 = (successful_raids_team2 / team2_raids) * 100

    analysis_results = {
        "Unsuccessful Raids": unsuccessful_raids_team2,
        "Successful_raids": successful_raids_team2,
        "All Out": team2_All_Out,
        "Tackle Points": team2_tackle_points,
        "Touch Points" : team2_touch_points,
        "Bonus Points": team2_bonus_points,
        "Total Points": team2_total_points,
        "Total Raids": team2_raids
    }

    return analysis_results

def analyze_kabaddi_data_by_half(df):
    # Split the DataFrame into two halves based on the 'Half' column
    df_half1 = df[df['Half'] == 1]
    df_half2 = df[df['Half'] == 2]

    # Analyze data for each half separately for Team 1 and Team 2
    analysis_results_team1_half1 = analyze_kabaddi_data_team1(df_half1)
    analysis_results_team1_half2 = analyze_kabaddi_data_team1(df_half2)
    analysis_results_team2_half1 = analyze_kabaddi_data_team2(df_half1)
    analysis_results_team2_half2 = analyze_kabaddi_data_team2(df_half2)

    return (analysis_results_team1_half1, analysis_results_team1_half2,
            analysis_results_team2_half1, analysis_results_team2_half2)

# Function to draw a tornado chart comparing data between two halves
def draw_tornado_chart_for_half(analysis_results_team1, analysis_results_team2, half):
    categories = list(analysis_results_team1.keys())
    values_team1 = list(analysis_results_team1.values())
    values_team2 = [-value for value in analysis_results_team2.values()]

    fig, ax = plt.subplots(figsize=(16, 7.5))

    index = range(len(categories))

    ax.barh(index, values_team1, color='skyblue', label='Team 1', height=0.4)
    ax.barh(index, values_team2, color='orange', label='Team 2', height=0.4)

    for i, value in enumerate(values_team1):
        ax.text(value, i, str(value), ha='left', va='center', color='black', fontsize=18)  # Adjust font size here

    for i, value in enumerate(values_team2):
        ax.text(value, i, str(-value), ha='right', va='center', color='black', fontsize=18)  # Adjust font size here

    ax.set_yticks(index)
    ax.set_yticklabels(categories, fontsize=16)  # Adjust font size here
    ax.set_xlabel('Points', fontsize=16)  # Adjust font size here
    ax.set_title(f'COMPARISON CHART {half}', fontsize=16)  # Adjust font size here
    ax.legend(fontsize=16)  # Adjust font size here

    st.pyplot(fig)


# Function to draw a tornado chart
# Function to draw a tornado chart
def draw_tornado_chart(analysis_results_team1, analysis_results_team2):
    categories = list(analysis_results_team1.keys())
    values_team1 = list(analysis_results_team1.values())
    values_team2 = [-value for value in analysis_results_team2.values()]  

    fig, ax = plt.subplots(figsize=(16, 7.5))

    index = range(len(categories))

    ax.barh(index, values_team1, color='skyblue', label='Team 1', height=0.4)
    ax.barh(index, values_team2, color='orange', label='Team 2', height=0.4)

    for i, value in enumerate(values_team1):
        ax.text(value, i, str(value), ha='left', va='center', color='black', fontsize=18)  # Adjust font size here

    for i, value in enumerate(values_team2):
        ax.text(value, i, str(-value), ha='right', va='center', color='black', fontsize=18)  # Adjust font size here

    ax.set_yticks(index)
    ax.set_yticklabels(categories, fontsize=16)  # Adjust font size here
    ax.set_xlabel('Points', fontsize=16)  # Adjust font size here
    ax.set_title('COMPARISON CHART', fontsize=16)  # Adjust font size here
    ax.legend(fontsize=16)  # Adjust font size here

    st.pyplot(fig)


# Function to generate Team 1 scorecard
def Team1_Scorecard(df):
    team1_raiders_data = df[df['RD'] == 'R'].groupby('R No.')[['Bonus Points', 'Touch Points', 'Empty Raid', 'Do-or-die Raids']].sum()
    total_raids = df[df['RD'] == 'R']['R No.'].value_counts()
    unsuccessful_raids = df[(df['RD'] == 'R') & (df['Tackle Points'] == 1)].groupby('R No.').size()
    team1_raiders_data_reset = team1_raiders_data.reset_index()
    super_raids = (df[(df['RD'] == 'D') & ((team1_raiders_data_reset['Bonus Points'] + team1_raiders_data_reset['Touch Points']) >= 3)]
                   .groupby('R No.')
                   .size()
                   .astype(int))
    super_tackles = (df[(df['RD'] == 'R') & (df['Tackle Points'] == 2)]
                     .groupby('Defence player No.')
                     .size()
                     .astype(int))
    team1_defense_data = df[df['RD'] == 'D'].groupby('Defence player No.')[['Tackle Points']].sum()
    unsuccesful_tackles = df[(df['RD'] == 'D') & (df['Tackle Points'] == 0)][['2nd Player Out', '3rd Player Out', '4th Player Out','Defence player No.']].stack().value_counts()

    team1_combined_data = pd.concat([team1_raiders_data, team1_defense_data, unsuccessful_raids], axis=1)
    team1_combined_data = team1_combined_data.rename(columns={0: 'Unsuccessful Raids'})
    team1_combined_data['Unsuccessful Tackles'] = 0
    team1_combined_data['Unsuccessful Tackles'] = team1_combined_data['Unsuccessful Tackles'].fillna(0) + unsuccesful_tackles
    team1_combined_data['Total Points'] = team1_combined_data['Bonus Points'].fillna(0) + team1_combined_data['Touch Points'].fillna(0) + team1_combined_data['Tackle Points'].fillna(0)
    team1_combined_data['Total Raids'] = total_raids
    team1_combined_data['Super Raids'] = super_raids
    team1_combined_data['Super Tackles'] = super_tackles

    team1_combined_dict = team1_combined_data.fillna(0).to_dict()
    team1_scorecard = pd.DataFrame(team1_combined_dict)

    if not team1_scorecard.empty:
        team1_scorecard.rename_axis('Jersey No.', inplace=True)

    return team1_scorecard

# Function to generate Team 2 scorecard
def Team2_Scorecard(df):
    team2_raiders_data = df[df['RD'] == 'D'].groupby('R No.')[['Bonus Points', 'Touch Points', 'Empty Raid', 'Do-or-die Raids']].sum()
    total_raids = df[df['RD'] == 'D']['R No.'].value_counts()
    unsuccessful_raids = df[(df['RD'] == 'D') & (df['Tackle Points'] == 1)].groupby('R No.').size()
    team2_raiders_data_reset = team2_raiders_data.reset_index()
    super_raids = (df[(df['RD'] == 'R') & ((team2_raiders_data_reset['Bonus Points'] + team2_raiders_data_reset['Touch Points']) >= 3)]
                   .groupby('R No.')
                   .size()
                   .astype(int))
    super_tackles = (df[(df['RD'] == 'D') & (df['Tackle Points'] == 2)]
                     .groupby('Defence player No.')
                     .size()
                     .astype(int))
    team2_defense_data = df[df['RD'] == 'R'].groupby('Defence player No.')[['Tackle Points']].sum()
    unsuccesful_tackles = df[(df['RD'] == 'R') & (df['Tackle Points'] == 0)][['2nd Player Out', '3rd Player Out', '4th Player Out','Defence player No.']].stack().value_counts()

    team2_combined_data = pd.concat([team2_raiders_data, team2_defense_data, unsuccessful_raids], axis=1)
    team2_combined_data = team2_combined_data.rename(columns={0: 'Unsuccessful Raids'})
    team2_combined_data['Unsuccessful Tackles'] = 0
    team2_combined_data['Unsuccessful Tackles'] = team2_combined_data['Unsuccessful Tackles'].fillna(0) + unsuccesful_tackles
    team2_combined_data['Total Points'] = team2_combined_data['Bonus Points'].fillna(0) + team2_combined_data['Touch Points'].fillna(0) + team2_combined_data['Tackle Points'].fillna(0)
    team2_combined_data['Total Raids'] = total_raids
    team2_combined_data['Super Raids'] = super_raids
    team2_combined_data['Super Tackles'] = super_tackles

    team2_combined_dict = team2_combined_data.fillna(0).to_dict()
    team2_scorecard = pd.DataFrame(team2_combined_dict)

    if not team2_scorecard.empty:
        team2_scorecard.rename_axis('Jersey No.', inplace=True)

    return team2_scorecard

## Team1 Bonus Point Distribution

def Team1_bonus_points_distribution(df):
    # Filter DataFrame for bonus point attempts and bonus points
    bonus_points_df = df[df['RD'] == 'R'][['Bonus Point attempt', 'Bonus Points']]

    # Exclude 'None' values
    bonus_points_df = bonus_points_df[(bonus_points_df['Bonus Point attempt'] != 'None')]

    # Filter bonus points where attempts were made (1) and where points were scored (1)
    successful_bonus_points_df = bonus_points_df[bonus_points_df['Bonus Points'] == 1]

    # Count successful bonus points for each corner
    successful_bonus_points_count = successful_bonus_points_df['Bonus Point attempt'].value_counts()

    # Get total attempts for bonus points for each corner
    total_attempts_per_corner = bonus_points_df['Bonus Point attempt'].value_counts()

    # Calculate unsuccessful attempts for bonus points for each corner
    unsuccessful_attempts_per_corner = total_attempts_per_corner.sub(successful_bonus_points_count, fill_value=0)

    # Define labels and values for the Pie chart
    labels = ['Left Corner', 'Right Corner']
    successful_values = [successful_bonus_points_count.get('LC', 0), successful_bonus_points_count.get('RC', 0)]
    unsuccessful_values = [unsuccessful_attempts_per_corner.get('LC', 0), unsuccessful_attempts_per_corner.get('RC', 0)]

    # Create Donut chart for successful bonus points
    fig_successful_bonus_points = go.Figure(data=[go.Pie(labels=labels, values=successful_values)])
    fig_successful_bonus_points.update_traces(textinfo='label+value', hole=0.5)
    fig_successful_bonus_points.update_layout(title_text="Successful Bonus Points Distribution by Corner")

    # Create Donut chart for unsuccessful bonus points
    fig_unsuccessful_bonus_points = go.Figure(data=[go.Pie(labels=labels, values=unsuccessful_values)])
    fig_unsuccessful_bonus_points.update_traces(textinfo='label+value', hole=0.5)
    fig_unsuccessful_bonus_points.update_layout(title_text="Unsuccessful Bonus Points Distribution by Corner")

    return fig_successful_bonus_points, fig_unsuccessful_bonus_points


## Team2 Bonus Point distribution 

def Team2_bonus_points_distribution(df):
    # Filter DataFrame for bonus point attempts and bonus points
    bonus_points_df = df[df['RD'] == 'D'][['Bonus Point attempt', 'Bonus Points']]

    # Exclude 'None' values
    bonus_points_df = bonus_points_df[(bonus_points_df['Bonus Point attempt'] != 'None')]

    # Filter bonus points where attempts were made (1) and where points were scored (1)
    successful_bonus_points_df = bonus_points_df[bonus_points_df['Bonus Points'] == 1]

    # Count successful bonus points for each corner
    successful_bonus_points_count = successful_bonus_points_df['Bonus Point attempt'].value_counts()

    # Get total attempts for bonus points for each corner
    total_attempts_per_corner = bonus_points_df['Bonus Point attempt'].value_counts()

    # Calculate unsuccessful attempts for bonus points for each corner
    unsuccessful_attempts_per_corner = total_attempts_per_corner.sub(successful_bonus_points_count, fill_value=0)

    # Define labels and values for the Pie chart
    labels = ['Left Corner', 'Right Corner']
    successful_values = [successful_bonus_points_count.get('LC', 0), successful_bonus_points_count.get('RC', 0)]
    unsuccessful_values = [unsuccessful_attempts_per_corner.get('LC', 0), unsuccessful_attempts_per_corner.get('RC', 0)]

    # Create Donut chart for successful bonus points
    fig_successful_bonus_points = go.Figure(data=[go.Pie(labels=labels, values=successful_values)])
    fig_successful_bonus_points.update_traces(textinfo='label+value', hole=0.5)
    fig_successful_bonus_points.update_layout(title_text="Successful Bonus Points Distribution by Corner")

    # Create Donut chart for unsuccessful bonus points
    fig_unsuccessful_bonus_points = go.Figure(data=[go.Pie(labels=labels, values=unsuccessful_values)])
    fig_unsuccessful_bonus_points.update_traces(textinfo='label+value', hole=0.5)
    fig_unsuccessful_bonus_points.update_layout(title_text="Unsuccessful Bonus Points Distribution by Corner")

    return fig_successful_bonus_points, fig_unsuccessful_bonus_points

###------------------------------------------Raider Escape SKill-----------------------------------------------------------###

def Team1_raider_escape(df):
    filtered_df = df[df['RD'] == 'R']  # Filter rows where RD column is 'R'
    raider_escape_counts = filtered_df['Raider Escape'].value_counts()
    labels = raider_escape_counts.index.tolist()
    values = raider_escape_counts.values.tolist()
    
    fig_donut_chart = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5)])
    fig_donut_chart.update_traces(textinfo='label+value')
    fig_donut_chart.update_layout(title_text="Raider Escape")
    
    st.plotly_chart(fig_donut_chart)


def Team2_raider_escape(df):
    filtered_df = df[df['RD'] == 'D']  # Filter rows where RD column is 'R'
    raider_escape_counts = filtered_df['Raider Escape'].value_counts()
    labels = raider_escape_counts.index.tolist()
    values = raider_escape_counts.values.tolist()
    
    fig_donut_chart = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5)])
    fig_donut_chart.update_traces(textinfo='label+value')
    fig_donut_chart.update_layout(title_text="Raider Escape")
    
    st.plotly_chart(fig_donut_chart)




####--------------------------------------------------Video Links-------------------------------------------------------####
from functools import reduce
def Team1_Video_data(df, unique_r_nos, bonus_points_values, touch_points_values, empty_raids, do_or_die_raids):
    # Filter rows where 'RD' column is 'R'
    df_r = df[df['RD'] == 'R']
    
    # Apply the selected filters
    if not unique_r_nos and not bonus_points_values and not touch_points_values and not empty_raids and not do_or_die_raids:
        filtered_df = df_r
    else:
        conditions = []
        if unique_r_nos:
            conditions.append(df_r["R No."].isin(unique_r_nos))
        if bonus_points_values:
            conditions.append(df_r["Bonus Points"].isin(bonus_points_values))
        if touch_points_values:
            conditions.append(df_r["Touch Points"].isin(touch_points_values))
        if empty_raids:
            conditions.append(df_r["Empty Raid"].isin(empty_raids))
        if do_or_die_raids:
            conditions.append(df_r["Do-or-die Raids"].isin(do_or_die_raids))
        
        filtered_df = df_r[reduce(lambda x, y: x & y, conditions)]
    
    return filtered_df

## video Link Team 2

def Team2_Video_data(df, unique_r_nos, bonus_points_values, touch_points_values, empty_raids, do_or_die_raids):
    # Filter rows where 'RD' column is 'R'
    df_r = df[df['RD'] == 'D']
    
    # Apply the selected filters
    if not unique_r_nos and not bonus_points_values and not touch_points_values and not empty_raids and not do_or_die_raids:
        filtered_df2 = df_r
    else:
        conditions = []
        if unique_r_nos:
            conditions.append(df_r["R No."].isin(unique_r_nos))
        if bonus_points_values:
            conditions.append(df_r["Bonus Points"].isin(bonus_points_values))
        if touch_points_values:
            conditions.append(df_r["Touch Points"].isin(touch_points_values))
        if empty_raids:
            conditions.append(df_r["Empty Raid"].isin(empty_raids))
        if do_or_die_raids:
            conditions.append(df_r["Do-or-die Raids"].isin(do_or_die_raids))
        
        filtered_df2 = df_r[reduce(lambda x, y: x & y, conditions)]
    
    return filtered_df2

###-------------------------------------Attacks and Attack Locations-------------------------------------------------------------------------##
def Team1_Attack_and_Attack_Locations(chart_type, df):
    if chart_type == 'Total':
        filtered_attacks = df[df['RD'] == 'R'][['Raider Attack', 'Second Attack Raider']].stack()
        filtered_attacks = filtered_attacks[filtered_attacks != 'None']
        attacks_count = filtered_attacks.value_counts()
        labels = attacks_count.index.tolist()
        values = attacks_count.values.tolist()
        fig_attacks = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig_attacks.update_traces(textinfo='label+value')
        fig_attacks.update_layout(title_text="Total Attacks")
        st.plotly_chart(fig_attacks)

        df_R = df[df['RD'] == 'R']

        # Drop rows with missing values in 'Start1' and 'End1' columns
        df_R = df_R.dropna(subset=['Start1', 'End1'])

        # Combine all start and end coordinates
        all_start_coordinates = df_R['Start1'].str.strip('()').str.split(', ', expand=True).astype(float)
        all_end_coordinates = df_R['End1'].str.strip('()').str.split(', ', expand=True).astype(float)

        # Combine all raider attack names
        all_raider_attack_names = df_R['Raider Attack']
        

       
        # If 'Start2' and 'End2' are not NaN, combine their coordinates and raider attack names as well
        df_R_second_attack = df_R.dropna(subset=['Start2', 'End2'])
        if not df_R_second_attack.empty:
            all_start_coordinates = pd.concat([
                all_start_coordinates,
                df_R_second_attack['Start2'].str.strip('()').str.split(', ', expand=True).astype(float)
            ])
            all_end_coordinates = pd.concat([
                all_end_coordinates,
                df_R_second_attack['End2'].str.strip('()').str.split(', ', expand=True).astype(float)
            ])
            all_raider_attack_names = pd.concat([
                all_raider_attack_names,
                df_R_second_attack['Second Attack Raider']
            ])

        # Create a scatter plot for all start and end coordinates
        start_trace = go.Scatter(
            x=all_start_coordinates[0],
            y=all_start_coordinates[1],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Start',
            text=all_raider_attack_names,
            hoverinfo='text'
        )

        end_trace = go.Scatter(
            x=all_end_coordinates[0],
            y=all_end_coordinates[1],
            mode='markers',
            marker=dict(color='blue', size=10),
            name='End',
            text=all_raider_attack_names,
            hoverinfo='text'
        )

        # Create the figure
        fig  = go.Figure()

        # Add start and end traces to the figure
        fig.add_trace(start_trace)
        fig.add_trace(end_trace)

        # Add lines connecting start and end coordinates
        for i in range(len(all_start_coordinates)):
            fig.add_trace(go.Scatter(
                x=[all_start_coordinates.iloc[i, 0], all_end_coordinates.iloc[i, 0]],
                y=[all_start_coordinates.iloc[i, 1], all_end_coordinates.iloc[i, 1]],
                mode='lines',
                line=dict(color='rgba(255, 255, 255, 0.8)', width=2),
                showlegend=False
            ))

        # Update layout to include the background image
        fig.update_layout(
            images=[dict(
                source=background_image,
                xref="x",
                yref="y",
                x=0,
                y=0,  # Adjust the y-coordinate to align the image correctly
                sizex=470,
                sizey=330,  # Adjust the size as per your image's dimensions
                sizing="stretch",
                opacity=1,
                layer="below"
            )],
            xaxis=dict(range=[0, 470], showgrid=False),  # Adjust x-axis range as per your image's width
            yaxis=dict(range=[330, 0], showgrid=False),  # Adjust y-axis range as per your image's height
            title='Total Attack Locations',
            showlegend=True,
        )

        st.plotly_chart(fig)

    elif chart_type == 'Successful':
        # Filter DataFrame for successful attacks
        filtered_attacks = df[(df['RD'] == 'R')]
        condition_touch_points = filtered_attacks['Touch Points'].between(1, 7)
        condition_raider_attack = filtered_attacks['Raider Attack'].notnull()
        condition_second_attack = filtered_attacks['Second Attack Raider'].notnull()
        condition_successful_raider_attack = condition_touch_points & condition_raider_attack & (filtered_attacks['Second Attack Raider'].isna())
        condition_successful_second_attack = condition_touch_points & condition_raider_attack & condition_second_attack

        # Apply conditions and select raider attack or second attack locations accordingly
        filtered_raider_attacks = filtered_attacks[condition_successful_raider_attack]
        filtered_second_attacks = filtered_attacks[condition_successful_second_attack]

        # Plot successful attacks
        attacks_count = pd.concat([filtered_raider_attacks['Raider Attack'], filtered_second_attacks['Second Attack Raider']]).value_counts()
        labels = attacks_count.index.tolist()
        values = attacks_count.values.tolist()
        fig_attacks = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig_attacks.update_traces(textinfo='label+value')
        fig_attacks.update_layout(title_text="Successful Attacks")
        

        # Plot attack locations for successful raider attacks
        if not filtered_raider_attacks.empty:
            # Combine all start and end coordinates for successful raider attacks
            # Combine all start and end coordinates for successful raider attacks
            all_raider_start_coordinates = filtered_raider_attacks['Start1'].str.strip('()').str.split(', ', expand=True).astype(float)
            all_raider_end_coordinates = filtered_raider_attacks['End1'].str.strip('()').str.split(', ', expand=True).astype(float)

            # Combine all start and end coordinates for successful second attacks
            all_second_start_coordinates = filtered_second_attacks['Start2'].str.strip('()').str.split(', ', expand=True).astype(float)
            all_second_end_coordinates = filtered_second_attacks['End2'].str.strip('()').str.split(', ', expand=True).astype(float)

            # Concatenate raider and second attack coordinates
            all_start_coordinates = pd.concat([all_raider_start_coordinates, all_second_start_coordinates])
            all_end_coordinates = pd.concat([all_raider_end_coordinates, all_second_end_coordinates])
            
            # Concatenate raider and second attack names
            No_of_Players = filtered_attacks['Total Players at Defence']
            all_raider_attack_names = pd.concat([filtered_raider_attacks['Raider Attack'], filtered_second_attacks['Second Attack Raider'],
                                                 No_of_Players])
            
           
    
            # Create a scatter plot for all start and end coordinates
            start_trace = go.Scatter(
                x=all_start_coordinates[0],
                y=all_start_coordinates[1],
                mode='markers',
                marker=dict(color='red', size=10),
                name='Start',
                text=all_raider_attack_names,
                hoverinfo='text'
            )

            end_trace = go.Scatter(
                x=all_end_coordinates[0],
                y=all_end_coordinates[1],
                mode='markers',
                marker=dict(color='blue', size=10),
                name='End',
                text=all_raider_attack_names,
                hoverinfo='text'
            )

            # Create the figure
            fig  = go.Figure()

            # Add start and end traces to the figure
            fig.add_trace(start_trace)
            fig.add_trace(end_trace)

            # Add lines connecting start and end coordinates
            for i in range(len(all_start_coordinates)):
                fig.add_trace(go.Scatter(
                    x=[all_start_coordinates.iloc[i, 0], all_end_coordinates.iloc[i, 0]],
                    y=[all_start_coordinates.iloc[i, 1], all_end_coordinates.iloc[i, 1]],
                    mode='lines',
                    line=dict(color='rgba(255, 255, 255, 0.8)', width=2),
                    showlegend=False
                ))

            # Update layout to include the background image
            fig.update_layout(
                images=[dict(source=background_image,xref="x",yref="y",x=0,y=0,  # Adjust the y-coordinate to align the image correctly
                    sizex=470,sizey=330,  # Adjust the size as per your image's dimensions
                    sizing="stretch",opacity=1,layer="below"
                )],
                xaxis=dict(range=[0, 470], showgrid=False),  # Adjust x-axis range as per your image's width
                yaxis=dict(range=[330, 0], showgrid=False),  # Adjust y-axis range as per your image's height
                title='Successful Attack Locations',
                showlegend=True,
            )
            st.plotly_chart(fig_attacks)
            st.plotly_chart(fig)


    elif chart_type == 'Unsuccessful':
        filtered_attacks = df[(df['RD'] == 'R') & (df['Touch Points'] == 0)][['Raider Attack', 'Second Attack Raider']].stack()
        filtered_attacks = filtered_attacks[filtered_attacks != 'None']
        attacks_count = filtered_attacks.value_counts()
        labels = attacks_count.index.tolist()
        values = attacks_count.values.tolist()
        fig_attacks = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig_attacks.update_traces(textinfo='label+value')
        fig_attacks.update_layout(title_text="Unsuccessful Attacks")
        st.plotly_chart(fig_attacks)

        df_R = df[(df['RD'] == 'R') & (df['Touch Points'] == 0)]

        # Drop rows with missing values in 'Start1' and 'End1' columns
        df_R = df_R.dropna(subset=['Start1', 'End1'])

        # Combine all start and end coordinates
        all_start_coordinates = df_R['Start1'].str.strip('()').str.split(', ', expand=True).astype(float)
        all_end_coordinates = df_R['End1'].str.strip('()').str.split(', ', expand=True).astype(float)

        # Combine all raider attack names
        all_raider_attack_names = df_R['Raider Attack']
       

        # If 'Start2' and 'End2' are not NaN, combine their coordinates and raider attack names as well
        df_R_second_attack = df_R.dropna(subset=['Start2', 'End2'])
        if not df_R_second_attack.empty:
            all_start_coordinates = pd.concat([
                all_start_coordinates,
                df_R_second_attack['Start2'].str.strip('()').str.split(', ', expand=True).astype(float)
            ])
            all_end_coordinates = pd.concat([
                all_end_coordinates,
                df_R_second_attack['End2'].str.strip('()').str.split(', ', expand=True).astype(float)
            ])
            all_raider_attack_names = pd.concat([
                all_raider_attack_names,
                df_R_second_attack['Second Attack Raider']
            ])

        # Create a scatter plot for all start and end coordinates
        start_trace = go.Scatter(
            x=all_start_coordinates[0],
            y=all_start_coordinates[1],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Start',
            text=all_raider_attack_names,
            hoverinfo='text'
        )

        end_trace = go.Scatter(
            x=all_end_coordinates[0],
            y=all_end_coordinates[1],
            mode='markers',
            marker=dict(color='blue', size=10),
            name='End',
            text=all_raider_attack_names,
            hoverinfo='text'
        )

        # Create the figure
        fig  = go.Figure()

        # Add start and end traces to the figure
        fig.add_trace(start_trace)
        fig.add_trace(end_trace)

        # Add lines connecting start and end coordinates
        for i in range(len(all_start_coordinates)):
            fig.add_trace(go.Scatter(
                x=[all_start_coordinates.iloc[i, 0], all_end_coordinates.iloc[i, 0]],
                y=[all_start_coordinates.iloc[i, 1], all_end_coordinates.iloc[i, 1]],
                mode='lines',
                line=dict(color='rgba(255, 255, 255, 0.8)', width=2),
                showlegend=False
            ))

        # Update layout to include the background image
        fig.update_layout(
            images=[dict(
                source=background_image,
                xref="x",
                yref="y",
                x=0,
                y=0,  # Adjust the y-coordinate to align the image correctly
                sizex=470,
                sizey=330,  # Adjust the size as per your image's dimensions
                sizing="stretch",
                opacity=1,
                layer="below"
            )],
            xaxis=dict(range=[0, 470], showgrid=False),  # Adjust x-axis range as per your image's width
            yaxis=dict(range=[330, 0], showgrid=False),  # Adjust y-axis range as per your image's height
            title='Unsuccessful Attack Locations',
            showlegend=True,
        )

        st.plotly_chart(fig)




def Team2_Attack_and_Attack_Locations(chart_type, df):
    if chart_type == 'Total':
        filtered_attacks = df[df['RD'] == 'D'][['Raider Attack', 'Second Attack Raider']].stack()
        filtered_attacks = filtered_attacks[filtered_attacks != 'None']
        attacks_count = filtered_attacks.value_counts()
        labels = attacks_count.index.tolist()
        values = attacks_count.values.tolist()
        fig_attacks = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig_attacks.update_traces(textinfo='label+value')
        fig_attacks.update_layout(title_text="Total Attacks")
        st.plotly_chart(fig_attacks)

        df_D = df[df['RD'] == 'D']

        # Drop rows with missing values in 'Start1' and 'End1' columns
        df_D = df_D.dropna(subset=['Start1', 'End1'])

        # Combine all start and end coordinates
        all_start_coordinates = df_D['Start1'].str.strip('()').str.split(', ', expand=True).astype(float)
        all_end_coordinates = df_D['End1'].str.strip('()').str.split(', ', expand=True).astype(float)

        # Combine all raider attack names
        all_raider_attack_names = df_D['Raider Attack']
        

        # If 'Start2' and 'End2' are not NaN, combine their coordinates and raider attack names as well
        df_D_second_attack = df_D.dropna(subset=['Start2', 'End2'])
        if not df_D_second_attack.empty:
            all_start_coordinates = pd.concat([
                all_start_coordinates,
                df_D_second_attack['Start2'].str.strip('()').str.split(', ', expand=True).astype(float)
            ])
            all_end_coordinates = pd.concat([
                all_end_coordinates,
                df_D_second_attack['End2'].str.strip('()').str.split(', ', expand=True).astype(float)
            ])
            all_raider_attack_names = pd.concat([
                all_raider_attack_names,
                df_D_second_attack['Second Attack Raider']
            ])

        # Create a scatter plot for all start and end coordinates
        start_trace = go.Scatter(
            x=all_start_coordinates[0],
            y=all_start_coordinates[1],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Start',
            text=all_raider_attack_names,
            hoverinfo='text'
        )

        end_trace = go.Scatter(
            x=all_end_coordinates[0],
            y=all_end_coordinates[1],
            mode='markers',
            marker=dict(color='blue', size=10),
            name='End',
            text=all_raider_attack_names,
            hoverinfo='text'
        )

        # Create the figure
        fig2  = go.Figure()

        # Add start and end traces to the figure
        fig2.add_trace(start_trace)
        fig2.add_trace(end_trace)

        # Add lines connecting start and end coordinates
        for i in range(len(all_start_coordinates)):
            fig2.add_trace(go.Scatter(
                x=[all_start_coordinates.iloc[i, 0], all_end_coordinates.iloc[i, 0]],
                y=[all_start_coordinates.iloc[i, 1], all_end_coordinates.iloc[i, 1]],
                mode='lines',
                line=dict(color='rgba(255, 255, 255, 0.8)', width=2),
                showlegend=False
            ))

        # Update layout to include the background image
        fig2.update_layout(
            images=[dict(
                source=background_image,
                xref="x",
                yref="y",
                x=0,
                y=0,  # Adjust the y-coordinate to align the image correctly
                sizex=470,
                sizey=330,  # Adjust the size as per your image's dimensions
                sizing="stretch",
                opacity=1,
                layer="below"
            )],
            xaxis=dict(range=[0, 470], showgrid=False),  # Adjust x-axis range as per your image's width
            yaxis=dict(range=[330, 0], showgrid=False),  # Adjust y-axis range as per your image's height
            title='Total Attack Locations',
            showlegend=True,
        )

        st.plotly_chart(fig2)

    elif chart_type == 'Successful':
        # Filter DataFrame for successful attacks
        filtered_attacks = df[(df['RD'] == 'D')]
        condition_touch_points = filtered_attacks['Touch Points'].between(1, 7)
        condition_raider_attack = filtered_attacks['Raider Attack'].notnull()
        condition_second_attack = filtered_attacks['Second Attack Raider'].notnull()
        condition_successful_raider_attack = condition_touch_points & condition_raider_attack & (filtered_attacks['Second Attack Raider'].isna())
        condition_successful_second_attack = condition_touch_points & condition_raider_attack & condition_second_attack

        # Apply conditions and select raider attack or second attack locations accordingly
        filtered_raider_attacks = filtered_attacks[condition_successful_raider_attack]
        filtered_second_attacks = filtered_attacks[condition_successful_second_attack]

        # Plot successful attacks
        attacks_count = pd.concat([filtered_raider_attacks['Raider Attack'], filtered_second_attacks['Second Attack Raider']]).value_counts()
        labels = attacks_count.index.tolist()
        values = attacks_count.values.tolist()
        fig_attacks = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig_attacks.update_traces(textinfo='label+value')
        fig_attacks.update_layout(title_text="Successful Attacks")
        

        # Plot attack locations for successful raider attacks
        if not filtered_raider_attacks.empty:
            # Combine all start and end coordinates for successful raider attacks
            # Combine all start and end coordinates for successful raider attacks
            all_raider_start_coordinates = filtered_raider_attacks['Start1'].str.strip('()').str.split(', ', expand=True).astype(float)
            all_raider_end_coordinates = filtered_raider_attacks['End1'].str.strip('()').str.split(', ', expand=True).astype(float)

            # Combine all start and end coordinates for successful second attacks
            all_second_start_coordinates = filtered_second_attacks['Start2'].str.strip('()').str.split(', ', expand=True).astype(float)
            all_second_end_coordinates = filtered_second_attacks['End2'].str.strip('()').str.split(', ', expand=True).astype(float)

            # Concatenate raider and second attack coordinates
            all_start_coordinates = pd.concat([all_raider_start_coordinates, all_second_start_coordinates])
            all_end_coordinates = pd.concat([all_raider_end_coordinates, all_second_end_coordinates])

            # Concatenate raider and second attack names
            all_raider_attack_names = pd.concat([filtered_raider_attacks['Raider Attack'], filtered_second_attacks['Second Attack Raider']])
            

            # Create a scatter plot for all start and end coordinates
            start_trace = go.Scatter(
                x=all_start_coordinates[0],
                y=all_start_coordinates[1],
                mode='markers',
                marker=dict(color='red', size=10),
                name='Start',
                text=all_raider_attack_names,
                hoverinfo='text'
            )

            end_trace = go.Scatter(
                x=all_end_coordinates[0],
                y=all_end_coordinates[1],
                mode='markers',
                marker=dict(color='blue', size=10),
                name='End',
                text=all_raider_attack_names,
                hoverinfo='text'
            )

            # Create the figure
            fig  = go.Figure()

            # Add start and end traces to the figure
            fig.add_trace(start_trace)
            fig.add_trace(end_trace)

            # Add lines connecting start and end coordinates
            for i in range(len(all_start_coordinates)):
                fig.add_trace(go.Scatter(
                    x=[all_start_coordinates.iloc[i, 0], all_end_coordinates.iloc[i, 0]],
                    y=[all_start_coordinates.iloc[i, 1], all_end_coordinates.iloc[i, 1]],
                    mode='lines',
                    line=dict(color='rgba(255, 255, 255, 0.8)', width=2),
                    showlegend=False
                ))

            # Update layout to include the background image
            fig.update_layout(
                images=[dict(source=background_image,xref="x",yref="y",x=0,y=0,  # Adjust the y-coordinate to align the image correctly
                    sizex=470,sizey=330,  # Adjust the size as per your image's dimensions
                    sizing="stretch",opacity=1,layer="below"
                )],
                xaxis=dict(range=[0, 470], showgrid=False),  # Adjust x-axis range as per your image's width
                yaxis=dict(range=[330, 0], showgrid=False),  # Adjust y-axis range as per your image's height
                title='Successful Attack Locations',
                showlegend=True,
            )
            st.plotly_chart(fig_attacks)
            st.plotly_chart(fig)
        else:
                st.write("Successful attacks found.")

    elif chart_type == 'Unsuccessful':
        filtered_attacks = df[(df['RD'] == 'D') & (df['Touch Points'] == 0)][['Raider Attack', 'Second Attack Raider']].stack()
        filtered_attacks = filtered_attacks[filtered_attacks != 'None']
        attacks_count = filtered_attacks.value_counts()
        labels = attacks_count.index.tolist()
        values = attacks_count.values.tolist()
        fig_attacks = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig_attacks.update_traces(textinfo='label+value')
        fig_attacks.update_layout(title_text="Unsuccessful Attacks")
        st.plotly_chart(fig_attacks)

        df_D = df[(df['RD'] == 'D') & (df['Touch Points'] == 0)]

        # Drop rows with missing values in 'Start1' and 'End1' columns
        df_D = df_D.dropna(subset=['Start1', 'End1'])

        # Combine all start and end coordinates
        all_start_coordinates = df_D['Start1'].str.strip('()').str.split(', ', expand=True).astype(float)
        all_end_coordinates = df_D['End1'].str.strip('()').str.split(', ', expand=True).astype(float)

        # Combine all raider attack names
        all_raider_attack_names = df_D['Raider Attack']

        # If 'Start2' and 'End2' are not NaN, combine their coordinates and raider attack names as well
        df_D_second_attack = df_D.dropna(subset=['Start2', 'End2'])
        if not df_D_second_attack.empty:
            all_start_coordinates = pd.concat([
                all_start_coordinates,
                df_D_second_attack['Start2'].str.strip('()').str.split(', ', expand=True).astype(float)
            ])
            all_end_coordinates = pd.concat([
                all_end_coordinates,
                df_D_second_attack['End2'].str.strip('()').str.split(', ', expand=True).astype(float)
            ])
            all_raider_attack_names = pd.concat([
                all_raider_attack_names,
                df_D_second_attack['Second Attack Raider']
            ])

        # Create a scatter plot for all start and end coordinates
        start_trace = go.Scatter(
            x=all_start_coordinates[0],
            y=all_start_coordinates[1],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Start',
            text=all_raider_attack_names,
            hoverinfo='text'
        )

        end_trace = go.Scatter(
            x=all_end_coordinates[0],
            y=all_end_coordinates[1],
            mode='markers',
            marker=dict(color='blue', size=10),
            name='End',
            text=all_raider_attack_names,
            hoverinfo='text'
        )

        # Create the figure
        fig2  = go.Figure()

        # Add start and end traces to the figure
        fig2.add_trace(start_trace)
        fig2.add_trace(end_trace)

        # Add lines connecting start and end coordinates
        for i in range(len(all_start_coordinates)):
            fig2.add_trace(go.Scatter(
                x=[all_start_coordinates.iloc[i, 0], all_end_coordinates.iloc[i, 0]],
                y=[all_start_coordinates.iloc[i, 1], all_end_coordinates.iloc[i, 1]],
                mode='lines',
                line=dict(color='rgba(255, 255, 255, 0.8)', width=2),
                showlegend=False
            ))

        # Update layout to include the background image
        fig2.update_layout(
            images=[dict(
                source=background_image,
                xref="x",
                yref="y",
                x=0,
                y=0,  # Adjust the y-coordinate to align the image correctly
                sizex=470,
                sizey=330,  # Adjust the size as per your image's dimensions
                sizing="stretch",
                opacity=1,
                layer="below"
            )],
            xaxis=dict(range=[0, 470], showgrid=False),  # Adjust x-axis range as per your image's width
            yaxis=dict(range=[330, 0], showgrid=False),  # Adjust y-axis range as per your image's height
            title='Unsuccessful Attack Locations',
            showlegend=True,
        )

        st.plotly_chart(fig2)

###----------------------Tackle and Tackle locations--------------------------------------------------------------###
def Tackels_Team1_and_Defence_Locations(chart_type, df):
    # Filter the dataframe based on the chart type
    if chart_type == 'total':
        filtered_df = df[(df['RD'] == 'D') & (df['Tackle Points'] != 'None') & (df['Defence Attack'] != 'None')]
    elif chart_type == 'successful':
        filtered_df = df[(df['RD'] == 'D') & (df['Tackle Points'].between(1, 2)) & (df['Defence Attack'] != 'None')]
    elif chart_type == 'unsuccessful':
        filtered_df = df[(df['RD'] == 'D') & (df['Tackle Points'] == 0) & (df['Defence Attack'] != 'None')]
    else:
        return  # Handle invalid chart type
    
    # Plot the pie chart
    tackles_count = filtered_df['Defence Attack'].value_counts()
    labels = tackles_count.index.tolist()
    values = tackles_count.values.tolist()
    fig_pie_chart = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig_pie_chart.update_traces(textinfo='label+value')
    fig_pie_chart.update_layout(title_text=f"{chart_type.capitalize()} Tackles")
    st.plotly_chart(fig_pie_chart)
    
    # Plot the location chart
    df_filtered = filtered_df.dropna(subset=['Start3', 'End3'])
    all_start_coordinates = df_filtered['Start3'].str.strip('()').str.split(', ', expand=True).astype(float)
    all_end_coordinates = df_filtered['End3'].str.strip('()').str.split(', ', expand=True).astype(float)
    all_attack_names = df_filtered['Defence Attack']
    
    fig_location_chart = go.Figure()
    fig_location_chart.add_layout_image(source=background_image, x=0, y=330, xref="x", yref="y", sizex=470, sizey=330, sizing="stretch", opacity=1, layer="below")
    start_trace = go.Scatter(x=all_start_coordinates[0], y=all_start_coordinates[1], mode='markers', marker=dict(color='green', size=10), name='Start (Defense)', text=all_attack_names, hoverinfo='text')
    end_trace = go.Scatter(x=all_end_coordinates[0], y=all_end_coordinates[1], mode='markers', marker=dict(color='orange', size=10), name='End (Defense)', text=all_attack_names, hoverinfo='text')
    fig_location_chart.add_trace(start_trace)
    fig_location_chart.add_trace(end_trace)

    for i in range(len(all_start_coordinates)):
        fig_location_chart.add_trace(go.Scatter(
            x=[all_start_coordinates.iloc[i, 0], all_end_coordinates.iloc[i, 0]],
            y=[all_start_coordinates.iloc[i, 1], all_end_coordinates.iloc[i, 1]],
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.8)', width=2),
            showlegend=False))

    fig_location_chart.update_layout(
        images=[dict(
            source=background_image,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=470,
            sizey=330,
            sizing="stretch",
            opacity=1,
            layer="below"
        )],
        xaxis=dict(range=[0, 470], showgrid=False),
        yaxis=dict(range=[330, 0], showgrid=False),
        title=f"{chart_type.capitalize()} Tackle Locations",
        showlegend=True,
    )
    st.plotly_chart(fig_location_chart)


def Tackels_Team2_and_Defence_Locations(chart_type, df):
    # Filter the dataframe based on the chart type
    if chart_type == 'total':
        filtered_df = df[(df['RD'] == 'R') & (df['Tackle Points'] != 'None') & (df['Defence Attack'] != 'None')]
    elif chart_type == 'successful':
        filtered_df = df[(df['RD'] == 'R') & (df['Tackle Points'].between(1, 2)) & (df['Defence Attack'] != 'None')]
    elif chart_type == 'unsuccessful':
        filtered_df = df[(df['RD'] == 'R') & (df['Tackle Points'] == 0) & (df['Defence Attack'] != 'None')]
    else:
        return  # Handle invalid chart type
    
    # Plot the pie chart
    tackles_count = filtered_df['Defence Attack'].value_counts()
    labels = tackles_count.index.tolist()
    values = tackles_count.values.tolist()
    fig_pie_chart = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig_pie_chart.update_traces(textinfo='label+value')
    fig_pie_chart.update_layout(title_text=f"{chart_type.capitalize()} Tackles")
    st.plotly_chart(fig_pie_chart)
    
    # Plot the location chart
    df_filtered = filtered_df.dropna(subset=['Start3', 'End3'])
    all_start_coordinates = df_filtered['Start3'].str.strip('()').str.split(', ', expand=True).astype(float)
    all_end_coordinates = df_filtered['End3'].str.strip('()').str.split(', ', expand=True).astype(float)
    all_attack_names = df_filtered['Defence Attack']
    
    fig_location_chart = go.Figure()
    fig_location_chart.add_layout_image(source=background_image, x=0, y=330, xref="x", yref="y", sizex=470, sizey=330, sizing="stretch", opacity=1, layer="below")
    start_trace = go.Scatter(x=all_start_coordinates[0], y=all_start_coordinates[1], mode='markers', marker=dict(color='green', size=10), name='Start (Defense)', text=all_attack_names, hoverinfo='text')
    end_trace = go.Scatter(x=all_end_coordinates[0], y=all_end_coordinates[1], mode='markers', marker=dict(color='orange', size=10), name='End (Defense)', text=all_attack_names, hoverinfo='text')
    fig_location_chart.add_trace(start_trace)
    fig_location_chart.add_trace(end_trace)

    for i in range(len(all_start_coordinates)):
        fig_location_chart.add_trace(go.Scatter(
            x=[all_start_coordinates.iloc[i, 0], all_end_coordinates.iloc[i, 0]],
            y=[all_start_coordinates.iloc[i, 1], all_end_coordinates.iloc[i, 1]],
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.8)', width=2),
            showlegend=False))

    fig_location_chart.update_layout(
        images=[dict(
            source=background_image,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=470,
            sizey=330,
            sizing="stretch",
            opacity=1,
            layer="below"
        )],
        xaxis=dict(range=[0, 470], showgrid=False),
        yaxis=dict(range=[330, 0], showgrid=False),
        title=f"{chart_type.capitalize()} Tackle Locations",
        showlegend=True,
    )
    st.plotly_chart(fig_location_chart)




def draw_pie_chart(df):
    # Filter the DataFrame based on the condition RD == 'R'
    filtered_df = df[df['RD'] == 'R']
    
    # Group by Total Players at Defence and sum Touch Points within each group
    grouped_df = filtered_df.groupby('Total Players at Defence')['Touch Points'].sum().reset_index()
    
    # Create pie chart trace
    trace = go.Pie(labels=grouped_df['Total Players at Defence'], values=grouped_df['Touch Points'], name='Touch Points')
    
    # Create layout
    layout = go.Layout(title='Pie Chart for Total Players at Defence and Sum of Touch Points (RD = R)')
    
    # Create figure
    fig = go.Figure(data=[trace], layout=layout)
    
    # Plot chart
    st.plotly_chart(fig)






# Import the do_or_die function
def do_or_die_Team1(df):
    # Filter the DataFrame based on the condition RD == 'R'
    filtered_df = df[df['RD'] == 'R']
    
    # Filter the DataFrame to include only do-or-die raids
    do_or_die_df = filtered_df[filtered_df['Do-or-die Raids'].notnull()]
    
    # Count the total number of do-or-die raids
    total_do_or_die_raids = do_or_die_df['Do-or-die Raids'].sum()
    
    # Calculate the number of successful and unsuccessful raids within do-or-die raids
    successful_raids = ((do_or_die_df['team1 Points'] > 0) & (do_or_die_df['team1 Points'] <= 7)).sum()
    unsuccessful_raids = (do_or_die_df['team1 Points'] == 0).sum()
    
    # Create labels and values for the pie chart
    labels = ['Total Do-or-die Raids', 'Successful Raids', 'Unsuccessful Raids']
    values = [total_do_or_die_raids, successful_raids, unsuccessful_raids]
    
    # Create the pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_traces(textinfo='label+value')
    fig.update_layout(title='Do-or-die Raids')
    
    # Plot chart
    st.plotly_chart(fig,use_container_width=True, width=500)




# Import the do_or_die function
def do_or_die_Team2(df):
    # Filter the DataFrame based on the condition RD == 'R'
    filtered_df = df[df['RD'] == 'D']
    
    # Filter the DataFrame to include only do-or-die raids
    do_or_die_df = filtered_df[filtered_df['Do-or-die Raids'].notnull()]
    
    # Count the total number of do-or-die raids
    total_do_or_die_raids = do_or_die_df['Do-or-die Raids'].sum()
    
    # Calculate the number of successful and unsuccessful raids within do-or-die raids
    successful_raids = ((do_or_die_df['Team2 Points'] > 0) & (do_or_die_df['Team2 Points'] <= 7)).sum()
    unsuccessful_raids = (do_or_die_df['Team2 Points'] == 0).sum()
    
    # Create labels and values for the pie chart
    labels = ['Total Do-or-die Raids', 'Successful Raids', 'Unsuccessful Raids']
    values = [total_do_or_die_raids, successful_raids, unsuccessful_raids]
    
    # Create the pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_traces(textinfo='label+value')
    fig.update_layout(title='Do-or-die Raids')
    
    # Plot chart
    st.plotly_chart(fig,use_container_width=True, width=500)





   









            
