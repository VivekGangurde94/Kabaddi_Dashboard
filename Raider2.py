import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
background_image = Image.open('Kabaddi.png')

import plotly.graph_objects as go

def Team2_bonus_points_distribution_Raider(df,r_no):
    # Filter DataFrame for bonus point attempts and bonus points
    bonus_points_df = df[(df['RD'] == 'D') & (df['R No.'] == r_no)][['Bonus Point attempt', 'Bonus Points']]
    
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

    # Check if both charts are empty
    if successful_bonus_points_count.empty and unsuccessful_attempts_per_corner.empty:
        st.warning("# No Bonus Points Scored and Attempt")
        # You can customize this message according to your needs
    elif successful_bonus_points_count.empty:
        st.warning("# No Bonus Points Scored")
        # You can customize this message according to your needs
    elif unsuccessful_attempts_per_corner.empty:
        st.warning("# No Bonus Points Attempt")
        

    return fig_successful_bonus_points, fig_unsuccessful_bonus_points


##---------------------------------------------escape-----------------------------------------#

def Team2_raider_escape_Raider(df,r_no):
    filtered_df = df[(df['RD'] == 'D') & (df['R No.'] == r_no)]  # Filter rows where RD column is 'R'
    raider_escape_counts = filtered_df['Raider Escape'].value_counts()
    labels = raider_escape_counts.index.tolist()
    values = raider_escape_counts.values.tolist()  
    fig_donut_chart = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5)])
    fig_donut_chart.update_traces(textinfo='label+value')
    fig_donut_chart.update_layout(title_text="Raider Escape")
    st.plotly_chart(fig_donut_chart)

# ----------------------------------------------------Attack Locations--------------------------------------------------------#

def Team2_Attack_and_Attack_Locations_Raider(chart_type, df, r_no):
    if chart_type == 'Total':
        filtered_attacks = df[(df['RD'] == 'D') & (df['R No.'] == r_no)][['Raider Attack', 'Second Attack Raider']].stack()
        filtered_attacks = filtered_attacks[filtered_attacks != 'None']
        attacks_count = filtered_attacks.value_counts()
        labels = attacks_count.index.tolist()
        values = attacks_count.values.tolist()
        fig_attacks = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig_attacks.update_traces(textinfo='label+value')
        fig_attacks.update_layout(title_text="Total Attacks")
        

        if not filtered_attacks.empty:
            df_R = df[(df['RD'] == 'D') & (df['R No.'] == r_no)]
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
                images=[dict(source=background_image,xref="x",yref="y",x=0,
                    y=330,  # Adjust the y-coordinate to align the image correctly
                    sizex=470,sizey=330,  # Adjust the size as per your image's dimensions
                    sizing="stretch",opacity=1,
                    layer="below"
                )],
                xaxis=dict(range=[0, 470], showgrid=False),  # Adjust x-axis range as per your image's width
                yaxis=dict(range=[0, 330], showgrid=False),  # Adjust y-axis range as per your image's height
                title='Total Attack Locations',
                showlegend=True,
            )
            st.plotly_chart(fig_attacks)
            st.plotly_chart(fig)
        else:
            st.warning("No attacks took place.")

    elif chart_type == 'Successful':
            # Filter DataFrame for successful attacks
        filtered_attacks = df[(df['RD'] == 'D') & (df['R No.'] == r_no)]
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
                images=[dict(source=background_image,xref="x",yref="y",x=0,y=330,  # Adjust the y-coordinate to align the image correctly
                    sizex=470,sizey=330,  # Adjust the size as per your image's dimensions
                    sizing="stretch",opacity=1,layer="below"
                )],
                xaxis=dict(range=[0, 470], showgrid=False),  # Adjust x-axis range as per your image's width
                yaxis=dict(range=[0, 330], showgrid=False),  # Adjust y-axis range as per your image's height
                title='Successful Attack Locations',
                showlegend=True,
            )
            st.plotly_chart(fig_attacks)
            st.plotly_chart(fig)
        else:
                st.warning("No Successful attacks found.")




    
    elif chart_type == 'Unsuccessful':
            # Filter DataFrame for unsuccessful attacks
            filtered_attacks = df[(df['RD'] == 'D') & (df['R No.'] == r_no)]

            # Condition 1: Touch points equal to zero
            condition_touch_points_zero = filtered_attacks['Touch Points'] == 0

            # Condition 2: Touch points between 1 and 7
            condition_touch_points_1_7 = filtered_attacks['Touch Points'].between(1, 7)

            # Condition 3: Both raider attack and second attack are present
            condition_raider_attack = filtered_attacks['Raider Attack'].notnull()
            condition_second_attack = filtered_attacks['Second Attack Raider'].notnull()

            # Select raider attack and second attack raider when touch points are equal to zero and both are present
            filtered_unsuccessful_attacks_zero = filtered_attacks.loc[
                (condition_touch_points_zero) & (condition_raider_attack) & (condition_second_attack)]

            # Select only the Raider Attack column when touch points are between 1 and 7 and second attack is present
            filtered_unsuccessful_attacks_1_7 = filtered_attacks.loc[
                (condition_touch_points_1_7) & (condition_raider_attack) & (condition_second_attack)]

            # Concatenate the filtered DataFrames
            filtered_unsuccessful_attacks = pd.concat(
                [filtered_unsuccessful_attacks_zero, filtered_unsuccessful_attacks_1_7], ignore_index=True)

            # Plot unsuccessful attack locations
            if not filtered_unsuccessful_attacks.empty:
                # Drop rows with missing values in 'Start1' and 'End1' columns
                filtered_unsuccessful_attacks = filtered_unsuccessful_attacks.dropna(subset=['Start1', 'End1'])

                # Combine all start and end coordinates
                all_start_coordinates = filtered_unsuccessful_attacks['Start1'].str.strip('()').str.split(', ',
                                                                                                        expand=True).astype(
                    float)
                all_end_coordinates = filtered_unsuccessful_attacks['End1'].str.strip('()').str.split(', ',
                                                                                                    expand=True).astype(
                    float)

                # Combine all raider attack names
                all_raider_attack_names = filtered_unsuccessful_attacks['Raider Attack']

                # If 'Start2' and 'End2' are not NaN, combine their coordinates and raider attack names as well
                df_second_attack = filtered_unsuccessful_attacks.dropna(subset=['Start2', 'End2'])

                # Apply condition: Only consider raider attack when touch points are zero
                if not df_second_attack.empty:
                    all_start_coordinates = pd.concat([
                        all_start_coordinates,
                        df_second_attack['Start2'].str.strip('()').str.split(', ', expand=True).astype(float)
                    ])
                    all_end_coordinates = pd.concat([
                        all_end_coordinates,
                        df_second_attack['End2'].str.strip('()').str.split(', ', expand=True).astype(float)
                    ])
                    all_raider_attack_names = pd.concat([
                        all_raider_attack_names,
                        df_second_attack['Second Attack Raider']
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
                fig = go.Figure()

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
                        y=330,  # Adjust the y-coordinate to align the image correctly
                        sizex=470,
                        sizey=330,  # Adjust the size as per your image's dimensions
                        sizing="stretch",
                        opacity=1,
                        layer="below"
                    )],
                    xaxis=dict(range=[0, 470], showgrid=False),  # Adjust x-axis range as per your image's width
                    yaxis=dict(range=[0, 330], showgrid=False),  # Adjust y-axis range as per your image's height
                    title='Unsuccessful Attack Locations',
                    showlegend=True,
                )
                # Plot pie chart for unsuccessful attacks
                filtered_attacks = filtered_unsuccessful_attacks[['Raider Attack', 'Second Attack Raider']].stack()
                filtered_attacks = filtered_attacks[filtered_attacks != 'None']
                attacks_count = filtered_attacks.value_counts()
                labels = attacks_count.index.tolist()
                values = attacks_count.values.tolist()
                fig_attacks = go.Figure(data=[go.Pie(labels=labels, values=values)])
                fig_attacks.update_traces(textinfo='label+value')
                fig_attacks.update_layout(title_text="Unsuccessful Attacks")
                st.plotly_chart(fig_attacks)
                st.plotly_chart(fig)

                
            else:
                st.warning("No unsuccessful attacks found.")



##-------------------------------------Touch_Points_Score at player present at mat-------------------------------------------------###

def Team2_Touch_Points_at_Player_present_Raider(df, r_no):
    # Filter the DataFrame based on the condition RD == 'R' and the raider number
    filtered_df = df[(df['RD'] == 'D') & (df['R No.'] == r_no)]
    
    # Group by Total Players at Defence and sum Touch Points within each group
    grouped_df = filtered_df.groupby('Total Players at Defence')['Touch Points'].sum().reset_index()
    
    if not grouped_df.empty:
        # Create pie chart trace
        labels = grouped_df['Total Players at Defence'].astype(str) + " Defender"
        values = grouped_df['Touch Points']
        trace = go.Pie(labels=labels, values=values, name='Touch Points')
        
        # Create layout
        layout = go.Layout(title='Points Scored with Number of Defenders Present')
        
        # Create figure
        fig = go.Figure(data=[trace], layout=layout)
        fig.update_traces(textinfo='label+value')
        
        # Display the pie chart using plotly_chart
        st.plotly_chart(fig)
    else:
        st.warning("No touch points found for the selected player.")



##------------------------------------do-or-Die Raids--------------------------------------------------------------------------------------------##
def do_or_die_Team2_Raider(df,r_no):
    # Filter the DataFrame based on the condition RD == 'R'
    filtered_df = df[(df['RD'] == 'D') & (df['R No.'] == r_no)]
    
    # Filter the DataFrame to include only do-or-die raids
    do_or_die_df = filtered_df[filtered_df['Do-or-die Raids'].notnull()]
    
    # Count the total number of do-or-die raids
    total_do_or_die_raids = do_or_die_df['Do-or-die Raids'].sum()
    
    # Calculate the number of successful and unsuccessful raids within do-or-die raids
    successful_raids = ((do_or_die_df['Team2 Points'] > 0) & (do_or_die_df['team1 Points'] <= 7)).sum()
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

    ##-----------------------------Raider Tackeld -----------------------------------------------------------------------------------------##


def Team2_Raider_tackled(df,r_no):
    # Filter DataFrame for rows where RD column is R and Tackle Point is 1
    raider_tackled_df = df[(df['RD'] == 'D') & (df['R No.'] == r_no) & (df['Tackle Points'] == 1)]
    
    if raider_tackled_df.empty:
        st.title("No raider were tackled.")
        return
    
    # Count the occurrences of each value in the Defence Attack column
    defence_attacks_count = raider_tackled_df['Defence Attack'].value_counts()
    
    # Create labels and values for the Pie chart
    labels = defence_attacks_count.index.tolist()
    values = defence_attacks_count.values.tolist()
    
    # Create Pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_traces(textinfo='label+value')
    fig.update_layout(title_text="Raider Tackled - Defence Attack Distribution")
    st.plotly_chart(fig)

    raider_tackled_df = raider_tackled_df.dropna(subset=['Start3', 'End3'])
    all_start_coordinates = raider_tackled_df['Start3'].str.strip('()').str.split(', ', expand=True).astype(float)
    all_end_coordinates = raider_tackled_df['End3'].str.strip('()').str.split(', ', expand=True).astype(float)
    all_attack_names = raider_tackled_df['Defence Attack']
    
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
            y=330,
            sizex=470,
            sizey=330,
            sizing="stretch",
            opacity=1,
            layer="below"
        )],
        xaxis=dict(range=[0, 470], showgrid=False),
        yaxis=dict(range=[0, 330], showgrid=False),
        title= " Tackle Locations",
        showlegend=True,
    )
    st.plotly_chart(fig_location_chart)

#---------------------------------------Scorecard ------------------------------------------------------------------------#

def Team2_Scorecard_player(df, selected_player):
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

    if selected_player:
        team2_combined_data = team2_combined_data.loc[selected_player]

    if not team2_combined_data.empty:
        team2_combined_data.rename_axis('Jersey No.', inplace=True)

    return team2_combined_data