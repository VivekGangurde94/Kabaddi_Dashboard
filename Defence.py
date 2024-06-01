import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
background_image = Image.open('Kabaddi.png')



def Tackels_Team1_and_Defence_Locations2(chart_type, df, selected_player):
    if chart_type == 'Total':
        # Filter the DataFrame based on the conditions (touch points between 1 and 7)
        filtered_df = df[(df['RD'] == 'D') & (df['Tackle Points'] != 'None') & (df['Defence Attack'] != 'None')]
        
        if selected_player and selected_player != "All Players":
            filtered_df = filtered_df[(filtered_df['Defence player No.'].eq(selected_player)) | (filtered_df['2nd Player Out'].eq(selected_player))]
    
        # Plot the pie chart
        tackles_count = filtered_df['Defence Attack'].value_counts()
        labels = tackles_count.index.tolist()
        values = tackles_count.values.tolist()
        fig_pie_chart = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig_pie_chart.update_traces(textinfo='label+value')
        fig_pie_chart.update_layout(title_text=f"Tackles")
        
        if not filtered_df.empty:
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
                title=  "Tackle Locations",
                showlegend=True,
            )
            st.plotly_chart(fig_pie_chart)
            st.plotly_chart(fig_location_chart)
        else:
            st.warning("# No Tackle took place.")

    elif chart_type == 'Successful':
        filtered_df = df[(df['RD'] == 'D') & (df['Tackle Points'].between(1, 2)) & (df['Defence Attack'] != 'None')]

        if selected_player and selected_player != "All Players":
            filtered_df = filtered_df[(filtered_df['Defence player No.'].eq(selected_player)) | (filtered_df['2nd Player Out'].eq(selected_player))]
    
        # Plot the pie chart
        tackles_count = filtered_df['Defence Attack'].value_counts()
        labels = tackles_count.index.tolist()
        values = tackles_count.values.tolist()
        fig_pie_chart = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig_pie_chart.update_traces(textinfo='label+value')
        fig_pie_chart.update_layout(title_text=f"Tackles")
        st.plotly_chart(fig_pie_chart)

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
            title=  "Tackle Locations",
            showlegend=True,
        )
        st.plotly_chart(fig_location_chart)

    elif chart_type == 'Unsuccessful':
        filtered_df = df[(df['RD'] == 'D') & (df['Tackle Points'] == 0) & (df['Defence Attack'] != 'None')]
        
        if selected_player and selected_player != "All Players":
            filtered_df = filtered_df[(filtered_df['Defence player No.'].eq(selected_player)) | (filtered_df['2nd Player Out'].eq(selected_player))]
        
        # Plot the pie chart
        tackles_count = filtered_df['Defence Attack'].value_counts()
        labels = tackles_count.index.tolist()
        values = tackles_count.values.tolist()
        fig_pie_chart = go.Figure(data=[go.Pie(labels=labels, values=values)])
        fig_pie_chart.update_traces(textinfo='label+value')
        fig_pie_chart.update_layout(title_text=f"Tackles")
            
        if not filtered_df.empty:
            df_filtered = filtered_df.dropna(subset=['Start3', 'End3'])
            all_start_coordinates = df_filtered['Start3'].str.strip('()').str.split(', ', expand=True).astype(float)
            all_end_coordinates = df_filtered['End3'].str.strip('()').str.split(', ', expand=True).astype(float)
            all_attack_names = df_filtered['Defence Attack']
                
            fig_location_chart = go.Figure()
            fig_location_chart.add_layout_image(source=background_image, x=0, y=330, xref="x", yref="y", sizex=470, sizey=330, sizing="stretch", opacity=1, layer="below")
            
            if len(all_start_coordinates.columns) >= 2 and len(all_end_coordinates.columns) >= 2:
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
                    title=  "Tackle Locations",
                    showlegend=True,
                )
                st.plotly_chart(fig_pie_chart)
                st.plotly_chart(fig_location_chart)
            else:
                st.warning("# No valid tackle location data available.")
        else:
            st.warning("# No Unsuccessful Tackle took place.")

#----------------------------------------video Data -----------------------------------------#

from functools import reduce

def Team1_defender_Video_data(df, unique_r_nos, tackle_points_values):
    # Filter rows where 'RD' column is 'D'
    df_d = df[df['RD'] == 'D']
    
    # Apply the selected filters
    if not unique_r_nos and not tackle_points_values:
        filtered_df = df_d
    else:
        conditions = []
        if unique_r_nos:
            # Check if the selected number is present in any of the four columns
            conditions.append(df_d[['Defence player No.', '2nd Player Out', '3rd Player Out', '4th Player Out']].isin(unique_r_nos).any(axis=1))
        if tackle_points_values:
            conditions.append(df_d["Tackle Points"].isin(tackle_points_values))
        
        filtered_df = df_d[reduce(lambda x, y: x & y, conditions)]
    
    return filtered_df

# ---------------------------------------Defender Out by Attack --------------------------------------------------------------------------------#
        
def Defender_out(df, selected_player):
    filtered_df = df[(df['RD'] == 'D') & ((df['Defence player No.'] == selected_player) | (df['2nd Player Out'] == selected_player))]
    condition_defence_null = filtered_df['Defence Attack'].isnull()
    condition_touch_points = filtered_df['Touch Points'].between(1, 7)
    condition_raider_attack = filtered_df['Raider Attack'].notnull()
    condition_second_attack = filtered_df['Second Attack Raider'].notnull()
    condition_successful_raider_attack = condition_defence_null & condition_touch_points & condition_raider_attack & (filtered_df['Second Attack Raider'].isna())
    condition_successful_second_attack = condition_defence_null & condition_touch_points & condition_raider_attack & condition_second_attack

    # Apply conditions and select raider attack or second attack locations accordingly
    filtered_raider_attacks = filtered_df[condition_successful_raider_attack]
    filtered_second_attacks = filtered_df[condition_successful_second_attack]

    # Plot successful attacks
    attacks_count = pd.concat([filtered_raider_attacks['Raider Attack'], filtered_second_attacks['Second Attack Raider']]).value_counts()
    labels = attacks_count.index.tolist()
    values = attacks_count.values.tolist()
    fig_attacks = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig_attacks.update_traces(textinfo='label+value')
    fig_attacks.update_layout(title_text="Successful Attacks On Defender")
    

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
            title='Successful Attacks Locations on Defender',
            showlegend=True,
        )
        st.plotly_chart(fig_attacks)
        st.plotly_chart(fig)
    else:
            st.warning(" No Successful Attacks Found On Defender")





   
