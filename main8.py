
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl
import streamlit as st
from streamlit.proto.Selectbox_pb2 import Selectbox
import pandas as pd
from demo_data import (plot_cumulative_points, analyze_kabaddi_data_team1, analyze_kabaddi_data_team2,
                       Team1_Scorecard, Team2_Scorecard, draw_tornado_chart, team_points,draw_tornado_chart_for_half,
                       analyze_kabaddi_data_by_half,analyze_kabaddi_data_by_half,
                       Team1_bonus_points_distribution, Team2_bonus_points_distribution, Team1_Video_data,Team2_Video_data,
                       Tackels_Team2_and_Defence_Locations,Team1_Attack_and_Attack_Locations,Team2_Attack_and_Attack_Locations,
                       Tackels_Team1_and_Defence_Locations,Team1_raider_escape,Team2_raider_escape,do_or_die_Team1,do_or_die_Team2)

from Raider import(Team1_Attack_and_Attack_Locations_Raider,Team1_raider_escape_Raider,Team1_bonus_points_distribution_Raider,
                   Touch_Points_at_Player_present_Raider,Team1_Scorecard_players
                   ,do_or_die_Team1_Raider,Team1_Raider_tackled)

from Raider2 import(Team2_Attack_and_Attack_Locations_Raider,Team2_bonus_points_distribution_Raider,
                    Team2_raider_escape_Raider,Team2_Touch_Points_at_Player_present_Raider,Team2_Raider_tackled,
                    do_or_die_Team2_Raider, Team2_Scorecard_player)

from Defence import (Tackels_Team1_and_Defence_Locations2,Team1_defender_Video_data,Defender_out)
from Defence2 import(Tackels_Team2_and_Defence_Locations_Players,Team2_defender_Video_data,Team2_Defender_out)
import os
st.set_page_config(layout="wide")
def main():
    st.header("TAGTIX")
    st.title('Kabaddi Dashboard')

    # Use Streamlit file uploader to upload the Excel file
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

    if uploaded_file is not None:
        try:
            # Read the uploaded Excel file
            df = pd.read_excel(uploaded_file)
            

            # Navigation
            st.sidebar.write("# Navigation")
            page = st.sidebar.radio("Go to", ["Team Analysis", "Raider Analysis", "Defender Analysis"])

            if page == "Team Analysis":
                show_scoreboard(df)
            elif page == "Raider Analysis":
                show_analysis(df)
            elif page == "Defender Analysis":
                Defender_analysis(df)

        except Exception as e:
            st.error(f"Error reading the file: {e}")
    else:
        st.write("Please upload a file.")

def show_scoreboard(df):
        # Load the Excel file kabaddi_data2.xlsx
        # excel_file = "AhD vs Plaghar_final.xlsx"
        # df = pd.read_excel(excel_file)
        

        st.write('## Score')
        team1_points, team2_points = team_points(df)
        
        # Create two columns for laying out the scores side by side
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h2 style='text-align: center'>Ahmednagar</h2>", unsafe_allow_html=True)
            st.write(f"<h1 style='text-align: center; margin: 0'>{team1_points}</h1>", unsafe_allow_html=True)

        with col2:
            st.markdown("<h2 style='text-align: center'>Palghar</h2>", unsafe_allow_html=True)
            st.write(f"<h1 style='text-align: center; margin: 0'>{team2_points}</h1>", unsafe_allow_html=True)

        
        plot_cumulative_points(df)

        analysis_results_team1 = analyze_kabaddi_data_team1(df)
        analysis_results_team2 = analyze_kabaddi_data_team2(df)

        analysis_type = st.select_slider("Select Half", ["Total", "1st Half", "2nd Half"])


        # Perform analysis based on selected type
        if analysis_type == "Total":
            analysis_results_team1 = analyze_kabaddi_data_team1(df)
            analysis_results_team2 = analyze_kabaddi_data_team2(df)
            draw_tornado_chart(analysis_results_team1, analysis_results_team2)
        elif analysis_type == "1st Half":
            analysis_results_team1_half1, analysis_results_team1_half2, analysis_results_team2_half1, analysis_results_team2_half2 = analyze_kabaddi_data_by_half(df)
            draw_tornado_chart_for_half(analysis_results_team1_half1, analysis_results_team2_half1, "1st HALF")

        elif analysis_type == "2nd Half":
            analysis_results_team1_half1, analysis_results_team1_half2, analysis_results_team2_half1, analysis_results_team2_half2 = analyze_kabaddi_data_by_half(df)
            draw_tornado_chart_for_half(analysis_results_team1_half2, analysis_results_team2_half2, "2nd HALF")

        

        st.write('## Scoreboard')

        st.write('## Ahmednagar Scorecard')
        team1_scorecard = Team1_Scorecard(df)
        
        st.write(team1_scorecard)
        with st.sidebar:
            st.header("Ahmednagar Raider Video")
            unique_r_nos_team1 = st.multiselect("Select Jesrey No.", df[df['RD'] == 'R']["R No."].unique(), key='team1_unique_r_nos')
            bonus_points_values_team1 = st.multiselect("Select Bonus Points", df[df['RD'] == 'R']["Bonus Points"].unique(), key='team1_bonus_points_values')
            touch_points_values_team1 = st.multiselect("Select Touch Points",df[df['RD'] == 'R']["Touch Points"].unique(), key='team1_touch_points_values')
            Empty_Raid_values_team1 = st.multiselect("Select Empty Raid",df[df['RD'] == 'R']["Empty Raid"].unique(), key='team1_Empty_Raid_values')
            Do_or_die_Raids_values_team1 = st.multiselect("Select Do-or-die Raids",df[df['RD'] == 'R']["Do-or-die Raids"].unique(), key='team1_Do-or-die Raids_values')

        # Filter the data based on user input for Team 1
        filtered_df_team1 = Team1_Video_data(df, unique_r_nos_team1, bonus_points_values_team1, touch_points_values_team1,
                                             Empty_Raid_values_team1,Do_or_die_Raids_values_team1)
        
        # Display filtered video links for Team 1
        with st.expander("Ahmednagar Video"):
            st.write("Ahmednagar Video Links:")
            for link in filtered_df_team1["Video Links"]:
                st.markdown(f"[{link}]({link})")

        st.write('## Palghar Scorecard')
        team2_scorecard = Team2_Scorecard(df)
        st.write(team2_scorecard)

        with st.sidebar:
            st.header("Palghar Raider Video")
            unique_r_nos_team2 = st.multiselect("Select Jesrey No.", df[df['RD'] == 'D']["R No."].unique(), key='team2_unique_r_nos')
            bonus_points_values_team2 = st.multiselect("Select Bonus Points", df[df['RD'] == 'D']["Bonus Points"].unique(), key='team2_bonus_points_values')
            touch_points_values_team2 = st.multiselect("Select Touch Points",df[df['RD'] == 'D']["Touch Points"].unique(), key='team2_touch_points_values')
            Empty_Raid_values_team2 = st.multiselect("Select Empty Raid",df[df['RD'] == 'D']["Empty Raid"].unique(), key='team2_Empty_Raid_values')
            Do_or_die_Raids_values_team2 = st.multiselect("Select Do-or-die Raids",df[df['RD'] == 'D']["Do-or-die Raids"].unique(), key='team2_Do-or-die Raids_values')

        # Filter the data based on user input for Team 2
        filtered_df_team2 = Team2_Video_data(df, unique_r_nos_team2, bonus_points_values_team2, touch_points_values_team2,
                                             Empty_Raid_values_team2,Do_or_die_Raids_values_team2)
        
        # Display filtered video links for Team 2
        with st.expander("Palghar Video"):
            st.write("Palghar Video Links:")
            for link in filtered_df_team2["Video Links"]:
                st.markdown(f"[{link}]({link})")

        st.write('## Analysis')

        # Side-by-side selection for bonus points, attacks, and defense
        
        option = st.sidebar.radio("Select Chart", ("Bonus Points", "Attacks", "Tackles","Escape","Do-Or-Die"))

        if option == 'Bonus Points':
            st.write('## Bonus Points')

            col1, col2 = st.columns(2)

            with col1:
                st.write('### Ahmednagar')
                option_team1 = st.selectbox('Select Ahmednagar Bonus Points:', ['Successful', 'Unsuccessful'])

                if option_team1 == 'Successful':
                    fig_successful, _ = Team1_bonus_points_distribution(df)
                    st.plotly_chart(fig_successful, use_container_width=True, width=600)  # Adjust width here
                else:
                    _, fig_unsuccessful = Team1_bonus_points_distribution(df)
                    st.plotly_chart(fig_unsuccessful, use_container_width=True, width=600)  # Adjust width here

            with col2:
                st.write('### Palghar')
                option_team2 = st.selectbox('Select Palghar Bonus Points:', ['Successful', 'Unsuccessful'])

                if option_team2 == 'Successful':
                    fig_successful, _ = Team2_bonus_points_distribution(df)
                    st.plotly_chart(fig_successful, use_container_width=True, width=600)  # Adjust width here
                else:
                    _, fig_unsuccessful = Team2_bonus_points_distribution(df)
                    st.plotly_chart(fig_unsuccessful, use_container_width=True, width=600)  # Adjust width here

        elif option == 'Attacks':
            st.write('## Attacks')

            col1, col2 = st.columns(2)
            with col1:
                chart_type_team1 = st.selectbox('Ahmednagar Attacks By Raider', ['Total', 'Successful', 'Unsuccessful'])
                Team1_Attack_and_Attack_Locations(chart_type_team1, df)

            with col2:
                chart_type_team2 = st.selectbox('Palghar Attacks By Raider', ['Total', 'Successful', 'Unsuccessful'])
                Team2_Attack_and_Attack_Locations(chart_type_team2, df)


        elif option == 'Tackles':
            st.write('## Tackles')

            col1, col2 = st.columns(2)
            with col1:
                st.write('### Ahmednagar')
                option_team1 = st.selectbox('Select Ahmednagar Tackles:', ['Total', 'Successful', 'Unsuccessful'])
                Tackels_Team1_and_Defence_Locations(option_team1.lower(), df)
            with col2:
                st.write('### Palghar')
                option_team2 = st.selectbox('Select Palghar Tackles:', ['Total', 'Successful', 'Unsuccessful'])
                Tackels_Team2_and_Defence_Locations(option_team2.lower(), df)

        elif option == 'Escape':
            st.write('## Escape')
            col1, col2 = st.columns(2)
            with col1:
                st.write('### Ahmednagar')
                Team1_raider_escape(df)
            with col2:
                st.write('### Palghar')
                Team2_raider_escape(df)
        
        elif option == "Do-Or-Die":
            st.write('## Do-Or-Die')
            col1, col2 = st.columns(2)  # Adjust the width ratio here

            with col1:
                st.write('## Ahmednagar')
                do_or_die_Team1(df)

            with col2:
                st.write('## Palghar')
                do_or_die_Team2(df)




def show_analysis(df):
    # Filter the DataFrame to include only rows where the value of the "RD" column is "R"
    filtered_df = df[df['RD'] == 'R']
    filtered_df2 = df[df['RD'] == 'D']

    
    # Get the unique values from the "R No." column for both teams
    Team1_r_numbers = filtered_df['R No.'].unique()
    Team2_r_numbers = filtered_df2['R No.'].unique()

    # Sidebar for selecting the Raider Number for Team 1
    st.sidebar.write("# Team 1 Raider Number")
    team1_r_no = st.sidebar.selectbox("Select Raider Number for Team 1", ["Choose"] + list(Team1_r_numbers))
    


    # Sidebar for selecting the Raider Number for Team 2
    st.sidebar.write("# Team 2 Raider Number")
    team2_r_no = st.sidebar.selectbox("Select Raider Number for Team 2", ["Choose"] + list(Team2_r_numbers))

    # Navigation sidebar
    st.sidebar.write("# Navigation")
    page = st.sidebar.radio("Go to", ["Scorecard","Bonus Points", "Attack Locations","Touch Points","Escape", "Do-or-Die","Raider Tackled"])

    # Conditional display based on the selected page and the selected team
    if page == "Bonus Points":
        if team1_r_no != "Choose":
            show_team1_bonus_points_distribution(df, team1_r_no)
        elif team2_r_no != "Choose":
            show_Team2_bonus_points_distribution_Raider(df, team2_r_no)

    elif page == "Scorecard":
        if team1_r_no != "Choose":
         col1, col2 = st.columns(2)
         with col1:
            show_Team1_Scorecard_players(df, team1_r_no)

         with col2:
            st.header("Ahmednagar Raider Video")
            team1_r_no = st.multiselect("Select Raider: ", df[df['RD'] == 'R']["R No."].unique(), key='team1_unique_r_nos')
            bonus_points_values_team1 = st.multiselect("Select Bonus Points", df[df['RD'] == 'R']["Bonus Points"].unique(), key='team1_bonus_points_values')
            touch_points_values_team1 = st.multiselect("Select Touch Points",df[df['RD'] == 'R']["Touch Points"].unique(), key='team1_touch_points_values')
            Empty_Raid_values_team1 = st.multiselect("Select Empty Raid",df[df['RD'] == 'R']["Empty Raid"].unique(), key='team1_Empty_Raid_values')
            Do_or_die_Raids_values_team1 = st.multiselect("Select Do-or-die Raids",df[df['RD'] == 'R']["Do-or-die Raids"].unique(), key='team1_Do-or-die Raids_values')
            # Filter the data based on user input for Team 1
            filtered_df_team1 = Team1_Video_data(df, team1_r_no, bonus_points_values_team1, touch_points_values_team1,
                                                 Empty_Raid_values_team1,Do_or_die_Raids_values_team1)
            
            # Display filtered video links for Team 1
            with st.expander("Ahmednagar Video"):
                st.write("Ahmednagar Video Links:")
                for link in filtered_df_team1["Video Links"]:
                    st.markdown(f"[{link}]({link})")

        elif team2_r_no != "Choose":
            col1, col2 = st.columns(2)
            with col1:
                show_Team2_Scorecard_player(df,team2_r_no)
            with col2:
                st.header("Palghar Raider Video")
                team2_r_no = st.multiselect("Select Raider: ", df[df['RD'] == 'D']["R No."].unique(), key='team2_unique_r_nos')
                bonus_points_values_team2 = st.multiselect("Select Bonus Points", df[df['RD'] == 'D']["Bonus Points"].unique(), key='team2_bonus_points_values')
                touch_points_values_team2 = st.multiselect("Select Touch Points",df[df['RD'] == 'D']["Touch Points"].unique(), key='team2_touch_points_values')
                Empty_Raid_values_team2 = st.multiselect("Select Empty Raid",df[df['RD'] == 'D']["Empty Raid"].unique(), key='team2_Empty_Raid_values')
                Do_or_die_Raids_values_team2 = st.multiselect("Select Do-or-die Raids",df[df['RD'] == 'D']["Do-or-die Raids"].unique(), key='team2_Do-or-die Raids_values')

            # Filter the data based on user input for Team 2
                filtered_df_team2 = Team2_Video_data(df, team2_r_no, bonus_points_values_team2, touch_points_values_team2,
                                                     Empty_Raid_values_team2,Do_or_die_Raids_values_team2)
            
            # Display filtered video links for Team 2
                with st.expander("Palghar Video"):
                    st.write("Palghar Video Links:")
                    for link in filtered_df_team2["Video Links"]:
                        st.markdown(f"[{link}]({link})")
          

    elif page == "Attack Locations":
        selected_df = st.selectbox("Select Attacks", ["Attacks", "Attacks With Player"])
        # Filter the DataFrame based on the selection
        if selected_df == "Attacks":
            filtered_df = df[df['RD'] == 'R']
            filtered_df2 = df[df['RD'] == 'D']
        elif selected_df == "Attacks With Player":
            df['Raider Attack'] = df['Raider Attack'] + ", P: " + df['Total Players at Defence'].astype(str)
            df['Second Attack Raider'] = df['Second Attack Raider'] + ", P: " + df['Total Players at Defence'].astype(str)
            filtered_df = df[df['RD'] == 'R']
            filtered_df2 = df[df['RD'] == 'D']
            
        if team1_r_no != "Choose":
            show_team1_attack_locations(df, team1_r_no)
        elif team2_r_no != "Choose":
            show_Team2_Attack_and_Attack_Locations_Raider(df, team2_r_no)


    elif page == "Touch Points":
        if team1_r_no != "Choose":
            show_touch_points_at_player_present(df, team1_r_no)
        elif team2_r_no != "Choose":
            show_Team2_Touch_Points_at_Player_present_Raider(df, team2_r_no)


    elif page == "Do-or-Die":
        if team1_r_no != "Choose":
            show_team1_do_or_die(df, team1_r_no)
        elif team2_r_no != "Choose":
            show_do_or_die_Team2_Raider(df, team2_r_no)


    elif page == "Escape":
        if team1_r_no != "Choose":
            show_team1_raider_escape(df, team1_r_no)
        elif team2_r_no != "Choose":
            show_Team2_raider_escape_Raider(df, team2_r_no)


    elif page == "Raider Tackled":
        if team1_r_no != "Choose":
            show_team1_Raider_Tackled(df, team1_r_no)
        elif team2_r_no != "Choose":
            show_Team2_Raider_tackled(df, team2_r_no)
    
    
        

def show_Team1_Scorecard_players(df, selected_player):
    scorecard_data = Team1_Scorecard_players(df, selected_player)
    if not scorecard_data.empty:
        st.write("## Scorecard")
        st.write(scorecard_data)
    else:
        st.warning("No data available for the selected player.")

def show_Team2_Scorecard_player(df, selected_player):
    scorecard_data = Team2_Scorecard_player(df, selected_player)
    if not scorecard_data.empty:
        st.write("## Scorecard")
        st.write(scorecard_data)
    else:
        st.warning("No data available for the selected player.")


def show_team1_bonus_points_distribution(df, r_no):
    fig_successful_bonus_points, fig_unsuccessful_bonus_points = Team1_bonus_points_distribution_Raider(df, r_no)
    st.plotly_chart(fig_successful_bonus_points, use_container_width=True)
    st.plotly_chart(fig_unsuccessful_bonus_points, use_container_width=True)

def show_Team2_bonus_points_distribution_Raider(df, r_no):
    fig_successful_bonus_points, fig_unsuccessful_bonus_points = Team2_bonus_points_distribution_Raider(df, r_no)
    st.plotly_chart(fig_successful_bonus_points, use_container_width=True)
    st.plotly_chart(fig_unsuccessful_bonus_points, use_container_width=True)

def show_team1_raider_escape(df, r_no):
    Team1_raider_escape_Raider(df, r_no)

def show_Team2_raider_escape_Raider(df, r_no):
    Team2_raider_escape_Raider(df, r_no)

def show_team1_attack_locations(df, r_no):
    st.write("Select the chart type:")
    chart_type = st.selectbox("Chart Type", ["Total", "Successful","Unsuccessful"])
    Team1_Attack_and_Attack_Locations_Raider(chart_type, df, r_no)

def show_Team2_Attack_and_Attack_Locations_Raider(df, r_no):
    st.write("Select the chart type:")
    chart_type = st.selectbox("Chart Type", ["Total", "Successful","Unsuccessful"])
    Team2_Attack_and_Attack_Locations_Raider(chart_type, df, r_no)



def show_team1_do_or_die(df, r_no):
    do_or_die_Team1_Raider(df, r_no)

def show_do_or_die_Team2_Raider(df, r_no):
    do_or_die_Team2_Raider(df, r_no)



def show_touch_points_at_player_present(df, r_no):
    Touch_Points_at_Player_present_Raider(df, r_no)

    
def show_Team2_Touch_Points_at_Player_present_Raider(df, r_no):
    Team2_Touch_Points_at_Player_present_Raider(df, r_no)



def show_team1_Raider_Tackled(df,r_no):
    Team1_Raider_tackled(df,r_no)

def show_Team2_Raider_tackled(df,r_no):
    Team2_Raider_tackled(df,r_no)



def Defender_analysis(df):
    filtered_df1 = df[df['RD'] == 'D']

    all_players1 = pd.concat([
    filtered_df1['2nd Player Out'].dropna(), 
    filtered_df1['3rd Player Out'].dropna(), 
    filtered_df1['4th Player Out'].dropna(), 
    filtered_df1['Defence player No.'].dropna()])


    filtered_df2 = df[df['RD'] == 'R']

    all_players2 = pd.concat([
    filtered_df2['2nd Player Out'].dropna(), 
    filtered_df2['3rd Player Out'].dropna(), 
    filtered_df2['4th Player Out'].dropna(), 
    filtered_df2['Defence player No.'].dropna()])

    

# Get unique player numbers
    Team1_d_numbers = all_players1.unique()

    Team2_d_numbers = all_players2.unique()

    st.sidebar.write("# Team 1 Defender Number")
    team1_d_no = st.sidebar.selectbox("Select Defender Number for Team 1", ["Choose"] + list(Team1_d_numbers))

    st.sidebar.write("# Team 2 Defender Number")
    team2_d_no = st.sidebar.selectbox("Select Defender Number for Team 2", ["Choose"] + list(Team2_d_numbers))

    # Navigation sidebar
    st.sidebar.write("# Navigation")
    page = st.sidebar.radio("Go to", ["Scorecard","Defender","Defender Out by Attacks"])

    if page == "Defender":
        if team1_d_no != "Choose":
            chart_type = st.selectbox("Select Chart Type", ["Total", "Successful", "Unsuccessful"])
            show_Tackels_Team1_and_Defence_Locations2(chart_type, df, team1_d_no)

        elif team2_d_no != "Choose":
            chart_type = st.selectbox("Select Chart Type", ["Total", "Successful", "Unsuccessful"])
            show_Tackels_Team2_and_Defence_Locations_Players(chart_type, df, team2_d_no)

    elif page == "Scorecard":
        if team1_d_no != "Choose":
            col1, col2 = st.columns(2)
            with col1:
                show_Team1_Scorecard_players(df, team1_d_no)
            with col2:
                st.header("Ahmednagar Defender Video")
                team1_r_nos = st.multiselect("Select Defender: ", 
                                  df[df['RD'] == 'D'][['Defence player No.', '2nd Player Out', '3rd Player Out', '4th Player Out']].stack().unique(),
                                  key='team1_unique_r_nos')
                tackle_points_values = st.multiselect("Select Tackle Points", df[df['RD'] == 'D']["Tackle Points"].unique(), key='tackle_points_values')

                filtered_df_team1 = Team1_defender_Video_data(df, team1_r_nos, tackle_points_values)

                with st.expander("Ahmednagar Video"):
                    st.write("Ahmednagar Video Links:")
                    for link in filtered_df_team1["Video Links"]:
                        st.markdown(f"[{link}]({link})")

                    
        elif team2_d_no != "Choose":
            col1, col2 = st.columns(2)
            with col1:
                show_Team2_Scorecard_player(df,team2_d_no)
            with col2:
                st.header("Palghar Defender Video")
                team2_d_no = st.multiselect("Select Defender: ", 
                                  df[df['RD'] == 'R'][['Defence player No.', '2nd Player Out', '3rd Player Out', '4th Player Out']].stack().unique(),
                                  key='team1_unique_r_nos')
                tackle_points_values = st.multiselect("Select Tackle Points", df[df['RD'] == 'D']["Tackle Points"].unique(), key='tackle_points_values')

                filtered_df_team1 = Team2_defender_Video_data(df, team2_d_no, tackle_points_values)

                with st.expander("Palghar Video"):
                    st.write("Palghar Video Links:")
                    for link in filtered_df_team1["Video Links"]:
                        st.markdown(f"[{link}]({link})")
            
    elif page == "Defender Out by Attacks":
        if team1_d_no != "Choose":
            show_Defender_out(df,team1_d_no)

        elif team2_d_no != "Choose":
            show_Team2_Defender_out(df,team2_d_no)


def show_Team1_Scorecard_players(df, selected_player):
    scorecard_data = Team1_Scorecard_players(df, selected_player)
    if not scorecard_data.empty:
        st.write("## Scorecard")
        st.write(scorecard_data)
    else:
        st.warning("No data available for the selected player.")

def show_Team2_Scorecard_player(df, selected_player):
    scorecard_data = Team2_Scorecard_player(df, selected_player)
    if not scorecard_data.empty:
        st.write("## Scorecard")
        st.write(scorecard_data)
    else:
        st.warning("No data available for the selected player.")

def show_Tackels_Team1_and_Defence_Locations2(chart_type, df, selected_player):
    Tackels_Team1_and_Defence_Locations2(chart_type, df, selected_player)  

def show_Tackels_Team2_and_Defence_Locations_Players(chart_type, df, selected_player):
    Tackels_Team2_and_Defence_Locations_Players(chart_type, df, selected_player) 


def show_Defender_out(df,selected_player):
    Defender_out(df,selected_player)
   
def show_Team2_Defender_out(df,selected_player):
    Team2_Defender_out(df,selected_player)


def show_Player_Profile(df):

    filtered_df = df[df['RD'] == 'R']
    filtered_df2 = df[df['RD'] == 'D']

    Team1_r_numbers = filtered_df['R No.'].unique()
    Team2_r_numbers = filtered_df2['R No.'].unique()

    # Sidebar for selecting the Raider Number for Team 1
    st.sidebar.write("# Team 1 Raider Number")
    team1_r_no = st.sidebar.selectbox("Select Raider Number for Team 1", ["Choose"] + list(Team1_r_numbers))
    


    # Sidebar for selecting the Raider Number for Team 2
    st.sidebar.write("# Team 2 Raider Number")
    team2_r_no = st.sidebar.selectbox("Select Raider Number for Team 2", ["Choose"] + list(Team2_r_numbers))

    # Navigation sidebar
    st.sidebar.write("# Navigation")
    page = st.sidebar.radio("Go to", ["Scorecard","Bonus Points", "Attack Locations","Touch Points","Escape", "Do-or-Die","Raider Tackled"])

    # Conditional display based on the selected page and the selected team
    if page == "Bonus Points":
        if team1_r_no != "Choose":
            show_team1_bonus_points_distribution(df, team1_r_no)
        elif team2_r_no != "Choose":
            show_Team2_bonus_points_distribution_Raider(df, team2_r_no)

    elif page == "Scorecard":
        if team1_r_no != "Choose":
         col1, col2 = st.columns(2)
         with col1:
            show_Team1_Scorecard_players(df, team1_r_no)

         with col2:
            st.header("Ahmednagar Raider Video")
            team1_r_no = st.multiselect("Select Raider: ", df[df['RD'] == 'R']["R No."].unique(), key='team1_unique_r_nos')
            bonus_points_values_team1 = st.multiselect("Select Bonus Points", df[df['RD'] == 'R']["Bonus Points"].unique(), key='team1_bonus_points_values')
            touch_points_values_team1 = st.multiselect("Select Touch Points",df[df['RD'] == 'R']["Touch Points"].unique(), key='team1_touch_points_values')
            Empty_Raid_values_team1 = st.multiselect("Select Empty Raid",df[df['RD'] == 'R']["Empty Raid"].unique(), key='team1_Empty_Raid_values')
            Do_or_die_Raids_values_team1 = st.multiselect("Select Do-or-die Raids",df[df['RD'] == 'R']["Do-or-die Raids"].unique(), key='team1_Do-or-die Raids_values')
            # Filter the data based on user input for Team 1
            filtered_df_team1 = Team1_Video_data(df, team1_r_no, bonus_points_values_team1, touch_points_values_team1,
                                                 Empty_Raid_values_team1,Do_or_die_Raids_values_team1)
            
            # Display filtered video links for Team 1
            with st.expander("Ahmednagar Video"):
                st.write("Ahmednagar Video Links:")
                for link in filtered_df_team1["Video Links"]:
                    st.markdown(f"[{link}]({link})")

        elif team2_r_no != "Choose":
            col1, col2 = st.columns(2)
            with col1:
                show_Team2_Scorecard_player(df,team2_r_no)
            with col2:
                st.header("Palghar Raider Video")
                team2_r_no = st.multiselect("Select Raider: ", df[df['RD'] == 'D']["R No."].unique(), key='team2_unique_r_nos')
                bonus_points_values_team2 = st.multiselect("Select Bonus Points", df[df['RD'] == 'D']["Bonus Points"].unique(), key='team2_bonus_points_values')
                touch_points_values_team2 = st.multiselect("Select Touch Points",df[df['RD'] == 'D']["Touch Points"].unique(), key='team2_touch_points_values')
                Empty_Raid_values_team2 = st.multiselect("Select Empty Raid",df[df['RD'] == 'D']["Empty Raid"].unique(), key='team2_Empty_Raid_values')
                Do_or_die_Raids_values_team2 = st.multiselect("Select Do-or-die Raids",df[df['RD'] == 'D']["Do-or-die Raids"].unique(), key='team2_Do-or-die Raids_values')

            # Filter the data based on user input for Team 2
                filtered_df_team2 = Team2_Video_data(df, team2_r_no, bonus_points_values_team2, touch_points_values_team2,
                                                     Empty_Raid_values_team2,Do_or_die_Raids_values_team2)
            
            # Display filtered video links for Team 2
                with st.expander("Palghar Video"):
                    st.write("Palghar Video Links:")
                    for link in filtered_df_team2["Video Links"]:
                        st.markdown(f"[{link}]({link})")
          

    elif page == "Attack Locations":
        selected_df = st.selectbox("Select Attacks", ["Attacks", "Attacks With Player"])
        # Filter the DataFrame based on the selection
        if selected_df == "Attacks":
            filtered_df = df[df['RD'] == 'R']
            filtered_df2 = df[df['RD'] == 'D']
        elif selected_df == "Attacks With Player":
            df['Raider Attack'] = df['Raider Attack'] + ", P: " + df['Total Players at Defence'].astype(str)
            df['Second Attack Raider'] = df['Second Attack Raider'] + ", P: " + df['Total Players at Defence'].astype(str)
            filtered_df = df[df['RD'] == 'R']
            filtered_df2 = df[df['RD'] == 'D']
            
        if team1_r_no != "Choose":
            show_team1_attack_locations(df, team1_r_no)
        elif team2_r_no != "Choose":
            show_Team2_Attack_and_Attack_Locations_Raider(df, team2_r_no)


    elif page == "Touch Points":
        if team1_r_no != "Choose":
            show_touch_points_at_player_present(df, team1_r_no)
        elif team2_r_no != "Choose":
            show_Team2_Touch_Points_at_Player_present_Raider(df, team2_r_no)


    elif page == "Do-or-Die":
        if team1_r_no != "Choose":
            show_team1_do_or_die(df, team1_r_no)
        elif team2_r_no != "Choose":
            show_do_or_die_Team2_Raider(df, team2_r_no)


    elif page == "Escape":
        if team1_r_no != "Choose":
            show_team1_raider_escape(df, team1_r_no)
        elif team2_r_no != "Choose":
            show_Team2_raider_escape_Raider(df, team2_r_no)


    elif page == "Raider Tackled":
        if team1_r_no != "Choose":
            show_team1_Raider_Tackled(df, team1_r_no)
        elif team2_r_no != "Choose":
            show_Team2_Raider_tackled(df, team2_r_no)
    
    
        

def show_Team1_Scorecard_players(df, selected_player):
    scorecard_data = Team1_Scorecard_players(df, selected_player)
    if not scorecard_data.empty:
        st.write("## Scorecard")
        st.write(scorecard_data)
    else:
        st.warning("No data available for the selected player.")

def show_Team2_Scorecard_player(df, selected_player):
    scorecard_data = Team2_Scorecard_player(df, selected_player)
    if not scorecard_data.empty:
        st.write("## Scorecard")
        st.write(scorecard_data)
    else:
        st.warning("No data available for the selected player.")


def show_team1_bonus_points_distribution(df, r_no):
    fig_successful_bonus_points, fig_unsuccessful_bonus_points = Team1_bonus_points_distribution_Raider(df, r_no)
    st.plotly_chart(fig_successful_bonus_points, use_container_width=True)
    st.plotly_chart(fig_unsuccessful_bonus_points, use_container_width=True)

def show_Team2_bonus_points_distribution_Raider(df, r_no):
    fig_successful_bonus_points, fig_unsuccessful_bonus_points = Team2_bonus_points_distribution_Raider(df, r_no)
    st.plotly_chart(fig_successful_bonus_points, use_container_width=True)
    st.plotly_chart(fig_unsuccessful_bonus_points, use_container_width=True)

def show_team1_raider_escape(df, r_no):
    Team1_raider_escape_Raider(df, r_no)

def show_Team2_raider_escape_Raider(df, r_no):
    Team2_raider_escape_Raider(df, r_no)

def show_team1_attack_locations(df, r_no):
    st.write("Select the chart type:")
    chart_type = st.selectbox("Chart Type", ["Total", "Successful","Unsuccessful"])
    Team1_Attack_and_Attack_Locations_Raider(chart_type, df, r_no)

def show_Team2_Attack_and_Attack_Locations_Raider(df, r_no):
    st.write("Select the chart type:")
    chart_type = st.selectbox("Chart Type", ["Total", "Successful","Unsuccessful"])
    Team2_Attack_and_Attack_Locations_Raider(chart_type, df, r_no)



def show_team1_do_or_die(df, r_no):
    do_or_die_Team1_Raider(df, r_no)

def show_do_or_die_Team2_Raider(df, r_no):
    do_or_die_Team2_Raider(df, r_no)



def show_touch_points_at_player_present(df, r_no):
    Touch_Points_at_Player_present_Raider(df, r_no)

    
def show_Team2_Touch_Points_at_Player_present_Raider(df, r_no):
    Team2_Touch_Points_at_Player_present_Raider(df, r_no)



def show_team1_Raider_Tackled(df,r_no):
    Team1_Raider_tackled(df,r_no)

def show_Team2_Raider_tackled(df,r_no):
    Team2_Raider_tackled(df,r_no)



if __name__ == '__main__':
    main()
