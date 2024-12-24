import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
from bs4 import BeautifulSoup
from io import BytesIO

st.set_page_config(layout='wide',initial_sidebar_state="expanded")

def clean_string(str):
    char='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890:,\\-+ '
    str = list(str)
    new_str =""
    for i in str: 
        if i in char: 
            new_str += i
        elif i == "'" or i==".":
            new_str += ""
        else: 
            new_str += " "

    return new_str

###Sidebar

analysis_mode = st.sidebar.radio('Mode',options=["Season wise analysis", "Player Career"])

if (analysis_mode == "Season wise analysis"):
    st.title("NBA Player Stats Explorer")
    st.markdown("""
        This app performs web scraping to gather NBA player statistics and generates insightful plots to analyze and draw meaningful conclusions from the data.!  
        * **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/).
""")

    st.sidebar.header("User Input features")
    selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950, 2025))))

    @st.cache_data
    def load_data(year): 
        url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
        html = pd.read_html(url, header=0)
        df = html[0]
        playerstats = df.fillna(0).drop(['Rk', 'GS'], axis='columns')[:-1]
        playerstats.set_index("Player", inplace=True)
        return playerstats[:-1]

    playerstats = load_data(selected_year)
    sorted_unique_team = sorted([str(i) for i in playerstats.Team.unique() if isinstance(i, str)])
    selected_team = st.sidebar.multiselect('Team', sorted_unique_team)

    # Position selection
    unique_position = ['C', 'PF', 'SF', 'PG', 'SG']
    selected_position = st.sidebar.multiselect('Position', unique_position)

    # Sorting selection
    sorting_options = {"Points": "PTS", "Player Name": "Player", "Age": "Age", "Field Goal": "FG%","3 Pointers":"3P","2 Pointers":"2P", "Field Goal%": "FG%", "3 Pointer %": "3P%", "2 Pointer %":"2P%","Effective Field Goal%": "eFG%","Free Throw":"FT" , "Free Throw%": "FT%", "Offensive Rebounds": "ORB",
    "Defensive Rebounds": "DRB",
    "Total Rebounds": "TRB",
    "Assists": "AST",
    "Steals": "STL",
    "Blocks": "BLK",
    "Turnovers": "TOV",
    "Personal Fouls": "PF"}
    selected_sort = st.sidebar.selectbox('Sort By:(descending)', sorting_options.keys())

    # Filter data based on selected team and position
    if len(selected_team) == 0 and len(selected_position) == 0:
        df_selected_team = playerstats
    elif len(selected_position) == 0:
        df_selected_team = playerstats[playerstats.Team.isin(selected_team)]
    elif len(selected_team) == 0:
        df_selected_team = playerstats[playerstats.Pos.isin(selected_position)]
    else: 
        df_selected_team = playerstats[(playerstats.Team.isin(selected_team)) & playerstats.Pos.isin(selected_position)]

    st.header('Display Player Stats of Selected Team(s)')
    st.write(f'Data dimension: {df_selected_team.shape[0]} rows and {df_selected_team.shape[1]} columns')


    st.write(df_selected_team)
 
    # Function to create CSV download link
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV file</a>'

    st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

    st.text("""Pos: Position, G: Games, GS: Games Started, MP: Minutes Played, FG: Field Goals, FGA: Field Goal Attempts, FG%: Field Goal Percentage""")
            
    st.text(""" 3P: Three-Point Field Goals, 3PA: Three-Point Attempts, 3P%: Three-Point Percentage, 2P: Two-Point Field Goals, 2PA: Two-Point Attempts, 2P%: Two-Point Percentage""")
    st.text(""" eFG%: Effective Field Goal Percentage, FT: Free Throws, FTA: Free Throw Attempts, FT%: Free Throw Percentage, ORB: Offensive Rebounds""")
    st.text("""DRB: Defensive Rebounds, TRB: Total Rebounds, AST: Assists, STL: Steals, BLK: Blocks, TOV: Turnovers, PF: Personal Fouls, PTS:Points
""")

   #General Analysis
    if st.button('General Analysis'):
        st.subheader("General Analysis:")
        fig, ax = plt.subplots(3,2,figsize=(10,14))
        plt.subplots_adjust(hspace=1)
        sns.kdeplot(df_selected_team['PTS'].astype('float'),ax=ax[0,0])
        ax[0,0].set_title("General Distribution of Points")

        sns.kdeplot(df_selected_team['3P'].astype('float'),ax=ax[0,1])
        ax[0,1].set_title("General Distribution of 3 Pointers")
        ax[0,1].set_xlabel("3 Pointers")

        sns.kdeplot(df_selected_team['2P'].astype('float'),ax=ax[1,0])
        ax[1,0].set_title("General Distribution of 2 Pointers")
        ax[1,0].set_xlabel("2 Pointers")

        sns.lineplot(x=df_selected_team['Age'], y=df_selected_team['PTS'],ax=ax[1,1])
        ax[1,1].set_title("Age vs Points")

        sns.scatterplot(x=df_selected_team['eFG%'],y=df_selected_team['PTS'],ax=ax[2,0])
        ax[2,0].set_title("PTS vs effective FG%")

        sns.scatterplot(x=df_selected_team["MP"],y=df_selected_team["PTS"],ax=ax[2,1])
        ax[2,1].set_title("Minutes Played vs Points")
        st.pyplot(fig)
        




##Player analaysis
elif (analysis_mode == "Player Career"):
    @st.cache_data
    def load_player_names():
        player_names = pd.read_html("https://en.wikipedia.org/wiki/List_of_NBA_All-Stars")[1]['Player'].tolist()

        for index,string in enumerate(player_names):
            if string[-1].lower() not in 'abcdefghijklmnopqrstuvwxyz' :
                player_names[index]=string[:-1]

        return player_names

    player_names = load_player_names()




    player_selected = st.sidebar.selectbox("Player Name",options=player_names)

    player_selected = clean_string(player_selected).lower().split()
    if len(player_selected[-1]) <2:
        player_selected = player_selected[:-1]
    player_exact_name = []
    temp=""
    if len(player_selected) >2 and len(player_selected[0])<2 : 
        player_exact_name.append(player_selected[0]+player_selected[1])
        player_exact_name.append(player_selected[-1])
    else: 
        player_exact_name.append(player_selected[0])
        player_exact_name.append(player_selected[-1])
   
    url = "https://www.basketball-reference.com/players/" +str(player_exact_name[1][0])+"/" +str(player_exact_name[1][:5]) + str(player_exact_name[0][:2])+ "01.html"
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        page_content = response.text
    else:
        print("Failed to retrieve the webpage.")
        exit()

    # Step 2: Parse the content using BeautifulSoup
    soup = BeautifulSoup(page_content, 'html.parser')

    # Step 3: Extract specific details
    # For example, extracting all <h1> tags (or any other tag you want)
    para = soup.find_all('p')
    heading = soup.find_all('h1')
    img_tags = soup.find_all('img')
    img_urls=[img['src'] for img in img_tags if 'src' in img.attrs]
    print(type(img_tags[0]))
    print(img_urls[0])


    # Step 4: Store the extracted details into a list

    p_list = [clean_string(p.get_text(strip=True)) for p in para]

    st.title(heading[0].get_text(strip=True))

    try:
        img_response = requests.get(img_urls[1])
        st.image(BytesIO(img_response.content))
    except:
        st.write()


    if p_list[0].startswith("Pronunciation"): 
        p_list.pop(0)

    list_sub_header = p_list[0].split()
    print(list_sub_header)

    twitter_handle = list_sub_header[-2].split(":")[-1]
    insta_handle = list_sub_header[-1].split(":")[-1]
    st.subheader(" ".join(list_sub_header[:-2]))
    st.markdown(f"""
    **Twitter**: [{twitter_handle}](https://twitter.com/{twitter_handle})
    **Instagram**: [{insta_handle}](https://instagram.com/{insta_handle})
    """)
    for i in range(1,12): 
        st.text(' '.join(p_list[i].split()))


    st.header("Regular Season Games Overview:")
    numeric_columns = ['PTS','eFG%','FG%','ORB','DRB','AST','STL','BLK','TOV','PF','2P','3P',"TRB","3P%","2P%","FT%"]
    # @st.cache_data
    def load_player_data(url_url):
        df = pd.read_html(url_url)
        #remove the last row of dataframe and the empty rows
        b=0
        for l in range(0,4):
            if set(numeric_columns).issubset(df[l].columns):
                b=l
                break
        reg_df = df[b][:-1].dropna(how='all')
        playoff_df = df[b+1][:-1].dropna(how='all')
        return reg_df,playoff_df



    data = list(load_player_data(url))

    if url == "https://www.basketball-reference.com/players/d/duranke01.html":
        data[0]=data[0].drop(12)
        st.write(data[0])

    for u in range(0,2):
        data[u].dropna(how='all')
        sp_col =numeric_columns.copy()
        sp_col.extend(["Pos","Season"])
        for col_name in sp_col:
            if col_name in data[u].columns:
                data[u] = data[u].dropna(subset=[f"{col_name}"])
        data[u]=data[u][~(data[u].iloc[:,2:].map(lambda x: str(x).startswith("Did"))).all(axis=1)]
        for col_name in numeric_columns:
            if col_name in data[u].columns:
                data[u][[f"{col_name}"]] = data[u][[f"{col_name}"]].astype('float')
            else:
                data[u][f'{col_name}']=0
        data[u].set_index("Season", inplace=True)
        


    st.dataframe(data[0])

    st.header("Playoffs:")
    st.dataframe(data[1])

    st.subheader("General Stats:")
    all_points = pd.concat([data[0]['PTS'].astype("float"),data[1]["PTS"].astype("float")],axis=0,join='outer')
    mean_points=round(all_points.mean(),2)
    std_var_points = round(all_points.std(),2)

    st.markdown(f"""
**Mean Points**: {mean_points} &nbsp; **Median Points**:{round(data[0]["PTS"].median() *0.5+ data[1]["PTS"].median()*0.5)}  
**95% Confiedence Interval**:[{round(mean_points - 2*std_var_points,2)},{round(mean_points + 2*std_var_points,2)}]  
  **Field Goal effeciency** ::**Mean**:{round(data[0]["FG%"].mean()*0.5 + data[1]["FG%"].mean() * 0.5,2)} \t **Median**:{round(data[0]["FG%"].median()*0.5 + data[1]["FG%"].median() * 0.5,2)}  
**Effective Field Goal effeciency** ::**Mean**:{round(data[0]["eFG%"].mean()*0.5 + data[1]["eFG%"].mean() * 0.5,2)}  \t **Median**: {round(data[0]["eFG%"].median()*0.5 + data[1]["eFG%"].median() * 0.5,2)}  
**3 Pointer effeciency**::**Mean**:{round(data[0]["3P%"].mean()*0.5 + data[1]["3P%"].mean() * 0.5,2)}  \t **Median**:{round(data[0]["3P%"].mean()*0.5 + data[1]["3P%"].mean() * 0.5,2)}  
**2 Pointer effeciency**::**Mean**:{data[0]["2P%"].mean()*0.5 + data[1]["2P%"].mean() * 0.5}  \t **Median**:{round(data[0]["2P%"].mean()*0.5 + data[1]["2P%"].mean() * 0.5,2)}  
**Free Throw effeciency**:: **Mean**:{round(data[0]["FT%"].mean()*0.5 + data[1]["FT%"].mean() * 0.5,2)}  \t **Median**:{round(data[0]["FT%"].mean()*0.5 + data[1]["FT%"].mean() * 0.5,2)}  
""")

    st.subheader("Regular Game Season Analysis vs Playoff Season Analysis:")
    fig, axes = plt.subplots(10, 2, figsize=(10, 50))
    plt.subplots_adjust(hspace=1)

    try:
        for alpha in range(0,2):
            # 1st plot
            data[alpha]['ORB_P'] = data[alpha]['ORB'] / (data[alpha]['TRB'])
            data[alpha]['DRB_P'] = data[alpha]['DRB'] / (data[alpha]["TRB"])
            axes[0, alpha].pie([data[alpha]["ORB_P"].mean(), data[alpha]["DRB_P"].mean()], labels=["Offensive rebound", "Defensive Rebound"], shadow=True, explode=(0.1, 0), autopct="%.1f%%")



            # 2nd plot
            data[alpha]['temp_tot'] = data[alpha]['AST'] + data[alpha]['STL'] + data[alpha]['BLK'] + data[alpha]['TOV']
            lst = [np.array(data[alpha]['AST'] / data[alpha]["temp_tot"]).mean(),
                np.array(data[alpha]['STL'] / data[alpha]['temp_tot']).mean(),
                np.array(data[alpha]['BLK'] / data[alpha]['temp_tot']).mean(), np.array(data[alpha]["TOV"] / data[alpha]["temp_tot"]).mean()]
            axes[1, alpha].pie(lst, labels=["Assists", "Steals", "Blocks", "Turnovers"], explode=(0.05, 0.1, 0.15, 0), autopct="%.1f%%")



            # 3rd plot
            selected_col = ['ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PF']
            sns.heatmap(data[alpha][selected_col].T, annot=False, cmap="coolwarm", xticklabels=data[alpha].index, yticklabels=selected_col, ax=axes[2, alpha], linecolor="white", linewidths=1)
            axes[2, alpha].set_title("Heatmap")



            # 4th plot
            data[alpha]['2P_part'] = data[alpha]['2P'] / (data[alpha]['2P'] + data[alpha]['3P'])
            data[alpha]['3P_part'] = data[alpha]['3P'] / (data[alpha]["2P"] + data[alpha]["3P"])
            axes[3, alpha].pie([data[alpha]["2P_part"].mean(), data[alpha]["3P_part"].mean()], labels=["2-Pointer", "3-Pointer"], shadow=True, explode=(0.1, 0), autopct="%1.1f%%")


            # 5th plot
            axes[4, alpha].bar(x=data[alpha].index.str[-2:], height=data[alpha]['PTS'], width=0.5, linewidth=0.5, edgecolor="black")
            axes[4, alpha].set_title("Season vs Points")


            # 6th plot
            team_wise_data = data[alpha].groupby("Team")['PTS'].apply(list)
            axes[5, alpha].boxplot(team_wise_data, patch_artist=True, boxprops=dict(facecolor='lightblue', color='darkblue'),
                            whiskerprops=dict(color="green"), flierprops=dict(marker="o", markerfacecolor="red", markersize=5),
                            labels=team_wise_data.index)
            axes[5, alpha].axhline(y=data[alpha]['PTS'].mean(), color="black", linestyle="--", label="Mean Points")
            axes[5, alpha].set_title("Team wise Points analysis")

            # 7th plot
            team_wise_data = data[alpha].groupby("Pos")['PTS'].apply(list)
            axes[6, alpha].boxplot(team_wise_data, patch_artist=True, boxprops=dict(facecolor='lightpink', color='red'),
                            whiskerprops=dict(color="blue"), flierprops=dict(marker="o", markerfacecolor="red", markersize=5),
                            labels=team_wise_data.index)
            axes[6, alpha].axhline(y=data[alpha]['PTS'].mean(), color="black", linestyle="--", label="Mean Points")
            axes[6, alpha].set_title("Position wise Points analysis")

            #8th plot
            axes[7,alpha].set_title("Shooting effeciency over Seasons")
            sns.kdeplot(data[alpha]['FG%'],ax=axes[7,alpha])

            #9th plot
            axes[8,alpha].set_title("3Pointer effeciency and 2Pointer effeciency")
            sns.kdeplot(data[alpha]["3P%"],color="blue",label="3 Pointer", ax=axes[8,alpha])
            sns.kdeplot(data[alpha]["2P%"],color="red",label="2 Pointer",ax=axes[8,alpha])
            axes[8,alpha].set_xlabel("3P% and 2P%")
            axes[8,alpha].legend()

            #10th plot
            axes[9,alpha].set_title("Free Throw effeciency")
            sns.kdeplot(data[alpha]['FT%'],color="blue",ax=axes[9,alpha])

        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)
    except:
        st.write("Incomplete information to plot")
