#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install basketball-reference-scraper


# In[2]:


import basketball_reference_scraper as bc
from basketball_reference_scraper.teams import get_roster
from basketball_reference_scraper.players import get_game_logs
     


# In[4]:


pip install requests


# In[5]:


pip install beautifulsoup


# In[7]:


import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_nba_per_game_stats(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    per_game_stats = []
    table = soup.find('table', id='per_game_stats')
    rows = table.find_all('tr', class_=lambda x: x != 'thead')

    for row in rows:
        player_stats = {}
        for data in row.find_all('td'):
            player_stats[data['data-stat']] = data.text

        per_game_stats.append(player_stats)

    return per_game_stats

# Example usage:
url = "https://www.basketball-reference.com/leagues/NBA_2023_per_game.html"
per_game_stats = get_nba_per_game_stats(url)

# Create a DataFrame
df = pd.DataFrame(per_game_stats)

# Print the DataFrame
print(df)


# In[10]:


for col in df.columns:
    print(col)


# In[11]:


from basketball_reference_scraper.teams import get_roster, get_team_stats, get_opp_stats, get_roster_stats, get_team_misc


# In[12]:


get_roster('GSW', 2022)

