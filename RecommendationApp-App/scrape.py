import requests
from bs4 import BeautifulSoup
import csv
import time

for i in range(2018,2025):
    url = f'http://www.cricmetric.com/ipl/ranks/{i}'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Open the CSV file in write mode
    with open(f'player_rankings_{i}.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row
        writer.writerow(['Rank', 'Player', 'Team', 'RAA', 'Wins', 'EFscore', 'Salary', 'Value'])
        
        # Find the table
        table = soup.find('table', class_='scoretable')
        
        # Find all table rows
        rows = table.find_all('tr')
        
        # Loop through each row and extract data
        for row in rows[1:]:  # Skipping the first row as it contains headers
            # Extract data from each cell in the row
            cells = row.find_all('td')
            
            # Extract text from each cell and strip whitespace
            rank = cells[0].text.strip()
            player = cells[1].text.strip()
            team = cells[2].text.strip()
            raa = cells[3].text.strip()
            wins = cells[4].text.strip()
            efscore = cells[5].text.strip()
            salary = cells[6].text.strip()
            value = cells[7].text.strip()
            
            # Write the extracted data to the CSV file
            writer.writerow([rank, player, team, raa, wins, efscore, salary, value])

            #time.sleep(3)

    print("Data has been written to player_rankings.csv")
