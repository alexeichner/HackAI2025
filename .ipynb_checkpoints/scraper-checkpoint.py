import os
import json
import requests
from bs4 import BeautifulSoup

def get_replay_links(base_url, pages=100):
    """Scrapes replay links from the first 'pages' pages."""
    replay_links = []
    for page in range(1, pages + 1):
        url = f"{base_url}&page={page}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve page {page}")
            continue
        
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a', href=True):
            if "gen9vgc2025regg-" in link['href']:
                replay_links.append("https://replay.pokemonshowdown.com" + link['href'])
    
    return replay_links

def download_replay_data(links, output_dir="replays"): 
    """Downloads JSON data for each replay link."""
    os.makedirs(output_dir, exist_ok=True)
    
    for link in links:
        json_url = link + ".json"
        replay_id = link.split("-")[-1]
        output_path = os.path.join(output_dir, f"{replay_id}.json")
        
        response = requests.get(json_url)
        if response.status_code == 200:
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(response.json(), file, indent=4)
            print(f"Saved {replay_id}.json")
        else:
            print(f"Failed to download {json_url}")

def main():
    base_url = "https://replay.pokemonshowdown.com/?format=gen9vgc2025regg&sort=rating"
    replay_links = get_replay_links(base_url, pages=100)
    print(f"Found {len(replay_links)} replays.")
    download_replay_data(replay_links)

if __name__ == "__main__":
    main()
