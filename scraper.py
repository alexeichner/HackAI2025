import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# 1️⃣ FUNCTION: Scrape Replay Links
def get_replay_links(base_url, pages=1):
    """Scrapes replay links from Pokemon Showdown using Selenium."""
    options = Options()
    options.add_argument("--headless")  # Run without opening a browser
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    replay_links = []
    
    for page in range(pages, pages + 30):
        url = f"{base_url}&page={page}"
        driver.get(url)
        time.sleep(3)  # Wait for JavaScript to load

        # Find all <a> tags
        links = driver.find_elements(By.TAG_NAME, "a")
        
        # Extract replay links
        for link in links:
            href = link.get_attribute("href")
            #if href and "gen9vgc2025regg-" in href:  # Ensure correct format
            if href and "gen9vgc2024regg-" in href:
                replay_links.append(href)

        print(f"Page {page}: Found {len(replay_links)} total replays so far.")

    driver.quit()
    return replay_links

# 2️⃣ FUNCTION: Download JSON Replay Data
def download_replay_json(replay_links, save_dir="replaysRegH"):
    """Downloads replay JSON files from provided links and saves them locally."""
    
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, link in enumerate(replay_links):
        json_url = f"{link}.json"
        replay_id = link.split("-")[-1]  # Extract replay ID from URL
        save_path = os.path.join(save_dir, f"{replay_id}.json")

        try:
            response = requests.get(json_url, timeout=10)
            response.raise_for_status()  # Raise error for bad status codes
            
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(response.text)

            print(f"[{i+1}/{len(replay_links)}] Saved: {save_path}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {json_url}: {e}")

        time.sleep(1)  # Be polite & avoid rate limiting

# 3️⃣ RUN THE SCRAPER & DOWNLOADER
if __name__ == "__main__":
    #base_url = "https://replay.pokemonshowdown.com/?format=gen9vgc2025regg&sort=rating"
    base_url = "https://replay.pokemonshowdown.com/?format=gen9vgc2024regg&sort=rating"

    print("\n🔍 Scraping Replay Links...")
    replay_links = get_replay_links(base_url, pages=1)  # Adjust pages as needed

    print(f"\n✅ Found {len(replay_links)} replays! Now downloading JSON data...\n")
    download_replay_json(replay_links)

    print("\n🎉 Done! All replays have been downloaded.")
