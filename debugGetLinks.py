from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By 
from webdriver_manager.chrome import ChromeDriverManager
import time

def get_replay_links(base_url, pages=100):
    """Scrapes replay links from Pokemon Showdown using Selenium."""
    options = Options()
    options.add_argument("--headless")  # Run without opening a browser
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    replay_links = []
    
    for page in range(1, pages + 1):
        url = f"{base_url}&page={page}"
        driver.get(url)
        time.sleep(3)  # Wait for JavaScript to load

        # Find all <a> tags
        links = driver.find_elements(By.TAG_NAME, "a")
        
        # Extract replay links
        for link in links:
            href = link.get_attribute("href")
            if href and "gen9vgc2025regg-" in href:  # Ensure correct format
                replay_links.append(href)

        print(f"Page {page}: Found {len(replay_links)} total replays so far.")

    driver.quit()
    return replay_links

base_url = "https://replay.pokemonshowdown.com/?format=gen9vgc2025regg&sort=rating"
replay_links = get_replay_links(base_url, pages=2)  # Test with 2 pages

print(f"Total replays found: {len(replay_links)}")
print(replay_links[:5])  # Print first 5 links to verify
