import os
import urllib.request
import re
import sys
import zipfile
from datetime import datetime

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHESS_DATA_DIR = os.path.join(DATA_DIR, "chess")
SHOGI_DATA_DIR = os.path.join(DATA_DIR, "shogi")
GO_DATA_DIR = os.path.join(DATA_DIR, "go")
OTHELLO_DATA_DIR = os.path.join(DATA_DIR, "othello")

# Ensure directories exist
os.makedirs(CHESS_DATA_DIR, exist_ok=True)
os.makedirs(SHOGI_DATA_DIR, exist_ok=True)
os.makedirs(GO_DATA_DIR, exist_ok=True)
os.makedirs(OTHELLO_DATA_DIR, exist_ok=True)

# User-Agent to avoid 403 Forbidden responses
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

def download_file(url, save_path):
    """Downloads a file with a progress bar and caching."""
    if os.path.exists(save_path):
        print(f"  [Cache Hit] Already downloaded: {os.path.basename(save_path)}")
        return True

    print(f"  [Downloading] {url} -> {save_path}")
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req) as response:
            total_size = int(response.info().get("Content-Length", 0))
            block_size = 1024 * 8
            downloaded = 0
            
            with open(save_path, "wb") as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    downloaded += len(buffer)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        sys.stdout.write(f"\r    Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)")
                        sys.stdout.flush()
            if total_size > 0:
                print()  # Newline after progress
        print(f"  [Success] Saved to {save_path}")
        return True
    except Exception as e:
        print(f"\n  [Error] Failed to download {url}: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False

def download_chess_pgn(player_name="Carlsen"):
    """Downloads PGN file (via ZIP) for a specific elite player from PGN Mentor."""
    pgn_path = os.path.join(CHESS_DATA_DIR, f"{player_name}.pgn")
    if os.path.exists(pgn_path):
        print(f"  [Cache Hit] Already extracted PGN: {player_name}.pgn")
        return True

    zip_path = os.path.join(CHESS_DATA_DIR, f"{player_name}.zip")
    url = f"https://www.pgnmentor.com/players/{player_name}.zip"
    
    print(f"\n=== Fetching Chess PGN for {player_name} ===")
    if download_file(url, zip_path):
        try:
            print(f"  [Extracting] {zip_path} -> {CHESS_DATA_DIR}")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(CHESS_DATA_DIR)
            print("  [Success] PGN extracted successfully.")
            # Remove zip file to clean up
            os.remove(zip_path)
            return True
        except Exception as e:
            print(f"  [Error] Failed to extract {zip_path}: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return False
    return False

def download_shogi_daily(date_str=None, max_games=100):
    """
    Crawls recent daily logs from Floodgate and downloads individual CSA files.
    date_str format: YYYY/MM/DD (e.g. '2026/05/23')
    """
    if date_str is None:
        date_str = "2026/05/23"

    print(f"\n=== Fetching Shogi Floodgate CSA logs for date: {date_str} (Max: {max_games}) ===")
    
    base_url = f"http://wdoor.c.u-tokyo.ac.jp/shogi/x/{date_str}/"
    
    try:
        req = urllib.request.Request(base_url, headers=HEADERS)
        with urllib.request.urlopen(req) as response:
            html = response.read().decode("utf-8")
    except Exception as e:
        print(f"  [Error] Failed to connect to Floodgate index: {e}")
        return False

    csa_files = re.findall(r'href="([^"]+\.csa)"', html)
    if not csa_files:
        print("  [Warning] No CSA game logs found on this date.")
        return False

    print(f"  [Found] {len(csa_files)} game records on Floodgate.")
    
    date_subfolder = date_str.replace("/", "_")
    shogi_save_dir = os.path.join(SHOGI_DATA_DIR, date_subfolder)
    os.makedirs(shogi_save_dir, exist_ok=True)

    downloaded_count = 0
    for csa_filename in csa_files[:max_games]:
        csa_url = base_url + csa_filename
        save_path = os.path.join(shogi_save_dir, csa_filename)
        
        if download_file(csa_url, save_path):
            downloaded_count += 1

    print(f"=== Complete Shogi Downloads: {downloaded_count} games saved ===")
    return downloaded_count > 0

def download_go_sgf(max_games=100):
    """Downloads professional AlphaGo SGF matches from CWI Go Database."""
    print(f"\n=== Fetching Go AlphaGo SGF logs (Max: {max_games}) ===")
    
    base_url = "http://homepages.cwi.nl/~aeb/go/games/games/AlphaGo/"
    
    # We will fetch specific matches: Fan Hui (5 games), Lee Sedol (5 games), NewYear2017 (up to 90 games)
    categories = {
        "FanHui": [f"FanHui/{i}.sgf" for i in range(1, 6)],
        "LeeSedol": [f"LeeSedol/{i}.sgf" for i in range(1, 6)],
        "NewYear2017": [f"NewYear2017/T{str(i).zfill(2)}.sgf" for i in range(1, 61)]
    }
    
    downloaded_count = 0
    for category, file_list in categories.items():
        cat_save_dir = os.path.join(GO_DATA_DIR, category)
        os.makedirs(cat_save_dir, exist_ok=True)
        
        for rel_path in file_list:
            if downloaded_count >= max_games:
                break
                
            filename = os.path.basename(rel_path)
            url = base_url + rel_path
            save_path = os.path.join(cat_save_dir, filename)
            
            if download_file(url, save_path):
                downloaded_count += 1
                
    print(f"=== Complete Go Downloads: {downloaded_count} SGF files saved ===")
    return downloaded_count > 0

def download_othello_pgn(max_years=3):
    """Downloads elite Othello WTHOR tournament PGN files."""
    print(f"\n=== Fetching Othello WTHOR Tournament PGNs ===")
    
    base_url = "https://raw.githubusercontent.com/MartinMSPedersen/othello-games/master/pgn/with_python_code/"
    
    # We download the most recent years (2024, 2023, 2022) to get massive high-quality datasets
    years = [2024, 2023, 2022][:max_years]
    
    downloaded_count = 0
    for year in years:
        filename = f"WTH_{year}.pgn"
        url = base_url + filename
        save_path = os.path.join(OTHELLO_DATA_DIR, filename)
        
        if download_file(url, save_path):
            downloaded_count += 1
            
    print(f"=== Complete Othello Downloads: {downloaded_count} PGN files saved ===")
    return downloaded_count > 0

if __name__ == "__main__":
    # Test downloader
    download_chess_pgn("Carlsen")
    download_shogi_daily("2026/05/23", max_games=2)
    download_go_sgf(max_games=5)
    download_othello_pgn(max_years=1)
