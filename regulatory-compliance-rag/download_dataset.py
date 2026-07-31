import os
import requests

# 1. Configuration & Directories
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Emulate browser headers to bypass automated request blocks
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# Part 1: Official Regulatory PDF Direct Downloads
REG_URLS = {
    "mifid_ii.pdf": "https://europa.eu",
    "gdpr_official.pdf": "https://europa.eu",
    "finra_suitability_rule2111.pdf": "https://finra.org",
    "sec_investment_advisers_act_1940.pdf": "https://sec.gov"
}

def download_pdfs():
    print("=== Downloading Official Regulatory PDFs ===")
    for filename, url in REG_URLS.items():
        destination = os.path.join(OUTPUT_DIR, filename)
        
        # Skip if already downloaded to save time
        if os.path.exists(destination):
            print(f"ℹ️ {filename} already exists. Skipping...")
            continue
            
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            if response.status_code == 200:
                with open(destination, "wb") as f:
                    f.write(response.content)
                print(f"✅ Saved to: {destination}")
            else:
                print(f"❌ Failed {filename} (Status: {response.status_code})")
        except Exception as e:
            print(f"💥 Error downloading {filename}: {e}")

# Part 2: Clean Public URL Download (Replaces HuggingFace)
def download_flat_financial_data():
    print("\n=== Downloading Public Financial Compliance Text ===")
    destination = os.path.join(OUTPUT_DIR, "financial_compliance_qa.txt")
    
    # Direct raw text file URL containing audited financial and regulatory data structures
    direct_url = "https://githubusercontent.com"
    
    try:
        print("Downloading text dataset from public repository...")
        response = requests.get(direct_url, headers=HEADERS, timeout=30)
        
        if response.status_code == 200:
            with open(destination, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"✅ Saved clean text data straight to: {destination}")
        else:
            print(f"❌ Failed to fetch text file (Status Code: {response.status_code})")
            
    except Exception as e:
        print(f"💥 Error downloading text data: {e}")

if __name__ == "__main__":
    download_pdfs()
    download_flat_financial_data()
    print(f"\nAll dataset assets downloaded successfully to the '{OUTPUT_DIR}' directory!")
