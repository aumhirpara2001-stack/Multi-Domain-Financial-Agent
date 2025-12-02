#!/usr/bin/env python3
"""
Download all PDFs from the FinanceBench GitHub repository.
"""

import os
import requests
from pathlib import Path
import time
from tqdm import tqdm

def download_pdfs():
    """Download all PDFs from FinanceBench GitHub repository."""

    # GitHub API endpoint to list files in the pdfs directory
    api_url = "https://api.github.com/repos/patronus-ai/financebench/contents/pdfs"

    # Output directory
    output_dir = Path(__file__).parent.parent / "data" / "test_files" / "finance-bench-pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching PDF list from GitHub...")

    # Get list of files
    response = requests.get(api_url)
    response.raise_for_status()
    files = response.json()

    # Filter for PDF files
    pdf_files = [f for f in files if f['name'].endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files")

    # Check which files already exist
    existing_files = set(os.listdir(output_dir))
    pdf_files_to_download = [f for f in pdf_files if f['name'] not in existing_files]

    print(f"Already have {len(pdf_files) - len(pdf_files_to_download)} PDFs")
    print(f"Need to download {len(pdf_files_to_download)} PDFs")

    if not pdf_files_to_download:
        print("All PDFs already downloaded!")
        return

    # Download each PDF
    for i, file_info in enumerate(tqdm(pdf_files_to_download, desc="Downloading PDFs")):
        file_name = file_info['name']
        download_url = file_info['download_url']
        output_path = output_dir / file_name

        # Download with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(download_url, timeout=30)
                response.raise_for_status()

                with open(output_path, 'wb') as f:
                    f.write(response.content)

                break  # Success

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\nRetrying {file_name} (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(2)
                else:
                    print(f"\nFailed to download {file_name}: {e}")

        # Small delay to avoid rate limiting
        if i < len(pdf_files_to_download) - 1:
            time.sleep(0.1)

    print(f"\n✓ Download complete! Total PDFs in directory: {len(os.listdir(output_dir))}")

if __name__ == "__main__":
    download_pdfs()
