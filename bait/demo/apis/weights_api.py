import os
import requests
import zipfile


def download_and_extract_zip(url, dest_folder):
    # Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)
    
    # Temporary zip file location
    zip_path = os.path.join(dest_folder, os.path.basename(url))
    weights_dir = os.path.join(dest_folder, os.path.basename(url).split(".")[0])

    if os.path.exists(weights_dir):
        print("Weights already exist.")
        return
    
    # Download the zip file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print("Download complete.")
    else:
        print(f"Failed to download: {response.status_code}")
        return

    # Extract the zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_folder)
    print("Extraction complete.")

    # Clean up the zip file
    os.remove(zip_path)
