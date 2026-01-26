
import os
import urllib.request
import zipfile
import shutil
import sys

URL = "https://github.com/rordenlab/dcm2niix/releases/download/v1.0.20240202/dcm2niix_win.zip"
DEST_ZIP = "dcm2niix.zip"
EXTRACT_TO = "."

def install():
    print(f"Downloading dcm2niix from {URL}...")
    try:
        urllib.request.urlretrieve(URL, DEST_ZIP)
        print("Download complete.")
        
        print("Extracting...")
        with zipfile.ZipFile(DEST_ZIP, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_TO)
            
        print("Extraction complete.")
        
        # Cleanup zip
        if os.path.exists(DEST_ZIP):
            os.remove(DEST_ZIP)
            
        # Verify
        exe_path = os.path.join(EXTRACT_TO, "dcm2niix.exe")
        if os.path.exists(exe_path):
            print(f"Success! dcm2niix.exe is located at {os.path.abspath(exe_path)}")
        else:
            print("Error: dcm2niix.exe not found after extraction.")
            
    except Exception as e:
        print(f"Failed to install: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install()
