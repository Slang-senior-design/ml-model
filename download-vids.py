import numpy as np
import pandas as pd
import urllib.request
import shutil
import sys
import os

c = sys.argv[1].lower()
df = pd.read_excel("./clean-dataset.xlsx")
data = df[df['Main New Gloss.1'] == c] #rows of the data to use

for index, row in data.iterrows():
    folder = row["Main New Gloss.1"].lower()
    if(not os.path.exists(folder)):
        os.makedirs(folder)
    url = row["vid_links"]
    file_name = url.split("/")[-1]
    print("./"+folder+"/"+file_name)
    # Download the file from `url` and save it locally under `file_name`:
    with urllib.request.urlopen(url) as response, open("./"+folder+"/"+file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)