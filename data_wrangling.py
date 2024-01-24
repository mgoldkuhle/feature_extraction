import os
import shutil
import pandas as pd

# Load the CSV file
load_path = 'C:/Users/mjgoldkuhle/ownCloud/LUMC/data/selected_features'
growths = pd.read_csv(os.path.join(load_path, 't2_3d_all_growth.csv'))

# Iterate over each row in the CSV
for index, row in growths.iterrows():
    patient_path = os.path.join('//vf-DataSafe/DataSafe$/div2/radi/Brughoek_predict_1255/01_followup_cleanedup/followup_part1', row['patient'])

    # only copy scans from the latest date
    dates = [folder for folder in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, folder))]
    latest_date = max(dates)

    # Construct the source and destination paths
    source_path = os.path.join(patient_path, latest_date)
    img_destination_path = os.path.join('D:/repos/WORC/WORCTutorial/Data/t2schwannoma', row['patient'], 'image.nii.gz')
    mask_destination_path = os.path.join('D:/repos/WORC/WORCTutorial/Data/t2schwannoma', row['patient'], 'mask.nii.gz')
    # create directories if they don't exist
    os.makedirs(os.path.dirname(os.path.join('D:/repos/WORC/WORCTutorial/Data/t2schwannoma', row['patient'])), exist_ok=True)

    # Find the files ending with t2.nii.gz and t2_seg.nii.gz
    for file in os.listdir(source_path):
        if file.endswith('t2.nii.gz'):
            # Copy the image file
            shutil.copy(os.path.join(source_path, file), img_destination_path)
        elif file.endswith('t2_seg.nii.gz'):
            # Copy the mask file and rename it to mask.nii.gz
            shutil.copy(os.path.join(source_path, file), mask_destination_path)
            