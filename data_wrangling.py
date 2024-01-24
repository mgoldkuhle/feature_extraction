import os
import shutil
import pandas as pd

# Load the CSV file
load_path = 'C:/Users/mjgoldkuhle/ownCloud/LUMC/data/selected_features'
growths = pd.read_csv(os.path.join(load_path, 'WORCschwannoma.csv'))

# Iterate over each row in the CSV
for index, row in growths.iterrows():
    patient_id = row['Patient']
    patient_path = os.path.join('//vf-DataSafe/DataSafe$/div2/radi/Brughoek_predict_1255/01_followup_cleanedup/followup_part1', patient_id)

    # only copy scans from the first date
    dates = [folder for folder in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, folder))]
    first_date = min(dates)
    # sequence of this patient
    sequence = row['sequence']
    # check if folder contains sequence scan, otherwise take next date and check again
    while not os.path.exists(os.path.join(patient_path, first_date, patient_id.split('_')[1] + '_' + first_date + '_' + sequence + '.nii.gz')):
        print(os.path.join(patient_path, first_date, patient_id + '_' + first_date + '_' + sequence +'.nii.gz'))
        dates.remove(first_date)
        first_date = min(dates)

    # Construct the source and destination paths
    source_path = os.path.join(patient_path, first_date)
    img_destination_path = os.path.join('D:/repos/WORC/WORCTutorial/Data/schwannoma', row['Patient'], 'image.nii.gz')
    mask_destination_path = os.path.join('D:/repos/WORC/WORCTutorial/Data/schwannoma', row['Patient'], 'mask.nii.gz')
    # create directories if they don't exist
    os.makedirs(os.path.dirname(os.path.join('D:/repos/WORC/WORCTutorial/Data/schwannoma', row['Patient'])), exist_ok=True)

    # Find the files ending with <sequence>.nii.gz and <sequence>_seg.nii.gz
    for file in os.listdir(source_path):
        if file.endswith(f'{sequence}.nii.gz'):
            # remove old image and mask files if they exist
            if os.path.exists(img_destination_path):
                os.remove(img_destination_path)
            # Copy the image file
            shutil.copy(os.path.join(source_path, file), img_destination_path)
        elif file.endswith(f'{sequence}_seg.nii.gz'):
            # remove old image and mask files if they exist
            if os.path.exists(mask_destination_path):
                os.remove(mask_destination_path)
            # Copy the mask file and rename it to mask.nii.gz
            shutil.copy(os.path.join(source_path, file), mask_destination_path)
            