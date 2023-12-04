import os
from radiomics import featureextractor
from pprint import pprint
from collections import OrderedDict
import pandas as pd

class FeatureExtractor:
    def __init__(
            self, 
            data_path = os.path.join('//vf-DataSafe/DataSafe$/div2/radi/Brughoek_predict_1255/01_followup_cleanedup/followup_part1/'), 
            out_path = 'C:/Users/mjgoldkuhle/ownCloud/LUMC/data/selected_features',
            save_failed = True
                ):
        self.data_path = data_path
        self.out_path = out_path
        self.patients = self.get_patients()
        self.patient_dates = self.get_patient_dates()
        self.df = pd.DataFrame()
        self.save_failed = save_failed
        self.failed = {}

    def get_patients(self):
        folder_list = os.listdir(self.data_path)
        return [folder.split('_')[1] for folder in folder_list]

    def get_patient_dates(self):
        patient_dates = {patient: os.listdir(os.path.join(self.data_path, 'id_' + patient)) for patient in self.patients}
        return patient_dates

    def extract_features(self):

        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.loadParams(os.path.join(self.out_path, 'RadiomicsLogicParams.json'))

        for patient, dates in self.patient_dates.items():    
        
            for date in dates:
                for sequence in ['t1ce', 't2']:
                    # define path to image and segmentation mask
                    image_path = os.path.join(self.data_path, 'id_' + patient, date, patient + '_' + date + '_' + sequence + '.nii.gz')
                    mask_path = os.path.join(self.data_path, 'id_' + patient, date, patient + '_' + date + '_' + sequence + '_seg.nii.gz')

                    print('Extracting ' + sequence + ' features for patient ' + patient + ' on date ' + date)

                    # execute feature extraction and save results to csv
                    # if extraction fails, skip to next patient
                    try:
                        extracted_1 = extractor.execute(image_path, mask_path, 1)
                        extracted_2 = extractor.execute(image_path, mask_path, 2)
                        # Add patient and date to the beginning of the dictionaries
                        extracted_1 = OrderedDict([('patient', patient), ('date', date), ('sequence', sequence), ('label', 1)] + list(extracted_1.items()))
                        extracted_2 = OrderedDict([('patient', patient), ('date', date), ('sequence', sequence), ('label', 2)] + list(extracted_2.items()))
                        print('Feature extraction (' + sequence + ') succeeded for patient ' + patient + ' on date ' + date + ':')
                        # pprint(extracted)

                        # Convert the extracted features to a DataFrame
                        df_1 = pd.DataFrame.from_dict(extracted_1, orient='index').transpose()
                        df_2 = pd.DataFrame.from_dict(extracted_2, orient='index').transpose()

                        # Concatenate the two DataFrames
                        df_temp = pd.concat([df_1, df_2], ignore_index=True)
                        self.df = pd.concat([self.df, df_temp], ignore_index=True)

                    except:
                        print('Feature extraction (' + sequence + ') failed for patient ' + patient + ' on date ' + date)
                        if patient in self.failed:
                            self.failed[patient].append((date, sequence))
                        else:
                            self.failed[patient] = [(date, sequence)]
                        continue

    def save_features(self):
        feature_path = os.path.join(self.out_path, 'extracted_features.csv')
        self.df.to_csv(feature_path, index=False)

        if self.save_failed:
            failed_path = os.path.join(self.out_path, 'failed_extractions.csv')
            failed_df = pd.DataFrame.from_dict(self.failed, orient='index')
            failed_df.reset_index(inplace=True)
            failed_df = failed_df.rename(columns={failed_df.columns[0]: 'patient'})
            failed_df.to_csv(failed_path, index=True)

if __name__ == '__main__':

    fe = FeatureExtractor()
    fe.extract_features()
    fe.save_features()