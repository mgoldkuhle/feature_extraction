## calculate growth rates for each patients based on the diameters of schwannoma tumors over time and create prediction label file for WORC

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import VS diameters
load_path = 'C:/Users/mjgoldkuhle/ownCloud/LUMC/data/selected_features'
diameters_t1ce_1 = pd.read_csv(os.path.join(load_path, 't1ce_3d_part1.csv'))
diameters_t1ce_1['sequence'] = 't1ce'
diameters_t1ce_2 = pd.read_csv(os.path.join(load_path, 't1ce_3d_part2.csv'))
diameters_t1ce_2['sequence'] = 't1ce'
diameters_t1ce_3 = pd.read_csv(os.path.join(load_path, 't1ce_3d_part3.csv'))
diameters_t1ce_3['sequence'] = 't1ce'
diameters_t2 = pd.read_csv(os.path.join(load_path, 't2_3d_part1.csv'))
diameters_t2['sequence'] = 't2'
diameters = pd.concat([diameters_t1ce_1, diameters_t2, diameters_t1ce_2, diameters_t1ce_3], ignore_index=True)

# segmentation code produced some 0 measures if there's no scan or a bad segmentation
diameters = diameters[diameters['max_diameter'] != 0]
# remove patients with less than 2 measurements in either sequence
diameters = diameters.groupby(['patient', 'sequence']).filter(lambda x: len(x) >= 2)
# if there are t1ce measurements for a patient, remove t2 measurements. t1ce seem to be more accurate
filtered_df = diameters.groupby('patient').apply(lambda x: x[x['sequence'] == 't1ce'] if ('t1ce' in x['sequence'].values and 't2' in x['sequence'].values) else x).reset_index(drop=True)
diameters = filtered_df
# remove patients with faulty measurements by patient ID after manual inspection
faulty_patients = ['id_20040107']
diameters = diameters[~diameters['patient'].isin(faulty_patients)]

print('Number of patients with two or more measures: {}'.format(len(diameters['patient'].unique())))
# 49 patients with two or more measures. 36 patients with three or more measures.

# parse date and transform to days since 01.01.1995
diameters['date'] = pd.to_datetime(diameters['date'], format='%Y%m%d')
diameters['days'] = (diameters['date'] - pd.to_datetime('1995-01-01')).dt.days

# plot diameters over time per patient in one plot, where each patient is a different color
plt.figure()
for patient in diameters['patient'].unique():
    patient_df = diameters[diameters['patient'] == patient]
    plt.plot(patient_df['days'], patient_df['max_diameter'], label=patient)
# plt.legend()
plt.xlabel('Days since 01.01.1995')
plt.ylabel('Max diameter [mm]')
plt.title('Max diameter over time per patient')
plt.show()


# use linear interpolation to interpolate values daily between available measurements
diameters_interpol = pd.DataFrame()
for patient in diameters['patient'].unique():
    patient_df = diameters[diameters['patient'] == patient]
    days = np.arange(patient_df['days'].min(), patient_df['days'].max())
    patient_df_interpol = pd.DataFrame({'patient': patient, 'days': days})
    # merge on patient and days to get the max diameter for each day
    patient_df_interpol = patient_df_interpol.merge(patient_df, on=['patient', 'days'], how='left')
    # fill missing values with linear interpolation
    patient_df_interpol['max_diameter'] = patient_df_interpol['max_diameter'].interpolate(method='linear')
    diameters_interpol = diameters_interpol.append(patient_df_interpol)

# plot diameters over time per patient in one plot, where each patient is a different color
plt.figure()
for patient in diameters_interpol['patient'].unique():
    patient_df = diameters_interpol[diameters_interpol['patient'] == patient]
    plt.plot(patient_df['days'], patient_df['max_diameter'], label=patient)
plt.legend()
plt.xlabel('Days since 01.01.1995')
plt.ylabel('Max diameter [mm]')
plt.title('Max diameter over time per patient')
plt.show()


# calculate growth for each patient
growths = {'Patient': [], 'growth': []}
for patient in diameters_interpol['patient'].unique():
    patient_df = diameters_interpol[diameters_interpol['patient'] == patient]
    # growth rate as the difference between the max diameter at the last day and the max diameter 365 days before
    # growth = patient_df['max_diameter'].iloc[-1] - patient_df['max_diameter'].iloc[-365]
    # growth rate as the difference between the max diameter at the last day and the max diameter at the first day
    growth = patient_df['max_diameter'].iloc[-1] - patient_df['max_diameter'].iloc[0]
    growths['Patient'].append(patient)
    growths['growth'].append(growth)
growths = pd.DataFrame(growths)

# plot histogram of growths
plt.figure()
plt.hist(growths['growth'], bins=40)
plt.xlabel('Growth rate [mm]')
plt.ylabel('Number of patients')
plt.title('Histogram of growth rates')
plt.show()

# plot growth curves of patients with growth rates above 2 mm
plt.figure()
for patient in growths[growths['growth'] > 2]['Patient']:
    patient_df = diameters_interpol[diameters_interpol['patient'] == patient]
    plt.plot(patient_df['days'], patient_df['max_diameter'], label=patient)
plt.legend()
plt.xlabel('Days since 01.01.1995')
plt.ylabel('Max diameter [mm]')
plt.title('Max diameter over time per patient')
plt.show()

# save diameters
diameters.to_csv(os.path.join(load_path, 'schwannoma_diameters.csv'), index=False)

# save csv with patient, growth rate, days between first and last measurement and boolean for growth rate above 2 mm and sequence
growths['days'] = (diameters.groupby('patient')['days'].max() - diameters.groupby('patient')['days'].min()).values
growths['above_2mm'] = growths['growth'] > 2
growths['sequence'] = diameters.groupby('patient')['sequence'].first().values
growths.to_csv(os.path.join(load_path, 'schwannoma_growths.csv'), index=False)
print('Saved growth rates to following file: ' + os.path.join(load_path, 'WORCschwannoma.csv'))


