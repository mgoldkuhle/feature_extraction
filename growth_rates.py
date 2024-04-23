## calculate growth rates for each patients based on the diameter or volume measures of schwannoma tumors over time and create prediction label file for WORC
## calculate growth rates for each patients based on the diameter or volume measures of schwannoma tumors over time and create prediction label file for WORC

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log10

# which measures to use as target?
target_measure = 'total_volume'  # 'max_diameter' or 'total_volume'
from math import log10

# which measures to use as target?
target_measure = 'total_volume'  # 'max_diameter' or 'total_volume'

# import VS measures
# import VS measures
load_path = 'C:/Users/mjgoldkuhle/ownCloud/LUMC/data/selected_features'
measures_t1ce_1 = pd.read_csv(os.path.join(load_path, 't1ce_3d_part1.csv'))
measures_t1ce_1['sequence'] = 't1ce'
measures_t1ce_2 = pd.read_csv(os.path.join(load_path, 't1ce_3d_part2.csv'))
measures_t1ce_2['sequence'] = 't1ce'
measures_t1ce_3 = pd.read_csv(os.path.join(load_path, 't1ce_3d_part3.csv'))
measures_t1ce_3['sequence'] = 't1ce'
# measures_t2 = pd.read_csv(os.path.join(load_path, 't2_3d_part1.csv'))
# measures_t2['sequence'] = 't2'
measures = pd.concat([measures_t1ce_1, measures_t1ce_2, measures_t1ce_3], ignore_index=True)
measures_t1ce_1 = pd.read_csv(os.path.join(load_path, 't1ce_3d_part1.csv'))
measures_t1ce_1['sequence'] = 't1ce'
measures_t1ce_2 = pd.read_csv(os.path.join(load_path, 't1ce_3d_part2.csv'))
measures_t1ce_2['sequence'] = 't1ce'
measures_t1ce_3 = pd.read_csv(os.path.join(load_path, 't1ce_3d_part3.csv'))
measures_t1ce_3['sequence'] = 't1ce'
# measures_t2 = pd.read_csv(os.path.join(load_path, 't2_3d_part1.csv'))
# measures_t2['sequence'] = 't2'
measures = pd.concat([measures_t1ce_1, measures_t1ce_2, measures_t1ce_3], ignore_index=True)

# VSDiameter produces a max diameter of 0 for patients with no extrameatal or intrameatal part
measures = measures[(measures[target_measure] != 0 & measures[target_measure].notna())]
# VSDiameter produces a max diameter of 0 for patients with no extrameatal or intrameatal part
measures = measures[(measures[target_measure] != 0 & measures[target_measure].notna())]
# remove patients with less than 2 measurements in either sequence
measures = measures.groupby(['patient', 'sequence']).filter(lambda x: len(x) >= 2)
measures = measures.groupby(['patient', 'sequence']).filter(lambda x: len(x) >= 2)
# if there are t1ce measurements for a patient, remove t2 measurements. t1ce seem to be more accurate
filtered_df = measures.groupby('patient').apply(lambda x: x[x['sequence'] == 't1ce'] if ('t1ce' in x['sequence'].values and 't2' in x['sequence'].values) else x).reset_index(drop=True)
measures = filtered_df
filtered_df = measures.groupby('patient').apply(lambda x: x[x['sequence'] == 't1ce'] if ('t1ce' in x['sequence'].values and 't2' in x['sequence'].values) else x).reset_index(drop=True)
measures = filtered_df
# remove patients with faulty measurements by patient ID after manual inspection
faulty_patients = ['id_20040107', 'id_65395101']
measures = measures[~measures['patient'].isin(faulty_patients)]
faulty_patients = ['id_20040107', 'id_65395101']
measures = measures[~measures['patient'].isin(faulty_patients)]

# diameter: 138 patients with two or more measures. 86 patients with three or more measures.
# volume: 223 patients with two or more measures. 151 patients with three or more measures.
print('Number of patients with two or more measures: {}'.format(len(measures['patient'].unique())))
# diameter: 138 patients with two or more measures. 86 patients with three or more measures.
# volume: 223 patients with two or more measures. 151 patients with three or more measures.
print('Number of patients with two or more measures: {}'.format(len(measures['patient'].unique())))

# parse date and transform to days since 01.01.1995
measures['date'] = pd.to_datetime(measures['date'], format='%Y%m%d')
measures['days'] = (measures['date'] - pd.to_datetime('1995-01-01')).dt.days
measures['date'] = pd.to_datetime(measures['date'], format='%Y%m%d')
measures['days'] = (measures['date'] - pd.to_datetime('1995-01-01')).dt.days

# plot measures over time per patient in one plot, where each patient is a different color
# plot measures over time per patient in one plot, where each patient is a different color
plt.figure()
for patient in measures['patient'].unique():
    patient_df = measures[measures['patient'] == patient]
    plt.plot(patient_df['days'], patient_df[target_measure].apply(log10), label=patient)
for patient in measures['patient'].unique():
    patient_df = measures[measures['patient'] == patient]
    plt.plot(patient_df['days'], patient_df[target_measure].apply(log10), label=patient)
# plt.legend()
plt.xlabel('Days since 01.01.1995')
plt.ylabel(f'log10({target_measure} [mm])')
plt.title(f'{target_measure} over time per patient')
plt.ylabel(f'log10({target_measure} [mm])')
plt.title(f'{target_measure} over time per patient')
plt.show()



# # use linear interpolation to interpolate values daily between available measurements
# measures_interpol = pd.DataFrame()
# for patient in measures['patient'].unique():
#     patient_df = measures[measures['patient'] == patient]
#     days = np.arange(patient_df['days'].min(), patient_df['days'].max())
#     patient_df_interpol = pd.DataFrame({'patient': patient, 'days': days})
#     # merge on patient and days to get the max diameter for each day
#     patient_df_interpol = patient_df_interpol.merge(patient_df, on=['patient', 'days'], how='left')
#     # fill missing values with linear interpolation
#     patient_df_interpol[target_measure] = patient_df_interpol[target_measure].interpolate(method='linear')
#     measures_interpol = measures_interpol.append(patient_df_interpol)

# # use linear interpolation to interpolate values daily between available measurements
# measures_interpol = pd.DataFrame()
# for patient in measures['patient'].unique():
#     patient_df = measures[measures['patient'] == patient]
#     days = np.arange(patient_df['days'].min(), patient_df['days'].max())
#     patient_df_interpol = pd.DataFrame({'patient': patient, 'days': days})
#     # merge on patient and days to get the max diameter for each day
#     patient_df_interpol = patient_df_interpol.merge(patient_df, on=['patient', 'days'], how='left')
#     # fill missing values with linear interpolation
#     patient_df_interpol[target_measure] = patient_df_interpol[target_measure].interpolate(method='linear')
#     measures_interpol = measures_interpol.append(patient_df_interpol)

# # plot measures over time per patient in one plot, where each patient is a different color
# plt.figure()
# for patient in measures_interpol['patient'].unique():
#     patient_df = measures_interpol[measures_interpol['patient'] == patient]
#     plt.plot(patient_df['days'], patient_df[target_measure], label=patient)
# plt.legend()
# plt.xlabel('Days since 01.01.1995')
# plt.ylabel(f'{target_measure} [mm]')
# plt.title(f'{target_measure} over time per patient')
# plt.show()
# # plot measures over time per patient in one plot, where each patient is a different color
# plt.figure()
# for patient in measures_interpol['patient'].unique():
#     patient_df = measures_interpol[measures_interpol['patient'] == patient]
#     plt.plot(patient_df['days'], patient_df[target_measure], label=patient)
# plt.legend()
# plt.xlabel('Days since 01.01.1995')
# plt.ylabel(f'{target_measure} [mm]')
# plt.title(f'{target_measure} over time per patient')
# plt.show()


# calculate growth for each patient
growths = {'Patient': [], 'growth': []}
for patient in measures['patient'].unique():
    patient_df = measures[measures['patient'] == patient]
for patient in measures['patient'].unique():
    patient_df = measures[measures['patient'] == patient]
    # growth rate as the difference between the max diameter at the last day and the max diameter 365 days before
    # growth = patient_df[target_measure].iloc[-1] - patient_df[target_measure].iloc[-365]
    # growth = patient_df[target_measure].iloc[-1] - patient_df[target_measure].iloc[-365]
    # growth rate as the difference between the max diameter at the last day and the max diameter at the first day
    growth = patient_df[target_measure].iloc[-1] - patient_df[target_measure].iloc[0]
    growth = patient_df[target_measure].iloc[-1] - patient_df[target_measure].iloc[0]
    growths['Patient'].append(patient)
    growths['growth'].append(growth)
growths = pd.DataFrame(growths)

# plot histogram of growths
plt.figure()
plt.hist(growths['growth'], bins=40)
plt.xlabel('Growth [mm]')
plt.xlabel('Growth [mm]')
plt.ylabel('Number of patients')
plt.title(f'Histogram of {target_measure} growth')
plt.title(f'Histogram of {target_measure} growth')
plt.show()


# # lowest growth patients etc
# print('Patient with lowest growth rate: ' + growths[growths['growth'] == growths['growth'].min()]['Patient'].values[0])
# # growth curve of patient with lowest growth rate
# low_patient = growths[growths['growth'] == growths['growth'].max()]['Patient'].values[0]
# low_patient_df = measures[measures['patient'] == low_patient]
# plt.figure()
# plt.plot(low_patient_df['days'], low_patient_df[target_measure])
# plt.xlabel('Days since 01.01.1995')
# plt.ylabel('Max diameter [mm]')
# plt.title('Max diameter over time for patient with lowest growth rate')
# plt.show()
# # print the measures per time point for this patient
# print(low_patient_df[['date', target_measure]])



# # lowest growth patients etc
# print('Patient with lowest growth rate: ' + growths[growths['growth'] == growths['growth'].min()]['Patient'].values[0])
# # growth curve of patient with lowest growth rate
# low_patient = growths[growths['growth'] == growths['growth'].max()]['Patient'].values[0]
# low_patient_df = measures[measures['patient'] == low_patient]
# plt.figure()
# plt.plot(low_patient_df['days'], low_patient_df[target_measure])
# plt.xlabel('Days since 01.01.1995')
# plt.ylabel('Max diameter [mm]')
# plt.title('Max diameter over time for patient with lowest growth rate')
# plt.show()
# # print the measures per time point for this patient
# print(low_patient_df[['date', target_measure]])


# plot growth curves of patients with growth rates above 2 mm
plt.figure()
for patient in growths[growths['growth'] > 1000]['Patient']:
    patient_df = measures[measures['patient'] == patient]
    plt.plot(patient_df['days'], patient_df[target_measure], label=patient)
for patient in growths[growths['growth'] > 1000]['Patient']:
    patient_df = measures[measures['patient'] == patient]
    plt.plot(patient_df['days'], patient_df[target_measure], label=patient)
plt.legend()
plt.xlabel('Days since 01.01.1995')
plt.ylabel(f'{target_measure} [mm]')
plt.title(f'{target_measure} over time per patient')
plt.ylabel(f'{target_measure} [mm]')
plt.title(f'{target_measure} over time per patient')
plt.show()

# save measures
# parse date back to string for WORC
measures['date'] = measures['date'].dt.strftime('%Y%m%d')
measures.to_csv(os.path.join(load_path, 'schwannoma_measures_t1ce.csv'), index=False)
# save measures
# parse date back to string for WORC
measures['date'] = measures['date'].dt.strftime('%Y%m%d')
measures.to_csv(os.path.join(load_path, 'schwannoma_measures_t1ce.csv'), index=False)

# save csv with patient, growth rate, days between first and last measurement and boolean for growth rate above 2 mm and sequence
growths['days'] = (measures.groupby('patient')['days'].max() - measures.groupby('patient')['days'].min()).values
growths['days'] = (measures.groupby('patient')['days'].max() - measures.groupby('patient')['days'].min()).values
growths['above_2mm'] = growths['growth'] > 2
# growths['sequence'] = measures.groupby('patient')['sequence'].first().values
growths.to_csv(os.path.join(load_path, 'schwannoma_growths_t1ce.csv'), index=False)
print('Saved growth rates to following file: ' + os.path.join(load_path, 'schwannoma_growths_t1ce.csv'))
# growths['sequence'] = measures.groupby('patient')['sequence'].first().values
growths.to_csv(os.path.join(load_path, 'schwannoma_growths_t1ce.csv'), index=False)
print('Saved growth rates to following file: ' + os.path.join(load_path, 'schwannoma_growths_t1ce.csv'))


