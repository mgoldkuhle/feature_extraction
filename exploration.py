import pandas as pd
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('C:/Users/mjgoldkuhle/ownCloud/LUMC/data/selected_features/extracted_features.csv')

# filter groups where both T1ce and T2 exist and only keep the extrameatal measures
df_filtered = df.groupby(['patient', 'date']).filter(lambda x: 't1ce' in x['sequence'].values and 't2' in x['sequence'].values)
df_filtered = df_filtered[df_filtered['label'] == 2] # 2: extrameatal measures

# Separate the T1ce and T2 sequences
t1ce = df_filtered[df_filtered['sequence'] == 't1ce']
t2 = df_filtered[df_filtered['sequence'] == 't2']

# compute t-test for maximum 3D diameter in mm
t_stat, p_val = ttest_rel(t1ce['original_shape_Maximum3DDiameter'], t2['original_shape_Maximum3DDiameter'])
print(f'T1ce vs T2: t-statistic = {t_stat}, p-value = {p_val}')

# mean difference between T1ce and T2 and std
diff = t1ce['original_shape_Maximum3DDiameter'].values - t2['original_shape_Maximum3DDiameter'].values
std_diff = diff.std()
mean_diff = diff.mean()
print(f'T1ce vs T2: mean difference = {round(mean_diff, 4)}mm, std = {round(std_diff, 4)}mm, max = {round(diff.max(), 4)}mm, \n' \
      f'min_t1ce = {round(t1ce["original_shape_Maximum3DDiameter"].min(), 2)}mm, max_t1ce = {round(t1ce["original_shape_Maximum3DDiameter"].max(), 2)}mm, \n' \
      f'min_t2 = {round(t2["original_shape_Maximum3DDiameter"].min(), 2)}mm, max_t2 = {round(t2["original_shape_Maximum3DDiameter"].max(), 2)}mm')

# plot the difference between T1ce and T2
def plot_histogram(data, bins=50, xlab='mm', title=''):
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=bins)
    plt.xlabel(xlab)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()

plot_histogram(diff, title='Difference between T1ce and T2', xlab='Difference in mm')

plot_histogram(t1ce['original_shape_Maximum3DDiameter'], title='T1ce max. Diameter', xlab='mm')

plot_histogram(t2['original_shape_Maximum3DDiameter'], title='T2 max. Diameter', xlab='mm')

# identify the patients where the difference exceeds 5mm
t1ce.reset_index(drop=True, inplace=True)
t2.reset_index(drop=True, inplace=True)
outliers = t1ce[abs(t1ce['original_shape_Maximum3DDiameter'] - t2['original_shape_Maximum3DDiameter']) >= 5]
df_filtered[df_filtered['patient'].isin(outliers['patient'])]['original_shape_Maximum3DDiameter']

# remove patients that have less than 3 unique timepoints available
patient_counts = df_filtered.groupby('patient')['date'].nunique()
patients_to_remove = patient_counts[patient_counts < 3].index
df_filtered2 = df_filtered[~df_filtered['patient'].isin(patients_to_remove)]

# how many unique patients are left?
df_filtered2['patient'].nunique()
