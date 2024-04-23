import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class MixedLinearModel:
    def __init__(self, csv_file = 'C:/Users/mjgoldkuhle/ownCloud/LUMC/data/selected_features/extracted_features.csv'):
        self.data = pd.read_csv(csv_file)
        self.data['date'] = self.data['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
        # Remove outliers
        self.data = self.data[(self.data['original_shape_VoxelVolume'] <= 500) & (self.data['date'].dt.year >= 2002)]

        # Add group column to differentiate between patients' intra and extrameatal tumors
        self.data['label_patient'] = self.data['label'].astype(str) + '_' + self.data['patient'].astype(str)


    def predict_last_value(self):
        # Group the data by 'label_patient' and sort by 'date'
        grouped = self.data.sort_values(by='date').groupby('label_patient')

        for name, group in grouped:
            n = len(group)
            if n > 1:
                # Split the 'original_shape_VoxelVolume' column into features and target
                features = group['original_shape_VoxelVolume'].iloc[:n-1].values.reshape(-1, 1)
                target = group['original_shape_VoxelVolume'].iloc[n-1:n].values

                # Fit a linear regression model
                model = LinearRegression()
                model.fit(features, target)

                # Predict the last value
                predicted = model.predict(features[-1].reshape(-1, 1))

                # Compare the predicted value with the actual value
                print(f'Patient: {name}, Actual: {target}, Predicted: {predicted[0]}, MSE: {mean_squared_error([target], [predicted[0]])}')

    def perform_mixed_linear_model(self):
        # Sort data by 'date'
        self.data.sort_values(by=['label_patient', 'date'], inplace=True)

        # Use data up to the second last timepoint to fit the model
        train_data = self.data.groupby('label_patient').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)

        # Define the covariates
        covariates = ['original_shape_Elongation + original_shape_Flatness + original_shape_LeastAxisLength + original_shape_MajorAxisLength + original_shape_Maximum2DDiameterColumn + original_shape_Maximum2DDiameterRow + original_shape_Maximum2DDiameterSlice + original_shape_Maximum3DDiameter + original_shape_MeshVolume + original_shape_MinorAxisLength + original_shape_Sphericity + original_shape_SurfaceArea + original_shape_SurfaceVolumeRatio + original_shape_VoxelVolume + original_firstorder_10Percentile + original_firstorder_90Percentile + original_firstorder_Energy + original_firstorder_Entropy + original_firstorder_InterquartileRange + original_firstorder_Kurtosis + original_firstorder_Maximum + original_firstorder_MeanAbsoluteDeviation + original_firstorder_Mean + original_firstorder_Median + original_firstorder_Minimum + original_firstorder_Range + original_firstorder_RobustMeanAbsoluteDeviation + original_firstorder_RootMeanSquared + original_firstorder_Skewness + original_firstorder_TotalEnergy + original_firstorder_Uniformity + original_firstorder_Variance + original_glcm_Autocorrelation + original_glcm_ClusterProminence + original_glcm_ClusterShade + original_glcm_ClusterTendency + original_glcm_Contrast + original_glcm_Correlation + original_glcm_DifferenceAverage + original_glcm_DifferenceEntropy + original_glcm_DifferenceVariance + original_glcm_Id + original_glcm_Idm + original_glcm_Idmn + original_glcm_Idn + original_glcm_Imc1 + original_glcm_Imc2 + original_glcm_InverseVariance + original_glcm_JointAverage + original_glcm_JointEnergy + original_glcm_JointEntropy + original_glcm_MCC + original_glcm_MaximumProbability + original_glcm_SumAverage + original_glcm_SumEntropy + original_glcm_SumSquares + original_gldm_DependenceEntropy + original_gldm_DependenceNonUniformity + original_gldm_DependenceNonUniformityNormalized + original_gldm_DependenceVariance + original_gldm_GrayLevelNonUniformity + original_gldm_GrayLevelVariance + original_gldm_HighGray']

        # Fit the model
        formula = 'original_shape_VoxelVolume ~ original_shape_Elongation + original_shape_Flatness + original_shape_LeastAxisLength + original_shape_MajorAxisLength + original_shape_Maximum2DDiameterColumn + original_shape_Maximum2DDiameterRow + original_shape_Maximum2DDiameterSlice + original_shape_Maximum3DDiameter + original_shape_MeshVolume + original_shape_MinorAxisLength + original_shape_Sphericity + original_shape_SurfaceArea + original_shape_SurfaceVolumeRatio + original_shape_VoxelVolume + original_firstorder_10Percentile + original_firstorder_90Percentile + original_firstorder_Energy + original_firstorder_Entropy + original_firstorder_InterquartileRange + original_firstorder_Kurtosis + original_firstorder_Maximum + original_firstorder_MeanAbsoluteDeviation + original_firstorder_Mean + original_firstorder_Median + original_firstorder_Minimum + original_firstorder_Range + original_firstorder_RobustMeanAbsoluteDeviation + original_firstorder_RootMeanSquared + original_firstorder_Skewness + original_firstorder_TotalEnergy + original_firstorder_Uniformity + original_firstorder_Variance + original_glcm_Autocorrelation + original_glcm_ClusterProminence + original_glcm_ClusterShade + original_glcm_ClusterTendency + original_glcm_Contrast + original_glcm_Correlation + original_glcm_DifferenceAverage + original_glcm_DifferenceEntropy + original_glcm_DifferenceVariance + original_glcm_Id + original_glcm_Idm + original_glcm_Idmn + original_glcm_Idn + original_glcm_Imc1 + original_glcm_Imc2 + original_glcm_InverseVariance + original_glcm_JointAverage + original_glcm_JointEnergy + original_glcm_JointEntropy + original_glcm_MCC + original_glcm_MaximumProbability + original_glcm_SumAverage + original_glcm_SumEntropy + original_glcm_SumSquares + original_gldm_DependenceEntropy + original_gldm_DependenceNonUniformity + original_gldm_DependenceNonUniformityNormalized + original_gldm_DependenceVariance + original_gldm_GrayLevelNonUniformity + original_gldm_GrayLevelVariance + original_gldm_HighGray'
        model = smf.mixedlm(formula, train_data, groups=train_data['label_patient'])
        self.result = model.fit()

        # Use the model to predict the 'original_shape_VoxelVolume' at the last timepoint
        test_data = self.data.groupby('label_patient').apply(lambda x: x.iloc[-1]).reset_index(drop=True)
        test_data['predicted_VoxelVolume'] = self.result.predict(test_data)

    def plot_results(self):
        fig, ax = plt.subplots()
        ax.plot(self.result.fittedvalues, label='Fitted values')
        ax.plot(self.result.resid, label='Residuals')
        ax.legend()
        plt.show()
        
    def plot_voxel_volume(self):
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='date', y='original_shape_VoxelVolume', hue='label_patient', palette='bright', data=self.data, errorbar=None, legend=None)
        plt.ylim(
            self.data['original_shape_VoxelVolume'].min(), 
            self.data['original_shape_VoxelVolume'].max()
            
            )
        plt.show()

    def calculate_mean_growth_rate(self):
        # Sort data by 'date'
        self.data.sort_values(by=['label_patient', 'date'], inplace=True)

        # Calculate difference in 'original_shape_VoxelVolume' and 'date' between consecutive measurements
        self.data['volume_diff'] = self.data.groupby('label_patient')['original_shape_VoxelVolume'].diff()
        self.data['date_diff'] = self.data.groupby('label_patient')['date'].diff().dt.days

        # Calculate growth rate and replace infinities with NaN
        self.data['growth_rate'] = self.data['volume_diff'] / self.data['date_diff']
        self.data['growth_rate'].replace([np.inf, -np.inf], np.nan, inplace=True)

        # Calculate and return mean growth rate
        return self.data['growth_rate'].mean(), self.data['growth_rate'].std()


model = MixedLinearModel()
model.plot_voxel_volume()
model.calculate_mean_growth_rate()
model.predict_last_value()
