import pickle
import os

class HealthInsurance(object):
    def __init__(self):
        self.home_path = os.path.dirname(os.path.abspath(__file__))
        self.age_scaler = pickle.load(open(self.home_path + '/../src/features/age_scaler.pkl', 'rb'))
        self.vintage_scaler = pickle.load(open(self.home_path + '/../src/features/vintage_scaler.pkl', 'rb'))
        self.annual_premium_scaler = pickle.load(open(self.home_path + '/../src/features/annual_premium_scaler.pkl', 'rb'))
        self.gender_scaler = pickle.load(open(self.home_path + '/../src/features/gender_scaler.pkl', 'rb'))
        self.vehicle_damage_scaler = pickle.load( open(self.home_path + '/../src/features/vehicle_damage_scaler.pkl', 'rb'))
        self.vehicle_age_scaler = pickle.load(open(self.home_path + '/../src/features/vehicle_age_scaler.pkl', 'rb'))
        self.region_code_scaler = pickle.load(open(self.home_path + '/../src/features/region_code_scaler.pkl', 'rb'))
        self.policy_sales_channel_scaler = pickle.load(open(self.home_path + '/../src/features/policy_sales_channel_scaler.pkl', 'rb'))
   
    def data_preparation(self, df):
        df1 = df.copy()
        df1['age'] = self.age_scaler.transform(df1[['age']].values)
        df1['vintage'] = self.vintage_scaler.transform(df1[['vintage']].values)
        df1['annual_premium'] = self.annual_premium_scaler.transform(df1[['annual_premium']].values)
        df1['gender'] = self.gender_scaler.transform(df1[['gender']].values)
        df1['vehicle_damage'] = self.vehicle_damage_scaler.transform(df1[['vehicle_damage']].values)
        df1['vehicle_age'] = self.vehicle_age_scaler.transform(df1[['vehicle_age']].values)
        df1['region_code'] = self.region_code_scaler.transform(df1[['region_code']].values)
        df1['policy_sales_channel'] = self.policy_sales_channel_scaler.transform(df1[['policy_sales_channel']].values)

        cols_selected = ['policy_sales_channel', 'previously_insured', 'region_code', 'vehicle_damage',
                         'age', 'annual_premium', 'vintage']
        return df1[cols_selected]

    def get_prediction(self, model, original_data, test_data):
        pred = model.predict_proba(test_data)
        original_data['prediction'] = pred[:,1]
        return original_data.to_json(orient='records', date_format='iso')