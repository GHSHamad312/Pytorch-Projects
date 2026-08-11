import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, file):
        df=pd.read_csv(file)
        self.x=torch.tensor(df[["equipment_id","operating_hours","rotation_speed_rpm","torque_nm","power_kw","ambient_temp_c","bearing_temp_c","vibration_amplitude_mm","oil_viscosity_cst","coolant_pressure_bar","acoustic_noise_db","voltage_fluctuation_v","prior_maintenance_count","error_log_count_24h","remaining_useful_life_days","equipment_failure_risk"]], dtype=torch.float32)
        self.y=torch.tensor(df["failure_mode_type"])
    def __length__(self):
        return len(self.x)

    def __getitem__(self,index):
        return self.x[index],self.y[index]