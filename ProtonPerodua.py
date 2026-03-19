import pandas as pd

# 1. Load your Malaysian dataset
df = pd.read_csv('D:/UIA/LAB 1 Python Intsllation with VSCODE-20250306/Machine Learning/Fuel/fuelconsumption2.csv')

# 2. Toyota Rush 2010 Data
# Specs: 1.5L, 4-Cylinder, 4-Speed Auto, Regular Petrol
rush_data = [
    [2010, 'TOYOTA', 'RUSH 1.5G/S', 'SUV: SMALL', 1.5, 4, 'A4', 'X', 9.2, 7.2, 8.2, 34, 193]
]

# 3. Create DataFrame
columns = ['YEAR', 'MAKE', 'MODEL', 'VEHICLE CLASS', 'ENGINE SIZE', 'CYLINDERS', 
           'TRANSMISSION', 'FUEL', 'FUEL CONSUMPTION', 'HWY (L/100 km)', 
           'COMB (L/100 km)', 'COMB (mpg)', 'EMISSIONS']

rush_df = pd.DataFrame(rush_data, columns=columns)

# 4. Append and Save
updated_df = pd.concat([df, rush_df], ignore_index=True)
updated_df.to_csv('Fuel_Consumption_Malaysia.csv', index=False)

print("Toyota Rush 2010 added to the dataset!")