import pandas as pd

# 1. Load the original global data
df = pd.read_csv('D:/UIA/LAB 1 Python Intsllation with VSCODE-20250306/Machine Learning/Fuel/fuelconsumption.csv')

# 2. Comprehensive list of Malaysian Local Cars (Variants)
local_cars_data = [
    [2023, 'PERODUA', 'AXIA 1.0 G/X', 'COMPACT', 1.0, 3, 'D-CVT', 'X', 3.95],
    [2023, 'PERODUA', 'AXIA 1.0 SE/AV', 'COMPACT', 1.0, 3, 'D-CVT', 'X', 3.65],
    [2023, 'PERODUA', 'MYVI 1.3 G', 'COMPACT', 1.3, 4, 'D-CVT', 'X', 4.50],
    [2023, 'PERODUA', 'MYVI 1.5 X/H/AV', 'COMPACT', 1.5, 4, 'D-CVT', 'X', 4.70],
    [2023, 'PERODUA', 'BEZZA 1.0 G (M)', 'COMPACT', 1.0, 3, 'M5', 'X', 4.40],
    [2023, 'PERODUA', 'BEZZA 1.0 G (A)', 'COMPACT', 1.0, 3, 'A4', 'X', 4.70],
    [2023, 'PERODUA', 'BEZZA 1.3 X', 'COMPACT', 1.3, 4, 'A4', 'X', 4.80],
    [2023, 'PERODUA', 'BEZZA 1.3 AV', 'COMPACT', 1.3, 4, 'A4', 'X', 4.50],
    [2023, 'PERODUA', 'ATIVA 1.0 TURBO', 'SUV: SMALL', 1.0, 3, 'D-CVT', 'X', 5.30],
    [2022, 'PERODUA', 'ALZA 1.5 (ALL)', 'MPV', 1.5, 4, 'D-CVT', 'X', 4.50],
    [2023, 'PERODUA', 'ARUZ 1.5 (ALL)', 'SUV: SMALL', 1.5, 4, 'A4', 'X', 6.41],
    [2022, 'PROTON', 'SAGA 1.3 (M)', 'COMPACT', 1.3, 4, 'M5', 'X', 5.60],
    [2022, 'PROTON', 'SAGA 1.3 (A)', 'COMPACT', 1.3, 4, 'A4', 'X', 6.00],
    [2023, 'PROTON', 'PERSONA 1.6 CVT', 'MID-SIZE', 1.6, 4, 'CVT', 'X', 6.60],
    [2023, 'PROTON', 'IRIZ 1.3 CVT', 'COMPACT', 1.3, 4, 'CVT', 'X', 6.90],
    [2023, 'PROTON', 'IRIZ 1.6 CVT', 'COMPACT', 1.6, 4, 'CVT', 'X', 7.40],
    [2024, 'PROTON', 'S70 1.5 TURBO', 'MID-SIZE', 1.5, 3, 'DCT7', 'X', 6.20],
    [2023, 'PROTON', 'X50 1.5T (STD/EXE/PRE)', 'SUV: SMALL', 1.5, 3, 'DCT7', 'X', 6.50],
    [2023, 'PROTON', 'X50 1.5TGDI (FLAGSHIP)', 'SUV: SMALL', 1.5, 3, 'DCT7', 'X', 6.40],
    [2024, 'PROTON', 'X70 1.5 TGDI', 'SUV: MID-SIZE', 1.5, 3, 'DCT7', 'X', 7.80],
    [2023, 'PROTON', 'X90 1.5 HYBRID', 'SUV: MID-SIZE', 1.5, 3, 'DCT7', 'Z', 6.20]
]

# 3. Create DataFrame and Append
columns = ["YEAR", "MAKE", "MODEL", "VEHICLE CLASS", "ENGINE SIZE", "CYLINDERS", "TRANSMISSION", "FUEL", "COMB (L/100 km)"]
local_df = pd.DataFrame(local_cars_data, columns=columns)
final_df = pd.concat([df, local_df], ignore_index=True)

# 4. Save to a NEW filename
final_df.to_csv('Fuel_Consumption_Malaysia.csv', index=False)
print("New file 'Fuel_Consumption_Malaysia.csv' created with Proton/Perodua data!")