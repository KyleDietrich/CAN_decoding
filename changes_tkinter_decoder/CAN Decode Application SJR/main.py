# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 00:55:03 2025

@author: sughosh.rao

Use virtual Env CAN_GUI_Env GUI
"""

#clear_all()

import pandas as pd


from CAN_Decode_Utils import * #, CAN_Preprocess, analyze_byte
from Plotting_routines import *
from Video_Player_Class2 import *




#%% READ CAN Data csv file. The CAN data needs to be in the format in the example with the correct column headings as in the example
Vehicle = "../Ram_4500"

#Import CAN data as string. 
df_raw_CAN = pandas.read_csv(Vehicle+"/Ram 4500 Test 4.csv", dtype=str)

print(df_raw_CAN.columns)
print(df_raw_CAN.head())

# --- Remove non-CAN rows (like the first "Trigger" row) ---
df_raw_CAN = df_raw_CAN[df_raw_CAN["Identifier"].notna()].copy()

# --- Rename Timestamp -> Time ---
df_raw_CAN["Time"] = df_raw_CAN["Timestamp"].astype(float)

# --- Rename Identifier -> ID (add 0x prefix so it matches the GUI style) ---
df_raw_CAN["ID"] = df_raw_CAN["Identifier"].apply(lambda x: "0x" + str(x).upper())

# --- Convert Byte1..Byte8 -> "0".."7" ---
byte_map = {
    "Byte1": "0",
    "Byte2": "1",
    "Byte3": "2",
    "Byte4": "3",
    "Byte5": "4",
    "Byte6": "5",
    "Byte7": "6",
    "Byte8": "7",
}
df_raw_CAN = df_raw_CAN.rename(columns=byte_map)

# --- Keep only what the project expects ---
cols_needed = ["ID", "Time"] + [str(i) for i in range(8)]
df_raw_CAN = df_raw_CAN[cols_needed]

print("Converted dataframe columns:", df_raw_CAN.columns)
print(df_raw_CAN.head())


#%% Preprocess the CAN messages
# filter the CAN messages to get a list of dataframes with one dataframe for 
# each CAN message
CAN_filtered = CAN_Filter(df_raw_CAN)

# Analyze each message to see which CANID+Byte combos change and how many times 
# they change  
Candidates, CAN_details, Changed_CAN_list = CANdidate_Bytes()   

Candidates = CANdidate_bits(Candidates)

print(CAN_details.head())
print(Candidates.sort_values("No. of changes", ascending=False).head(30))

#%%
# video_file = "../Lexus/Lexus_AV_Data_11-20_0001.cam0_190fps.avi"
# Vid_main(video_file)
#%%
