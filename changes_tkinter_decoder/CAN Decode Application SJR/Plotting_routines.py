# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 12:04:41 2025

@author: sughosh.rao

Plotting routines

Function to create time vs decimal vals with inputs CANID, Bytes, Bits.
function to plot.

"""
import matplotlib.pyplot as plt
import pandas
from CAN_Decode_Utils import *


if __name__=="__main__":
    from CAN_Decode_Utils import *
   

def plot(CANID,Bytes_list, Bits_list):
    time, data = get_plotting_data(CANID, Bytes_list, Bits_list)
    plt.plot(time,data)
    plt.xlabel('Time (s)')
    plt.ylabel('Data')
    plt.show()
    plt.grid(True)
    
    
def bin_to_dec(bin_string):
    out = int(bin_string,2) 
    return out
    
    
def get_plotting_data(CANID, Bytes_list, Bits_list):
    """Fron CAN ID, list of bytes and list of list of bits, construct data and 
    return Time list and Data list
    Bytes_list: should have list of bytes in order from most significant to least
    Bits_list: this is a list of list of bits where.  
        list of bits for each byte in the same order as the bytes list"""
    
    if len(Bytes_list) != len(Bits_list):
        raise ValueError("Bits list not equal to Bytes List")
    
    if get_msg_len(CANID)< max(Bytes_list):
        raise ValueError("Byte out of range of CANID payload length")
        
    Data = {}   
    for index, byte in enumerate(Bytes_list):
        data_df = get_byte(CANID, byte)
        for i in range(min(Bits_list[index]), max(Bits_list[index])+1,1):
            Data[CANID+str(byte)+str(i)] = data_df[str(i)]
    
    df = pandas.DataFrame(Data)
    
    df['Data']=  df[df.columns].apply(lambda x:''.join(x.dropna().astype(str)),axis=1)   
    df['Decimal_Data'] = df['Data'].apply(bin_to_dec)
    
    return data_df['Time'].to_list(), df['Decimal_Data'].to_list()
            
            
            