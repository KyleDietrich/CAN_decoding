# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 12:02:36 2025

@author: sughosh.rao

README
CAN data should have column names in this order: ID,Time,1,2,3... no of bytes
 
"""

import pandas
#import numpy as np
CAN_filtered = list() #setting global variable
CAN_details = pandas.DataFrame() #setting can details as global variable
Max_msg_len = 8 #set default maximum message length to 8 as global variable

def CAN_Filter(df_raw_CAN):
    """ Preprocess the collected CAN data 
    Inputs
    df_raw_CAN: data frame of CAN messages with columns "ID, Time, and bytes nos"     
    and outputs:
        4. CAN_filtered: list of dataframes with each dataframe for a separate CAN ID.
                         data frame columns:
            ID          : CAN ID
            Time        : time stamp
            1...n       : payload
        """
    global CAN_filtered
    global Max_msg_len
    Max_msg_len = df_raw_CAN.shape[1]-2  #maximum lenght of a CAN message in this vehicle  
    #Get unique CAN Ids present in the recorded CAN data
    CAN_ids = df_raw_CAN['ID'].unique()
     
    # This for loop creates individual data frames for each CAN ID and saves them in a list of dataframes
    for index, item in enumerate(CAN_ids):
        # filter the CAN messages for CAN id = "item"
        temp_df = df_raw_CAN[df_raw_CAN['ID'].explode().eq(item)]
                 
        # Save dataframe of each message ID into a list of dataframes
        CAN_filtered.append(temp_df.reset_index())
        
    return CAN_filtered

#%%
def CANdidate_Bytes():
    """  Outputs:    
    1.CAN_candidates: This is a dataframe that contains only the bytes that 
                    have changed, These are the candidate bytes in ascending
                    order of number of changes. The columns in this dataframe are:
        ID          : CAN ID
        Msg_Length  : Lenght of payload for this CAN ID
        No          : No. of times CAN ID is published
        Byte No     : Byte number in the message
        No. of Change: The number of times this byte has changed.
        
    2.CAN_details : dataframe with columns:
        ID          : CAN ID
        Msg_Length  : Length of payload for this CAN ID
        No          : Number of times message is published
        1...n       : column for each byte, containing no. of times that byte has changed
    
    3. Changed_CAN_list: list of dataframes with one dataframe for each CAN ID.
                     The columns in these dataframes:
        Time        : Time in seconds
        1..n        : columns for each byte. column contains 1 if data for 
                      that byte changed from previous instance, 0 if it 
                      stayed the same. """

    global Max_msg_len
    global CAN_filtered
    global CAN_details
    CAN_ids_len = list()
    Changed_CAN_list = list()
    
    for temp_df in CAN_filtered:
        #find the payload length for this CAN id by finding the first nan entry
        #if no 'nan' entries are found, it defaults to first column (0), so this 
        # checkes if 1st column is returned and if so, sets it to full length 
        # (32 bytes for this vehicle, can go up to 64 for other CANFD vehicles)
        # else, sets it to the length were nan was found
        inspect_idx = 2 if len(temp_df) > 2 else 0
        first_nan_column_idx = temp_df.columns.get_loc(temp_df.iloc[inspect_idx].isna().idxmax(axis=0))

        if  first_nan_column_idx == 0:      # check if 1st nan find has returned 1st column
            Message_length = Max_msg_len
        else:
            Message_length = first_nan_column_idx-3 #-3 since 1st 2 columns are id and time. 
            
        # looping from 1 to payload length of each message, to see if it has changed.
        Changed_CAN = {} #initialize a dictionary
        Changed_CAN['ID']=temp_df['ID']
        Changed_CAN['Time']=temp_df['Time'] # add time column to dictionary
        for i in range(Message_length): #looping through the columns(bytes) of each CAN ID
            #this is a column by column operation to see if the data has changed 
            #from one time step to the next in that column. if there is a change 1 else 0.
           Changed_CAN[str(i)]=temp_df[[str(i)]].ne(temp_df[[str(i)]].shift()).any(axis=1).astype(int) 
           
        Changed_CAN_list.append(pandas.DataFrame(Changed_CAN))
        
        #this sums all the columns for each byte to see how many times each byte changes
        # the -1 is because the first row is always 1 to negate this. 
        sum_of_all_rows_of_selected_columns = Changed_CAN_list[-1].iloc[:, 2:].sum(axis=0)-1
        
        #make list of CAN ids and no. of messages of each id.
        CAN_ids_len.append([temp_df['ID'].iloc[0], Message_length, temp_df.shape[0], *sum_of_all_rows_of_selected_columns.tolist()])


    #CREATE dataframe with CAN ID, Length of CAN msg, No. of times the 
    # message is published, no. of times each of the bytes changed.  
    CAN_details = pandas.DataFrame(CAN_ids_len)
    CAN_details.columns = ["ID","Msg_Length","No", *[str(i) for i in range(Max_msg_len)]]
    
    #CREATE dataframe with only bytes that have changed. this is the cadidate list:
    #Columns: ID, Msg_Length, No, Byte no, No.of changes.
    id_vars = ['ID', 'Msg_Length', 'No']
    value_vars = [str(i) for i in range(Max_msg_len)]

    # Use pd.melt to perform the stacking
    stacked_df = pandas.melt(CAN_details, id_vars=id_vars, value_vars=value_vars,
                         var_name='Byte No', value_name='No. of changes')

    condition = stacked_df['No. of changes']==0
    indices_to_drop = stacked_df[condition].index #remove rows where the byte has not changed at all

    stacked_df1 = stacked_df.drop(indices_to_drop)
    stacked_df2 = stacked_df1.dropna(subset=['No. of changes']) #remove rows that have nan since these bytes don't exist.
    CAN_candidates = stacked_df2.sort_values(by='No. of changes')
    
    return CAN_candidates.reset_index(), CAN_details, Changed_CAN_list



def get_msg_len(CANID):
    """returns message length for provided CANID"""
    length = CAN_details.loc[CAN_details['ID']==CANID]['Msg_Length'].to_list()[0]
    return length 

#%% FUNCTIONS FOR BITEWISE and BITS ANALYSIS

#Function for converting hex string to bits
def hex_to_bits(hex_str):
    if pandas.isna(hex_str):
        return [None] * 8  # Handle NaN values if present
    
    # Remove '0x' prefix if present
    hex_value = hex_str.replace('0x', '')
    
    # Convert hex to integer
    int_value = int(hex_value, 16)
    
    # Convert integer to 8-bit binary string
    binary_str = format(int_value, '08b')
    # Convert binary string to a list of integers (bits)
    return [int(bit) for bit in binary_str]


# Get the dataframe with all msgs of a certain CAN ID from the list of dfs
def get_CAN_ID_data(CANID): #,list_of_CANID_dfs):
    """searches the list of dataframes for the dataframe corresponding to the
    required CAN ID"""
    global CAN_filtered
    for index, df in enumerate(CAN_filtered): #(list_of_CANID_dfs):
        if df['ID'].iloc[0] == CANID:
            break
        elif index == len(CAN_filtered)-1:
            raise ValueError("CAN ID not found")
    df_out = df.copy(deep=True) #exporting a copy so that original df is not altered.
    return df_out



def get_byte(CANID,Byteno): #,list_of_CANID_dfs):
    """Given CANID and Byte, returns Dataframe with to bits in indivitual colums 
    of df along with Time column
    """ 
    df = get_CAN_ID_data(CANID) #list_of_CANID_dfs)
    df['bits']= df[str(Byteno)].apply(hex_to_bits) # Apply the function to the byte column

    df_out=pandas.DataFrame() #initialize empty dataframe
    df_out['Time']=df['Time']

    # Create new columns for each bit
    bit_columns = [str(i) for i in range(8)]
    df_out[bit_columns] = pandas.DataFrame(df['bits'].tolist(), index=df.index) 
    
    # Drop the intermediate 'bits' column if no longer needed
    df = df.drop(columns=['bits'])
    return df_out

def track_changes(CANID, Byteno):
    """This function calculates how many times the bits in a byte change when 
    supplied with CANID and Byteno"""
    df = get_byte(CANID, Byteno)
    columns_to_check = df.columns.tolist()
    columns_to_check.remove('Time')

    # Calculate changes for the selected columns
    changes = (df[columns_to_check].diff() != 0).sum()-1   
    return changes

def CANdidate_bits(Candidates):
    """Given dataframe of candidate bytes, generates the number of times the
    bits in the bytes have changed and updates the input """
    bit_changes = [] #initialize list 

    for index, item in Candidates.iterrows():
        # get how many times each bit has changed for candidate CANID/Byte combo
        change = track_changes(item['ID'], item['Byte No'])
        bit_changes.append(change.tolist())
    bit_changes = list(map(list, zip(*bit_changes)))
    for i in range(8):
        Candidates[str(i)] = bit_changes[i] 
    return Candidates

