import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import jensenshannon

def calculate_js_divergence(df1, df2):
    # Ensure both dataframes have the same columns in the same order
    all_tags = set(df1.columns).union(df2.columns)
    df1 = df1.reindex(columns=all_tags, fill_value=0)
    df2 = df2.reindex(columns=all_tags, fill_value=0)
    
    # Calculate Jensen-Shannon Divergence
    p = df1.values[0]
    q = df2.values[0]
    js_div = jensenshannon(p, q, base=2)
    return js_div

# Directory containing POS distribution files
tscc_folder_path = # Folder path here
tscc_files = os.listdir(tscc_folder_path)

# Combining all the student and instructor files tscc
student_dfs = []
instructor_dfs = []

for filename in tscc_files:
    file_path = os.path.join(tscc_folder_path, filename)
    file_number = filename.split('_')[0] 
    
    if 'student' in filename:
        temp_student = pd.read_csv(file_path)
        temp_student['filename'] = file_number  
        student_dfs.append(temp_student)  
        
    elif 'instructor' in filename:
        temp_instructor = pd.read_csv(file_path)
        temp_instructor['filename'] = file_number  
        instructor_dfs.append(temp_instructor) 

tscc_student_df = pd.concat(student_dfs, ignore_index=True)
tscc_instructor_df = pd.concat(instructor_dfs, ignore_index=True)

# Matching all the student and instructor together then looping through each row to calculate JSD
tscc_unique_filenames = tscc_student_df['filename'].unique()

jsd_results = pd.DataFrame(columns=['filename', 'jsd'])

student_data = tscc_student_df[tscc_student_df['filename'] == 3]
print(f"Number of rows for student data with filename '{3}':", len(student_data))

instructor_data = tscc_instructor_df[tscc_instructor_df['filename'] == 3]
print(f"Number of rows for instructor data with filename '{3}':", len(instructor_data))


for filename in tscc_unique_filenames:
    # Filter the data for the current filename in both student and instructor DataFrames
    student_data = tscc_student_df[tscc_student_df['filename'] == filename]
    instructor_data = tscc_instructor_df[tscc_instructor_df['filename'] == filename]
    
    num_rows = min(len(student_data), len(instructor_data))
    
    for i in range(num_rows):
        student_row = student_data.iloc[i]
        instructor_row = instructor_data.iloc[i]
        
        # Convert Series to DataFrame
        student_df = student_row.to_frame().T 
        instructor_df = instructor_row.to_frame().T  

        # Avoid calculating filename, drop non-numeric columns if any
        student_df_filtered = student_df.drop(columns=['filename'], errors='ignore')
        instructor_df_filtered = instructor_df.drop(columns=['filename'], errors='ignore')

        # Ensure the data is numeric and of type float64
        student_df_filtered = student_df_filtered.apply(pd.to_numeric, errors='coerce').astype(float)
        instructor_df_filtered = instructor_df_filtered.apply(pd.to_numeric, errors='coerce').astype(float)

        student_df_filtered = student_df_filtered.fillna(0)
        instructor_df_filtered = instructor_df_filtered.fillna(0)

        # Calculate the JSD for the current row
        jsd = calculate_js_divergence(student_df_filtered, instructor_df_filtered)
        
        # Append the filename and jsd to the jsd_results DataFrame
        new_row = pd.DataFrame({'filename': [filename], 'jsd': [jsd]})
        # Concatenate the new row with jsd_results
        jsd_results = pd.concat([jsd_results, new_row], ignore_index=True)

jsd_results.to_csv(os.path.join(tscc_folder_path, 'tscc_JSD_Results.csv'), index=False)