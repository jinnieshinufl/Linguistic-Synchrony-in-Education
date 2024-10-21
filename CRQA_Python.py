#https://arxiv.org/html/2402.16853v1#:~:text=Among%20others%2C%20the%20package%20crqa,e.g.%2C%20recurrence%20rate%20and%20determinism.
#https://pypi.org/project/PyRQA/#description

from pyrqa.analysis_type import Cross
from pyrqa.computation import RQAComputation
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric, MaximumMetric, TaxicabMetric
from pyrqa.computation import RPComputation

import matplotlib.pyplot as plt
import pandas as pd
import ast

def convert_to_float_list(string_list):
    return [float(item) for item in ast.literal_eval(string_list)]


file_path = "final_full_post_processed_equalized.csv"
df = pd.read_csv(file_path)

#sample only specific file name
filtered_df = df[df['filename'] == 'f3b69512-0b6a-49fa-bac7-df278420cedb']
# filtered_df = filtered_df.iloc[1:]
# filtered_df = filtered_df.head(4)
# print(filtered_df.head())


instructor_text_list = [convert_to_float_list(row) for row in filtered_df[filtered_df['speaker'] == 'instructor']['token_ids']]
student_text_list = [convert_to_float_list(row) for row in filtered_df[filtered_df['speaker'] == 'student']['token_ids']]

from itertools import chain
instructor_text_list = list(chain.from_iterable(instructor_text_list))
student_text_list  = list(chain.from_iterable(student_text_list))

time_series_instructor = TimeSeries(instructor_text_list,
                           embedding_dimension=1,
                           time_delay=1)
time_series_student = TimeSeries(student_text_list,
                           embedding_dimension=1,
                           time_delay=1)


time_series = (time_series_instructor, 
               time_series_student)

settings = Settings(time_series,
                    analysis_type=Cross,
                    neighbourhood=FixedRadius(1), #decides how close points should be to consider them as recurrent
                    similarity_measure=EuclideanMetric, #calculation for how close points are 
                    theiler_corrector=0) #excludes points if they are too close

computation = RQAComputation.create(settings,
                                    verbose=True)
result = computation.run()
print(result)

computation = RPComputation.create(settings)
resultRP = computation.run()

plt.figure(figsize=(5, 5))
plt.imshow(resultRP.recurrence_matrix, cmap='Greys', origin='lower', interpolation='none')
plt.title('Recurrence Plot')
plt.xlabel('Instructor') 
plt.ylabel('Student')
plt.show()

# import numpy as np
# import os

# # Assuming filtered_df is already defined and loaded with the data
# instructor_text_list = [convert_to_float_list(row) for row in filtered_df[filtered_df['speaker'] == 'instructor']['token_ids']]
# student_text_list = [convert_to_float_list(row) for row in filtered_df[filtered_df['speaker'] == 'student']['token_ids']]
# # Initialize a list to store results dictionaries
# results = []

# # Iterate through each pair of instructor and student lists
# for idx, (instructor_data, student_data) in enumerate(zip(instructor_text_list, student_text_list)):
#     # Create time series objects
#     time_series_instructor = TimeSeries(instructor_data, embedding_dimension=1, time_delay=1)
#     time_series_student = TimeSeries(student_data, embedding_dimension=1, time_delay=1)

#     # Bundle time series into settings
#     settings = Settings((time_series_instructor, time_series_student),
#                         analysis_type=Cross,
#                         neighbourhood=FixedRadius(1),
#                         similarity_measure=EuclideanMetric,
#                         theiler_corrector=0)

#     # Perform CRQA computation
#     computation = RQAComputation.create(settings, verbose=True)
#     result = computation.run()

#     # Store results in a dictionary
#     result_dict = {
#         'Pair Index': idx,
#         'Recurrence Rate': result.recurrence_rate,
#         'Determinism': result.determinism,
#         'Average Diagonal Line Length': result.average_diagonal_line,
#         'Longest Diagonal Line': result.longest_diagonal_line,
#         'Divergence': result.divergence,
#         'Entropy of Diagonal Lines': result.entropy_diagonal_lines,
#         'Laminarity': result.laminarity,
#         'Trapping Time': result.trapping_time,
#         'Longest Vertical Line': result.longest_vertical_line,
#         'Entropy of Vertical Lines': result.entropy_vertical_lines,
#         'Average White Vertical Line Length': result.average_white_vertical_line,
#         'Longest White Vertical Line': result.longest_white_vertical_line,
#         'Entropy of White Vertical Lines': result.entropy_white_vertical_lines,
#     }
#     results.append(result_dict)

# # Create a DataFrame from the results list
# results_df = pd.DataFrame(results)

# # Define the Excel file path
# excel_path = 'crqa_pairwise_results.xlsx'

# # Check if the file exists to determine the mode
# if os.path.exists(excel_path):
#     mode = 'a'
# else:
#     mode = 'w'

# # Save to Excel
# with pd.ExcelWriter(excel_path, mode=mode, engine='openpyxl') as writer:
#     results_df.to_excel(writer, sheet_name='CRQA Results', index=False)

# print(f"Results have been saved to '{excel_path}'.")

from itertools import chain

# # Assume df is already defined and loaded with the data
# unique_filenames = df['filename'].unique()

# results = []

# for filename in unique_filenames:
#     filtered_df = df[df['filename'] == filename]
    
#     instructor_text_list = [convert_to_float_list(row) for row in filtered_df[filtered_df['speaker'] == 'instructor']['token_ids']]
#     student_text_list = [convert_to_float_list(row) for row in filtered_df[filtered_df['speaker'] == 'student']['token_ids']]

#     # Flatten lists if needed
#     instructor_text_list = list(chain.from_iterable(instructor_text_list))
#     student_text_list = list(chain.from_iterable(student_text_list))

#     # Create TimeSeries objects
#     time_series_instructor = TimeSeries(instructor_text_list, embedding_dimension=1, time_delay=1)
#     time_series_student = TimeSeries(student_text_list, embedding_dimension=1, time_delay=1)

#     # Settings for CRQA
#     settings = Settings((time_series_instructor, time_series_student),
#                         analysis_type=Cross,
#                         neighbourhood=FixedRadius(1),
#                         similarity_measure=EuclideanMetric,
#                         theiler_corrector=0)

#     # Perform CRQA computation
#     computation = RQAComputation.create(settings, verbose=True)
#     result = computation.run()
    
#     # Manually create a dictionary from the RQAResult object
#     result_dict = {
#         'filename': filename,
#         'recurrence_rate': result.recurrence_rate,
#         'determinism': result.determinism,
#         'average_diagonal_line': result.average_diagonal_line,
#         'longest_diagonal_line': result.longest_diagonal_line,
#         'entropy_diagonal_lines': result.entropy_diagonal_lines,
#         'laminarity': result.laminarity,
#         'trapping_time': result.trapping_time,
#         'longest_vertical_line': result.longest_vertical_line,
#         'entropy_vertical_lines': result.entropy_vertical_lines,
#         'average_white_vertical_line': result.average_white_vertical_line,
#         'longest_white_vertical_line': result.longest_white_vertical_line,
#         'entropy_white_vertical_lines': result.entropy_white_vertical_lines
#     }
#     results.append(result_dict)

# # Convert results to a DataFrame
# results_df = pd.DataFrame(results)

# # Export to Excel
# results_df.to_excel('crqa_results.xlsx', index=False)

# print("Results have been saved to 'crqa_results.xlsx'.")