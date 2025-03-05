#https://arxiv.org/html/2402.16853v1#:~:text=Among%20others%2C%20the%20package%20crqa,e.g.%2C%20recurrence%20rate%20and%20determinism.
#https://pypi.org/project/PyRQA/#description

from pyrqa.analysis_type import Cross
from pyrqa.computation import RQAComputation
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric, MaximumMetric, TaxicabMetric
from itertools import chain

import matplotlib.pyplot as plt
import pandas as pd
import ast

def convert_to_float_list(string_list):
    return [float(item) for item in ast.literal_eval(string_list)]

# Load dataset
file_path = "tscc_full_post_processed_equalized.csv" 
df = pd.read_csv(file_path)

# Assume df is already defined and loaded with the data
unique_filenames = df['chat_id'].unique()

results = []

for filename in unique_filenames:
    filtered_df = df[df['chat_id'] == filename]
    
    instructor_text_list = [convert_to_float_list(row) for row in filtered_df[filtered_df['role'] == 'teacher']['padded_token_ids']]
    student_text_list = [convert_to_float_list(row) for row in filtered_df[filtered_df['role'] == 'student']['padded_token_ids']]

    # Flatten lists if needed
    instructor_text_list = list(chain.from_iterable(instructor_text_list))
    student_text_list = list(chain.from_iterable(student_text_list))

    # Create TimeSeries objects
    time_series_instructor = TimeSeries(instructor_text_list, embedding_dimension=1, time_delay=1)
    time_series_student = TimeSeries(student_text_list, embedding_dimension=1, time_delay=1)

    # Settings for CRQA
    settings = Settings((time_series_instructor, time_series_student),
                        analysis_type=Cross,
                        neighbourhood=FixedRadius(1),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=0)

    # CRQA computation
    computation = RQAComputation.create(settings, verbose=True)
    result = computation.run()
    
    # Manually create a dictionary from the RQAResult object
    result_dict = {
        'filename': filename,
        'recurrence_rate': result.recurrence_rate,
        'determinism': result.determinism,
        'average_diagonal_line': result.average_diagonal_line,
        'longest_diagonal_line': result.longest_diagonal_line,
        'entropy_diagonal_lines': result.entropy_diagonal_lines,
        'laminarity': result.laminarity,
        'trapping_time': result.trapping_time,
        'longest_vertical_line': result.longest_vertical_line,
        'entropy_vertical_lines': result.entropy_vertical_lines,
        'average_white_vertical_line': result.average_white_vertical_line,
        'longest_white_vertical_line': result.longest_white_vertical_line,
        'entropy_white_vertical_lines': result.entropy_white_vertical_lines
    }
    results.append(result_dict)

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Export to Excel
results_df.to_excel('tscc_crqa_results.xlsx', index=False)

print("Results have been saved to 'crqa_results.xlsx'.")