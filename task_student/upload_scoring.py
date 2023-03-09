import pandas as pd
import numpy as np
from functools import lru_cache
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os

def return_intersecting_cols(df_cols, cols=[], header=[0, 1]):
    if len(header) > 1:
        # due to header = [0,1], we need to only select the first value of each column 
        df_cols = [x[0] for x in df_cols]
    return [col_name for col_name in cols if col_name in df_cols]

# instead of global loading, we just add a cache, since the input doesn't change this should benefit the performance quite a lot (similar to loading the file on a module level)
@lru_cache(maxsize=2)
def load_ref(track='student'):
    header = [0,1]
    
    # load test set
    solution_file_path = os.path.join(os.path.dirname(__file__), f'nice_try/{track}.csv')
    df_test = pd.read_csv(solution_file_path, delimiter=';', header=header)

    # determine scoring and time cols
    scoring_cols = return_intersecting_cols(df_test.columns, ['SECCI', 'Temperatur', 'Salinität', 'SiO4', 'PO4', 'NO2', 'NO3', 'NOx', 'NH4'], header=header)
    time_cols = return_intersecting_cols(df_test.columns, ['Datum', 'Uhrzeit'], header=header)

    # load test set and train scaler with it
    scaler = StandardScaler()
    scaler.fit(df_test[scoring_cols].values)
    print('Cols:', scoring_cols)
    print('Means:', scaler.mean_)
    print('Variance:', scaler.var_)
    
    # scale test set
    df_test[scoring_cols] = scaler.transform(df_test[scoring_cols].values)

    return df_test, scoring_cols, time_cols, scaler


def _check_submission(submission_df, ref_df, time_cols, scoring_cols):
    if len(submission_df) != len(ref_df):
        return False, f"Shape missmatch, expected length {len(ref_df)}. Got {len(submission_df)}"
    
    submission_dates = set(np.unique(submission_df[time_cols[0]]))
    ref_dates = set(np.unique(ref_df[time_cols[0]]))
    if submission_dates != ref_dates:
        return False, f"Dates missmatch. Please check your upload file."

    return True, "Ok"

def score_all(ref, pred):
    final_score = mean_squared_error(ref, pred, squared=False)
    return final_score, f"Upload erfolgreich. Final score: {final_score:.2f}"

def evaluateSubmission(submissionFile, track='student'):
    # Load Reference
    try:
        ref_df, scoring_cols, time_cols, scaler = load_ref(track)
    except Exception as e:
        return float('nan'), f'Error loading reference, please conctact the BBDC organisers at bbdc@uni-bremen.de', True
    
    # Parse / Load submission
    try:
        submission_df = pd.read_csv(submissionFile, delimiter=';', header=[0,1])

        submission_ok, submission_msg = _check_submission(submission_df, ref_df, time_cols, scoring_cols)
        if not submission_ok:
            return float('nan'), f"Submission Error: {submission_msg}", True

    except Exception as e:
        return float('nan'), f'Invalid files: {str(e)}', True

    # Score submission
    try:
        score, score_msg = score_all(ref_df[scoring_cols].values, scaler.transform(submission_df[scoring_cols].values))
    except Exception as e:
        print(e)
        return float('nan'), str(e), True

    # Return score if everything went well
    return score, score_msg, True
