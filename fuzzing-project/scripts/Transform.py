import pandas as pd

CSV_PATH = '/home/zacky/opt/fuzzing-project/scripts/requests_responses.csv'

df = pd.read_csv(CSV_PATH, low_memory=True)
