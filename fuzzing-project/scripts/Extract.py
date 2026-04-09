import sqlite3
import pandas as pd

conn = sqlite3.connect('/home/zacky/opt/fuzzing-project/reports/grafana/report.db')

query = """
SELECT * FROM requests 
INNER JOIN responses ON requests.id = responses.id
"""


df_totaal = pd.read_sql_query(query, conn)

conn.close()



df_totaal.to_csv('/home/zacky/opt/fuzzing-project/scripts/requests_responses.csv', index=False)