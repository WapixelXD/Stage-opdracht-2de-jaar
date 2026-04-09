import sqlite3
import pandas as pd

# 1. Verbinding maken
conn = sqlite3.connect('/home/zacky/opt/fuzzing-project/reports/grafana/report.db')

# 2. Beide tabellen inlezen
df_requests = pd.read_sql_query("SELECT * FROM requests", conn)
df_responses = pd.read_sql_query("SELECT * FROM responses", conn)

# 3. Verbinding sluiten
conn.close()

# 4. Samenvoegen (Merge)
# Vervang 'id' door de kolomnaam die in beide tabellen hetzelfde is
# 'how=inner' zorgt dat je alleen rijen overhoudt die in beide tabellen voorkomen
df_totaal = pd.merge


df_totaal