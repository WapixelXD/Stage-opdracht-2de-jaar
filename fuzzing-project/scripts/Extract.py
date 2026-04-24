"""
Data Processing Utility for Fuzzing Reports.

This script connects to a specified SQLite database containing fuzzing logs,
joins the 'requests' and 'responses' tables on their matching ID, 
and exports the consolidated data into a single CSV file for analysis.


Inputs:
    - Database Connection: (Path determined relative to the project root)
      (The source database containing requests and responses.)

Output:
    - CSV File: (Path determined relative to the project root)
      (A flat file containing all merged request and response data.)

Raises:
    - sqlite3.Error: If the specified database cannot be accessed or the schema is incorrect.
    - pd.errors.EmptyDataError: If the resulting query returns no rows.


Usage Notes:
    The script assumes 'id' serves as a valid primary/foreign key linking records 
    between both tables. Ensure that this join logic remains accurate if the database schema changes.
"""

import sqlite3
import pandas as pd
import os


def run_analysis():
    """
    Connects to the database, executes the JOIN query, and saves results to CSV.
    Uses OS path manipulation to avoid hardcoding absolute paths.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    #Aanpassen als het  root bestand ergens anders zit.
    base_path = os.path.join(current_dir, "..") 

    
    db_file_path = os.path.join(base_path, "reports", "grafana", "report.db")
    
    
    output_csv_path = os.path.join(base_path, "scripts", "requests_responses.csv")


    try:
        
        conn = sqlite3.connect(db_file_path)

        query = """
        SELECT * FROM requests 
        INNER JOIN responses ON requests.id = responses.id
        """

        print(f"Connecting to database at: {db_file_path}")

       
        df_totaal = pd.read_sql_query(query, conn)

    except sqlite3.Error as e:
        print(f"\n[ERROR] Could not connect to or process database: {e}")
        return 
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            print("Database connection closed.")


    
    df_totaal.to_csv(output_csv_path, index=False)
    print(f"\n[SUCCESS] Successfully exported {len(df_totaal)} records.")
    print(f"Output saved to: {output_csv_path}")


if __name__ == "__main__":
    run_analysis()
