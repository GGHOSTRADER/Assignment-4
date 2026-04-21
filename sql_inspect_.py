import sqlite3
import pandas as pd

conn = sqlite3.connect("ncu_regulations.db")

df = pd.read_sql_query("SELECT * FROM articles ", conn)


# print(df[df["article_number"] == "Article 47"])
# print(df[["content"]].iloc[54])

# print(df[["article_number"]].tail(2))
print(df.columns)
df = pd.read_sql_query("SELECT * FROM regulations ", conn)

print(df)
# print(df[["reg_id", "category"]].tail(2))

conn.close()
