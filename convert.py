import pandas as pd

df = pd.read_csv('data.csv')
df = df.fillna("")

text_col = []
for _, row in df.iterrows():
    prompt = "Below is a question which requires an answer. Write an answer that is as accurate as possible to provide the best possible results in the most effecient amount of code to do so. \n\n"
    instruction = str(row["Instruction"])
    input_query = str(row["Question"])
    response = str(row["Answer"])

    if len(input_query.strip()) == 0:
        text = prompt + "### Instruction:\n" + instruction + "\n### Response:\n" + response
    else:
        text = (prompt + "### Instruction:\n" + instruction + "\n### Input:\n" + input_query + "\n### Response:\n" + response)

    text_col.append(text)

df.loc[:, "text"] = text_col
print(df.head())

df.to_csv("train.csv", index=False)




