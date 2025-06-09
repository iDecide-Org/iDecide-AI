import pandas as pd
import numpy as np
import re

# === Load raw dataset ===
df = pd.read_csv("data.csv", sep="\t", on_bad_lines='skip')  # replace with your actual path

# === Drop entries with missing majors or age under 13 ===
df = df.dropna(subset=["major"])
df = df[df["age"] >= 13]

# === Drop users who ticked fake vocabulary words (attention check) ===
fake_words = ["VCL6", "VCL9", "VCL12"]
df = df[(df[fake_words] == 0).all(axis=1)]

# === Compute RIASEC average scores ===
def average_columns(prefix):
    return df[[f"{prefix}{i}" for i in range(1, 9)]].mean(axis=1)

df["realistic"] = average_columns("R")
df["investigative"] = average_columns("I")
df["artistic"] = average_columns("A")
df["social"] = average_columns("S")
df["enterprising"] = average_columns("E")
df["conventional"] = average_columns("C")

# === Compute Big 5 scores ===
df["extraversion"] = df["TIPI1"] - df["TIPI6"]
df["agreeableness"] = df["TIPI7"] - df["TIPI2"]
df["conscientiousness"] = df["TIPI3"] - df["TIPI8"]
df["emotional_stability"] = df["TIPI9"] - df["TIPI4"]
df["openness"] = df["TIPI5"] - df["TIPI10"]

# === Clean 'major' values ===
def clean_major(text):
    text = text.strip().lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    return text

df["major"] = df["major"].astype(str).apply(clean_major)

# === Filter out bad patterns ===
bad_patterns = [
    r"^idk$",
    r"^i don't know$",
    r"^don't know$",
    r"^dont know$",
    r"^secret$",
    r"^yes$",
    r"^no$",
    r"^n/a$",
    r"^none$",
    r"^not sure$",
    r"^i did not attend university$",
    r"^did not attend university$",
    r"^i didn't go to university$",
    r"^college$",
    r"^school$",
    r"^education$",
    r"^unknown$",
    r"^na$",
    r"^uni$",
    r"^student$",
    r"^general$",
    r"^idont know$",
    r"^something$",
    r"^i dont know yet$",
    r"^nil$",
    r"^undecided$",
    r"^undecieded$",
    r"^undeclared$",
    r"^undetermined$",
    r"^.*\?$",
    r"\byet\b",         # contains the word 'yet'
    r"\bnot\b",         # contains the word 'not'
    r"\bdont\b",        # contains the word 'dont'
    r"\bknow\b",        # contains the word 'know'

]

pattern = re.compile("|".join(bad_patterns))
df = df[~df["major"].str.match(pattern)]

# === Remove super unique (rare) majors ===
major_counts = df["major"].value_counts()
common_majors = major_counts[major_counts > 1].index  # appears more than once
df = df[df["major"].isin(common_majors)]

# === Drop unused columns (optional) ===
columns_to_keep = [
    "realistic", "investigative", "artistic", "social",
    "enterprising", "conventional",
    "extraversion", "agreeableness", "conscientiousness",
    "emotional_stability", "openness", "major"
]
df_cleaned = df[columns_to_keep]

# === Save cleaned dataset ===
df_cleaned.to_csv("cleaned_riasec_big5_major.csv", index=False)
print("✅ Cleaned dataset saved as 'cleaned_riasec_big5_major.csv'")
