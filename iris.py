# iris.py
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# -------------------------------------------------------
# 1. Load Dataset (Seaborn auto-downloads iris dataset)
# -------------------------------------------------------
df = sns.load_dataset("iris")

# -------------------------------------------------------
# 2. Encode target column
# -------------------------------------------------------
label = LabelEncoder()
df["species_encoded"] = label.fit_transform(df["species"])

# -------------------------------------------------------
# 3. Prepare X, y using numpy arrays
# -------------------------------------------------------
X = df.drop(["species", "species_encoded"], axis=1).values
y = df["species_encoded"].values

# -------------------------------------------------------
# 4. Train-Test Split
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------
# 5. Train SVM Model
# -------------------------------------------------------
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# -------------------------------------------------------
# 6. Save the trained model with pickle
# -------------------------------------------------------
with open("iris_svm.pkl", "wb") as f:
    pickle.dump(model, f)

# -------------------------------------------------------
# 7. Save label encoder too (important!)
# -------------------------------------------------------
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label, f)

print("🎉 Model saved successfully as iris_svm.pkl and label_encoder.pkl")
