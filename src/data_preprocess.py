import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess(input_path="data/raw/creditcard.csv",
               output_path="data/processed/creditcard_processed.csv"):

    print("Chargement du dataset...")
    df = pd.read_csv(input_path)

    print("Normalisation de 'Amount'...")
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    print("Création du dossier de sortie…")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Sauvegarde du fichier preprocessé...")
    df.to_csv(output_path, index=False)

    print("Préprocessing terminé :", output_path)


if __name__ == "__main__":
    preprocess()
