import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(input_path="data/raw/creditcard.csv",
               output_path="data/processed/creditcard_processed.csv"):
    
    print("Chargement du dataset...")
    df = pd.read_csv(input_path)

    print("Normalisation de 'Amount'...")
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    print("Sauvegarde du fichier preprocessé...")
    df.to_csv(output_path, index=False)

    print("Préprocessing terminé. Fichier généré :", output_path)

if __name__ == "__main__":
    preprocess()
