import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# === DESCARGAS INICIALES ===
def setup_nltk():
    """Descarga los recursos de NLTK necesarios (solo la primera vez)."""
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)


# === FUNCIÓN DE LIMPIEZA DE TEXTO ===
def preprocess_text(text):
    """Limpia y normaliza un texto: minúsculas, tokenización, stopwords, lematización."""
    text = text.lower()
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


# === FUNCIÓN PARA CARGAR Y COMBINAR LOS ARCHIVOS ===
def load_raw_data(raw_folder, files):
    """Carga varios archivos de texto etiquetados y los combina en un único DataFrame."""
    dfs = []
    for fname in files:
        path = os.path.join(raw_folder, fname)
        df = pd.read_csv(path, sep='\t', header=None, names=['sentence', 'label'])
        dfs.append(df)
    df_raw = pd.concat(dfs, ignore_index=True)
    return df_raw


# === FUNCIÓN PRINCIPAL DE PREPROCESAMIENTO ===
def run_preprocessing(raw_folder='../data/raw/',
                      processed_path='../data/processed/clean_reviews.csv'):
    """Ejecuta todo el pipeline: carga, limpieza, preprocesamiento y guardado."""
    setup_nltk()

    # Archivos de origen
    files = [
        'imdb_labelled.txt',
        'yelp_labelled.txt',
        'amazon_cells_labelled.txt'
    ]

    print("Cargando datasets...")
    df_raw = load_raw_data(raw_folder, files)
    print("Dataset cargado:", df_raw.shape)

    # Eliminar duplicados y valores nulos
    df_raw.drop_duplicates(inplace=True)
    df_raw.dropna(inplace=True)

    #  texto
    print("Limpieza y normalización del texto...")
    df_raw['clean_sentence'] = df_raw['sentence'].apply(preprocess_text)

    df = df_raw[['clean_sentence', 'label']]
    df = df.dropna(subset=['clean_sentence'])
    df = df[df['clean_sentence'].str.strip() != '']

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)

    print(f"Dataset preprocesado guardado en: {processed_path}")
    print(f"Total de filas: {df.shape[0]}")
    print("Ejemplo de texto limpio:")
    print(df.sample(3, random_state=42))
    return df


if __name__ == "__main__":
    run_preprocessing()
