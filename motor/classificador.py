"""
Classificador de categoria (Machine Learning supervisionado).

Texto -> TF-IDF (palavras + pedaços de 3 a 5 letras) -> KNN ou SVM, os dois modelos vistos em aula.
O TF-IDF já sai NORMALIZADO (norma L2 = 1 em cada frase). É o mesmo cuidado do Wine Data Set: sem normalizar,
uma frase comprida "pesaria" mais que uma curta na distância do KNN.
"""
from pathlib import Path

import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from motor.texto import normalizar

ARQUIVO_MODELO = Path(__file__).resolve().parent.parent / "modelos" / "classificador.joblib"

# abaixo desta confiança a triagem diz "não tenho certeza" em vez de chutar (fora do escopo)
CONFIANCA_MINIMA = 0.35


def vetorizador() -> FeatureUnion:
    return FeatureUnion([
        ("palavras", TfidfVectorizer(preprocessor=normalizar, ngram_range=(1, 2), sublinear_tf=True)),
        # pedaços de palavra: "entupida"/"entupiu"/"entupimento" ficam parecidos, e erro de digitação pesa menos
        ("pedacos", TfidfVectorizer(preprocessor=normalizar, analyzer="char_wb", ngram_range=(3, 5),
                                    sublinear_tf=True)),
    ])


def criar_knn(k: int = 5) -> Pipeline:
    return Pipeline([("tfidf", vetorizador()),
                     ("modelo", KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance"))])


def criar_svm() -> Pipeline:
    return Pipeline([("tfidf", vetorizador()),
                     ("modelo", SVC(kernel="linear", C=1.0, probability=True, random_state=42))])


def salvar(modelo: Pipeline) -> None:
    ARQUIVO_MODELO.parent.mkdir(exist_ok=True)
    joblib.dump(modelo, ARQUIVO_MODELO)


def carregar() -> Pipeline:
    if not ARQUIVO_MODELO.exists():
        raise FileNotFoundError("Modelo não treinado. Rode: python treinar.py")
    return joblib.load(ARQUIVO_MODELO)


def prever(modelo: Pipeline, descricao: str) -> tuple:
    """Devolve (categoria ou None, confiança, top-3 [(categoria, probabilidade)])."""
    probs = modelo.predict_proba([descricao])[0]
    ranking = sorted(zip(modelo.classes_, probs), key=lambda x: x[1], reverse=True)
    categoria, confianca = ranking[0]
    top3 = [(c, round(float(p), 3)) for c, p in ranking[:3]]
    if confianca < CONFIANCA_MINIMA:
        return None, round(float(confianca), 3), top3
    return categoria, round(float(confianca), 3), top3
