"""
Treina o classificador de categoria e escolhe entre KNN e SVM.

A escolha usa validação cruzada POR GRUPO no treino (GroupKFold, 5 partes): as 6 variações de um mesmo
modelo de frase ficam sempre do mesmo lado, então o modelo é medido em frases que ele nunca viu.
Depois mede na validação escrita à mão (validacao_manual.csv). O teste final congelado fica para avaliar.py.

Uso:  python treinar.py
"""
import csv
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, KFold, cross_val_score

from dados.gerar_dados import gerar
from motor import classificador

RAIZ = Path(__file__).parent
RESULTADOS = RAIZ / "resultados"


def ler(arquivo: str):
    with open(RAIZ / "dados" / arquivo, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    return [l["descricao"] for l in linhas], [l["categoria"] for l in linhas], [l.get("grupo") for l in linhas]


def main() -> None:
    gerar()  # sempre regera: semente fixa, então sai igual
    x_treino, y_treino, grupos = ler("descricoes_treino.csv")
    x_teste, y_teste, _ = ler("validacao_manual.csv")
    print(f"Treino: {len(x_treino)} frases simuladas | Validação: {len(x_teste)} frases escritas à mão")

    # comparação didática: KFold comum (otimista, variações da mesma frase caem no treino e no teste)
    ingenua = cross_val_score(classificador.criar_svm(), x_treino, y_treino,
                              cv=KFold(5, shuffle=True, random_state=42)).mean()
    print(f"  (KFold comum, SVM: {ingenua:.2%} - otimista demais, não usado para escolher)")
    candidatos = {f"KNN k={k}": classificador.criar_knn(k) for k in (1, 3, 5, 7, 9, 11, 15)}
    candidatos["SVM linear"] = classificador.criar_svm()

    validacao = {}
    for nome, modelo in candidatos.items():
        notas = cross_val_score(modelo, x_treino, y_treino, groups=grupos, cv=GroupKFold(n_splits=5),
                                scoring="accuracy")
        validacao[nome] = (notas.mean(), notas.std())
        print(f"  {nome:<11} validação cruzada: {notas.mean():.2%} ± {notas.std():.2%}")

    # desempate: maior média; se empatar, menor desvio
    escolhido = max(validacao, key=lambda n: (round(validacao[n][0], 4), -validacao[n][1]))
    print(f"Escolhido pela validação cruzada por grupo: {escolhido}")

    resultados_teste = {}
    for nome, modelo in candidatos.items():
        modelo.fit(x_treino, y_treino)
        resultados_teste[nome] = accuracy_score(y_teste, modelo.predict(x_teste))

    final = candidatos[escolhido]
    previsto = final.predict(x_teste)
    classificador.salvar(final)

    erros = [{"descricao": d, "esperado": e, "previsto": p}
             for d, e, p in zip(x_teste, y_teste, previsto) if e != p]
    rotulos = sorted(set(y_treino))
    metricas = {
        "modelo_escolhido": escolhido,
        "kfold_comum_svm_otimista": round(ingenua, 4),
        "validacao_cruzada": {n: {"media": round(m, 4), "desvio": round(s, 4)} for n, (m, s) in validacao.items()},
        "acuracia_validacao_manual": {n: round(a, 4) for n, a in resultados_teste.items()},
        "relatorio_validacao_manual": classification_report(y_teste, previsto, output_dict=True, zero_division=0),
        "matriz_confusao": {"rotulos": rotulos, "valores": confusion_matrix(y_teste, previsto, labels=rotulos).tolist()},
        "erros_validacao_manual": erros,
    }
    RESULTADOS.mkdir(exist_ok=True)
    (RESULTADOS / "metricas_classificador.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2),
                                                           encoding="utf-8")
    print(f"Validação manual ({escolhido}): {resultados_teste[escolhido]:.2%} "
          f"({len(x_teste) - len(erros)}/{len(x_teste)})")
    print("Modelo salvo em modelos/classificador.joblib")


if __name__ == "__main__":
    main()
