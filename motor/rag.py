"""
RAG básico (Retrieval-Augmented Generation).

1. RECUPERAÇÃO: acha no histórico do bairro os serviços mais parecidos com o problema descrito
   (TF-IDF + similaridade de cosseno, só dentro da categoria prevista).
2. AUMENTO: esses serviços (descrição, bairro, valor pago, nota) viram o CONTEXTO do prompt.
3. GERAÇÃO: o modelo generativo escreve a orientação usando só esse contexto (motor/gerador.py).

A faixa de preço justo sai dos dados recuperados (quartis dos valores pagos), não da cabeça do modelo.
"""
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import quantiles
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from motor.texto import normalizar

ARQUIVO_HISTORICO = Path(__file__).resolve().parent.parent / "dados" / "historico_servicos.csv"
SIMILARIDADE_MINIMA = 0.10


@dataclass
class Servico:
    id: int
    descricao: str
    categoria: str
    complexidade: str
    bairro: str
    valor_pago: int
    nota: int
    similaridade: float = 0.0


@dataclass
class FaixaPreco:
    minimo: int
    maximo: int
    mediana: int
    amostras: int


class BaseDeConhecimento:
    def __init__(self, arquivo: Path = ARQUIVO_HISTORICO):
        with open(arquivo, encoding="utf-8") as f:
            self.servicos = [Servico(int(l["id"]), l["descricao"], l["categoria"], l["complexidade"],
                                     l["bairro"], int(l["valor_pago"]), int(l["nota"]))
                             for l in csv.DictReader(f)]
        self.vetorizador = TfidfVectorizer(preprocessor=normalizar, analyzer="char_wb", ngram_range=(3, 5),
                                           sublinear_tf=True)
        self.matriz = self.vetorizador.fit_transform([s.descricao for s in self.servicos])

    def recuperar(self, descricao: str, categoria: str, k: int = 5) -> List[Servico]:
        sims = cosine_similarity(self.vetorizador.transform([descricao]), self.matriz)[0]
        candidatos = [(i, s) for i, s in enumerate(sims)
                      if self.servicos[i].categoria == categoria and s >= SIMILARIDADE_MINIMA]
        candidatos.sort(key=lambda x: x[1], reverse=True)
        resultado = []
        for i, s in candidatos[:k]:
            serv = self.servicos[i]
            resultado.append(Servico(serv.id, serv.descricao, serv.categoria, serv.complexidade, serv.bairro,
                                     serv.valor_pago, serv.nota, round(float(s), 3)))
        return resultado

    @staticmethod
    def faixa(servicos: List[Servico]) -> Optional[FaixaPreco]:
        """Faixa justa = do 1º ao 3º quartil do que os vizinhos pagaram em serviços parecidos."""
        valores = sorted(s.valor_pago for s in servicos)
        if not valores:
            return None
        if len(valores) < 3:
            return FaixaPreco(valores[0], valores[-1], valores[len(valores) // 2], len(valores))
        q1, q2, q3 = quantiles(valores, n=4, method="inclusive")
        arred = lambda v: int(round(v / 5) * 5)
        return FaixaPreco(arred(q1), arred(q3), arred(q2), len(valores))
