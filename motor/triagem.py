"""Fluxo completo: descrição do morador -> categoria (ML) -> serviços parecidos + faixa (RAG) -> orientação (LLM)."""
from dataclasses import asdict
from typing import Optional

import httpx

from motor import classificador, gerador
from motor.rag import BaseDeConhecimento
from motor.texto import normalizar

# abaixo desta confiança do classificador, o LLM é consultado para decidir a categoria
LIMIAR_HIBRIDO = 0.60
# faixa só com os parecidos se houver pelo menos isto de amostras; senão usa a categoria inteira
AMOSTRAS_MINIMAS = 3

# aviso de segurança é REGRA FIXA: não depende do modelo de linguagem lembrar de avisar
PALAVRAS_DE_RISCO = ("choque", "faisca", "fogo", "incendio", "queimado", "fumaca", "gas", "curto",
                     "alagou", "alagando", "estourou", "derreteu")
AVISO_SEGURANCA = ("Atenção: pode haver risco. Desligue o disjuntor ou o registro geral e não mexa sozinho até o "
                   "prestador chegar. Em caso de fogo ou cheiro de gás, ligue 193.")


def tem_risco(descricao: str) -> bool:
    palavras = set(normalizar(descricao).split())
    return any(p in palavras for p in PALAVRAS_DE_RISCO)


class Triagem:
    def __init__(self, modelo=None, base: Optional[BaseDeConhecimento] = None):
        self.modelo = modelo or classificador.carregar()
        self.base = base or BaseDeConhecimento()

    def executar(self, descricao: str, usar_llm: bool = True, cliente: Optional[httpx.Client] = None,
                 gerar_orientacao: bool = True) -> dict:
        if len(normalizar(descricao).replace(" ", "")) < 5:
            return {"categoria": None, "confianca": 0.0, "alternativas": [], "decidido_por": "validacao",
                    "faixa": None, "servicos_parecidos": [],
                    "orientacao": "Descreva o problema com pelo menos algumas palavras (o que aconteceu e onde).",
                    "origem_orientacao": "texto-insuficiente"}
        categoria, confianca, top3 = classificador.prever(self.modelo, descricao)
        decidido_por = "ml"
        # modo híbrido: ML inseguro -> o LLM decide entre as 3 categorias mais prováveis (ou nenhuma)
        if usar_llm and confianca < LIMIAR_HIBRIDO:
            respondeu, escolha = gerador.classificar(descricao, [c for c, _ in top3], cliente)
            if respondeu:
                categoria, decidido_por = escolha, "llm"
            else:
                decidido_por = "ml-llm-indisponivel"
        if categoria is None:
            return {"categoria": None, "confianca": confianca, "alternativas": top3,
                    "faixa": None, "servicos_parecidos": [],
                    "decidido_por": decidido_por,
                    "orientacao": "Não consegui identificar o tipo de serviço. Escolha a categoria manualmente "
                                  "ou descreva com mais detalhes (o que quebrou, onde e desde quando).",
                    "origem_orientacao": "fora-do-escopo"}

        servicos = self.base.recuperar(descricao, categoria)
        if not servicos:  # nada parecido: usa todos da categoria para a faixa
            servicos = self.base.recuperar(categoria, categoria, k=10) or \
                       [s for s in self.base.servicos if s.categoria == categoria][:10]
        faixa = BaseDeConhecimento.faixa(servicos)
        if faixa.amostras < AMOSTRAS_MINIMAS or faixa.minimo == faixa.maximo:
            faixa = BaseDeConhecimento.faixa([s for s in self.base.servicos if s.categoria == categoria])

        if not gerar_orientacao:
            texto, origem = "", "nao-solicitada"
        elif usar_llm:
            texto, origem = gerador.gerar(descricao, categoria, faixa, servicos, cliente)
        else:
            texto, origem = gerador.texto_padrao(categoria, faixa), "padrao-sem-llm"

        aviso = AVISO_SEGURANCA if tem_risco(descricao) else None
        return {"categoria": categoria, "confianca": confianca, "alternativas": top3,
                "aviso_seguranca": aviso, "decidido_por": decidido_por, "faixa": asdict(faixa), "servicos_parecidos": [asdict(s) for s in servicos],
                "orientacao": texto, "origem_orientacao": origem}
