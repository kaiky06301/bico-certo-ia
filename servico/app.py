"""
Serviço de IA do Bico Certo (FastAPI).

POST /triagem  -> categoria (ML + LLM), faixa de preço justo (RAG), serviços parecidos e orientação (LLM)
GET  /health   -> modelo carregado e Ollama no ar?

Rodar:  uvicorn servico.app:app --port 8000      Documentação: http://localhost:8000/docs
"""
from typing import List, Optional

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field

from motor import gerador
from motor.triagem import Triagem

app = FastAPI(title="Bico Certo IA", version="1.0.0",
              description="Triagem do problema do morador: categoria, preço justo do bairro e orientação.")
triagem = Triagem()


class PedidoTriagem(BaseModel):
    descricao: str = Field(..., max_length=1000, examples=["a pia da cozinha entupiu e a água não desce"])
    usar_llm: bool = Field(True, description="false = só ML + RAG, sem chamar o modelo de linguagem")


class Faixa(BaseModel):
    minimo: int
    maximo: int
    mediana: int
    amostras: int


class ServicoParecido(BaseModel):
    id: int
    descricao: str
    categoria: str
    complexidade: str
    bairro: str
    valor_pago: int
    nota: int
    similaridade: float


class RespostaTriagem(BaseModel):
    categoria: Optional[str]
    confianca: float
    alternativas: List[list]
    decidido_por: str
    aviso_seguranca: Optional[str] = None
    faixa: Optional[Faixa]
    servicos_parecidos: List[ServicoParecido]
    orientacao: str
    origem_orientacao: str


@app.post("/triagem", response_model=RespostaTriagem)
def triar(pedido: PedidoTriagem) -> dict:
    return triagem.executar(pedido.descricao, usar_llm=pedido.usar_llm)


@app.get("/health")
def health() -> dict:
    try:
        httpx.get(f"{gerador.OLLAMA_URL}/api/tags", timeout=2).raise_for_status()
        ollama = "UP"
    except httpx.HTTPError:
        ollama = "DOWN (respostas usam o texto padrão)"
    return {"status": "UP", "modelo_ml": type(triagem.modelo.named_steps["modelo"]).__name__,
            "llm": gerador.MODELO, "ollama": ollama, "servicos_no_historico": len(triagem.base.servicos)}
