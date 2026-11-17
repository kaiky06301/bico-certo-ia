"""
IA generativa: o modelo qwen2.5:3b (rodando local no Ollama) escreve a orientação para o morador.

Guardrails (uso responsável):
- o prompt manda usar SÓ os serviços recuperados e a faixa calculada;
- todo valor em R$ que aparece na resposta tem que existir no contexto (senão o texto é descartado);
- se o Ollama estiver fora do ar, demorar ou falhar na checagem, entra um texto padrão montado com os dados.
"""
import os
import re
from typing import List, Optional, Tuple

import httpx

from motor.rag import FaixaPreco, Servico

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODELO = os.getenv("OLLAMA_MODELO", "qwen2.5:3b")
TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "60"))

SISTEMA = (
    "Você é o assistente do Bico Certo, um app que conecta moradores a prestadores de serviço do bairro. "
    "Escreva em português do Brasil um único parágrafo corrido (sem títulos, listas ou rótulos), com no máximo "
    "4 frases curtas, falando direto com o morador (use \"você\"), para alguém sem conhecimento técnico. "
    "Diga o que provavelmente é o problema, a faixa de preço justo e UMA pergunta que o morador deve fazer ao "
    "prestador antes de fechar. Use SOMENTE os valores em R$ que aparecem no CONTEXTO. "
    "Cite a faixa exatamente como está no contexto (\"entre R$ X e R$ Y\"). "
    "Não invente valores, marcas, prazos ou diagnósticos certeiros."
)


def montar_contexto(categoria: str, faixa: FaixaPreco, servicos: List[Servico]) -> str:
    linhas = [f"Categoria: {categoria}",
              f"Faixa de preço justo: R$ {faixa.minimo} a R$ {faixa.maximo} (mediana R$ {faixa.mediana}, "
              f"{faixa.amostras} serviços parecidos)",
              "Serviços parecidos já feitos no bairro:"]
    for s in servicos:
        linhas.append(f"- \"{s.descricao}\" em {s.bairro}: pago R$ {s.valor_pago}, nota {s.nota}/5")
    return "\n".join(linhas)


def valores_em_reais(texto: str) -> set:
    """Extrai os números que aparecem depois de "R$" (ex.: "R$ 1.200,00" -> 1200)."""
    achados = re.findall(r"R\$\s*([\d.]+(?:,\d{1,2})?)", texto)
    return {int(float(v.replace(".", "").replace(",", "."))) for v in achados}


def passa_na_checagem(resposta: str, contexto: str, faixa: Optional[FaixaPreco] = None) -> bool:
    """Todo R$ da resposta existe no contexto e, se a faixa for informada, ela foi citada (mínimo e máximo)."""
    valores = valores_em_reais(resposta)
    if not resposta.strip() or not valores <= valores_em_reais(contexto):
        return False
    return faixa is None or {faixa.minimo, faixa.maximo} <= valores


def texto_padrao(categoria: str, faixa: FaixaPreco) -> str:
    return (f"Pelo que você descreveu, parece um serviço de {categoria.lower()}. "
            f"Vizinhos pagaram entre R$ {faixa.minimo} e R$ {faixa.maximo} em serviços parecidos. "
            "Antes de fechar, pergunte ao prestador o que está incluso (material, mão de obra e deslocamento).")


def gerar(descricao: str, categoria: str, faixa: FaixaPreco, servicos: List[Servico],
          cliente: Optional[httpx.Client] = None) -> Tuple[str, str]:
    """Devolve (texto, origem). Origem: "llm", "padrao-falha-llm" ou "padrao-checagem"."""
    contexto = montar_contexto(categoria, faixa, servicos)
    corpo = {
        "model": MODELO,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 220},
        "messages": [
            {"role": "system", "content": SISTEMA},
            {"role": "user", "content": f"CONTEXTO:\n{contexto}\n\nPROBLEMA DO MORADOR: {descricao}"},
        ],
    }
    try:
        http = cliente or httpx.Client(timeout=TIMEOUT)
        resposta = http.post(f"{OLLAMA_URL}/api/chat", json=corpo)
        resposta.raise_for_status()
        texto = resposta.json()["message"]["content"].strip()
    except (httpx.HTTPError, KeyError, ValueError):
        return texto_padrao(categoria, faixa), "padrao-falha-llm"
    if not passa_na_checagem(texto, contexto, faixa):
        return texto_padrao(categoria, faixa), "padrao-checagem"
    return texto, "llm"


def classificar(descricao: str, opcoes: List[str], cliente: Optional[httpx.Client] = None) -> Tuple[bool, Optional[str]]:
    """
    LLM como classificador (zero-shot): escolhe UMA das opções ou "NENHUMA".
    Devolve (respondeu, categoria):
      (True, "Encanamento") -> escolheu uma opção;  (True, None) -> disse NENHUMA (fora do escopo);
      (False, None) -> Ollama fora do ar ou resposta fora da lista (quem chamou mantém o palpite do ML).
    """
    lista = "\n".join(f"- {o}" for o in opcoes)
    corpo = {
        "model": MODELO,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 12},
        "messages": [
            {"role": "system", "content":
                "Você classifica pedidos de serviço doméstico ou automotivo de um app de bairro. "
                "Responda SOMENTE com o nome exato de uma categoria da lista, ou NENHUMA se o pedido não for "
                "um serviço de nenhuma delas (ex.: comida, saúde, aulas, assuntos jurídicos)."},
            {"role": "user", "content": f"Categorias:\n{lista}\n\nPedido: {descricao}\n\nCategoria:"},
        ],
    }
    try:
        http = cliente or httpx.Client(timeout=TIMEOUT)
        resposta = http.post(f"{OLLAMA_URL}/api/chat", json=corpo)
        resposta.raise_for_status()
        bruto = resposta.json()["message"]["content"].strip().strip(".\"'- ").lower()
    except (httpx.HTTPError, KeyError, ValueError):
        return False, None
    if bruto.startswith("nenhuma"):
        return True, None
    for opcao in opcoes:
        if bruto == opcao.lower() or bruto.startswith(opcao.lower()):
            return True, opcao
    return False, None
