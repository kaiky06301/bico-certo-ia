"""Testes automatizados (pytest). O LLM é simulado com httpx.MockTransport: não precisa do Ollama."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from motor import classificador, gerador
from motor.rag import BaseDeConhecimento, FaixaPreco, Servico
from motor.texto import normalizar
from motor.triagem import Triagem


def llm_falso(resposta: str = "", status: int = 200):
    """Cliente HTTP que responde como o Ollama, sem rede."""
    def tratar(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"message": {"content": resposta}})
    return httpx.Client(transport=httpx.MockTransport(tratar))


@pytest.fixture(scope="module")
def triagem():
    return Triagem()


def test_normalizar_tira_acento_maiuscula_e_pontuacao():
    assert normalizar("  Tá VAZANDO água!!! ") == "ta vazando agua"


@pytest.mark.parametrize("descricao, esperado", [
    ("a pia da cozinha está entupida", "Encanamento"),
    ("o disjuntor desarma toda hora", "Eletrica"),
    ("preciso montar um guarda-roupa", "Montagem de moveis"),
    ("o carro não pega", "Mecanico"),
])
def test_classificador_acerta_casos_claros(triagem, descricao, esperado):
    categoria, confianca, top3 = classificador.prever(triagem.modelo, descricao)
    assert categoria == esperado
    assert 0 < confianca <= 1 and len(top3) == 3


def test_faixa_usa_quartis_dos_valores():
    servicos = [Servico(i, "x", "Eletrica", "baixa", "Centro", v, 5) for i, v in enumerate([100, 120, 140, 160, 300])]
    faixa = BaseDeConhecimento.faixa(servicos)
    assert (faixa.minimo, faixa.mediana, faixa.maximo, faixa.amostras) == (120, 140, 160, 5)


def test_rag_recupera_so_da_categoria(triagem):
    parecidos = triagem.base.recuperar("vazamento embaixo da pia", "Encanamento")
    assert parecidos and all(s.categoria == "Encanamento" for s in parecidos)
    assert parecidos == sorted(parecidos, key=lambda s: s.similaridade, reverse=True)


def test_checagem_barra_valor_inventado_e_exige_faixa():
    contexto = "Faixa: R$ 80 a R$ 150 (mediana R$ 120)"
    faixa = FaixaPreco(minimo=80, maximo=150, mediana=120, amostras=5)
    assert gerador.passa_na_checagem("Fica entre R$ 80 e R$ 150.", contexto, faixa)
    assert not gerador.passa_na_checagem("Fica uns R$ 500.", contexto, faixa)
    assert not gerador.passa_na_checagem("Chame um eletricista.", contexto, faixa)  # não citou a faixa


def test_valores_em_reais_entende_milhar_e_centavos():
    assert gerador.valores_em_reais("de R$ 1.200,00 a R$ 85") == {1200, 85}


def test_triagem_usa_texto_do_llm_quando_passa_na_checagem(triagem):
    base = triagem.executar("a pia da cozinha está entupida", usar_llm=False)
    faixa = base["faixa"]
    texto = f"Parece entupimento. Vizinhos pagaram entre R$ {faixa['minimo']} e R$ {faixa['maximo']}."
    r = triagem.executar("a pia da cozinha está entupida", cliente=llm_falso(texto))
    assert r["origem_orientacao"] == "llm" and r["orientacao"] == texto


def test_triagem_descarta_texto_com_valor_inventado(triagem):
    r = triagem.executar("a pia da cozinha está entupida", cliente=llm_falso("Custa R$ 9999."))
    assert r["origem_orientacao"] == "padrao-checagem"
    assert "R$ 9999" not in r["orientacao"]


def test_triagem_continua_quando_ollama_cai(triagem):
    r = triagem.executar("a pia da cozinha está entupida", cliente=llm_falso(status=500))
    assert r["categoria"] == "Encanamento"
    assert r["origem_orientacao"] == "padrao-falha-llm"
    assert r["faixa"]["minimo"] < r["faixa"]["maximo"]


def test_texto_vazio_pede_mais_detalhes(triagem):
    r = triagem.executar("   ")
    assert r["categoria"] is None and r["origem_orientacao"] == "texto-insuficiente"


def test_llm_diz_nenhuma_vira_fora_do_escopo(triagem):
    r = triagem.executar("quero pedir uma pizza", cliente=llm_falso("NENHUMA"))
    assert r["categoria"] is None


def test_aviso_de_seguranca_e_regra_fixa(triagem):
    assert triagem.executar("saiu faísca da tomada", usar_llm=False)["aviso_seguranca"]
    assert triagem.executar("quero pintar o quarto", usar_llm=False)["aviso_seguranca"] is None


def test_api_triagem_sem_llm():
    from servico.app import app
    cliente = TestClient(app)
    r = cliente.post("/triagem", json={"descricao": "o chuveiro elétrico queimou", "usar_llm": False})
    assert r.status_code == 200
    corpo = r.json()
    assert corpo["categoria"] == "Eletrica" and corpo["faixa"]["amostras"] >= 3
    assert cliente.post("/triagem", json={}).status_code == 422
