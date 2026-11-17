"""
Avaliação final (roda UMA vez sobre o teste congelado dados/teste_final.csv).

1. Classificação: regras x ML (SVM) x LLM sozinho x híbrido (ML + LLM quando o ML está inseguro)
2. RAG: a faixa de preço dos serviços parecidos cobre o valor real? (leave-one-out no histórico)
3. Geração: quantas respostas do LLM passam na checagem de valores, com exemplos
4. Falhas: Ollama fora do ar, texto vazio e pedido fora do escopo

Precisa do Ollama com o qwen2.5:3b rodando (ollama serve). Uso:  python avaliar.py
"""
import csv
import json
import time
from pathlib import Path
from statistics import mean

from motor import classificador, gerador
from motor.rag import BaseDeConhecimento
from motor.texto import normalizar
from motor.triagem import Triagem

RAIZ = Path(__file__).parent
RESULTADOS = RAIZ / "resultados"
FORA = "FORA_DO_ESCOPO"

# baseline de regras: palavras-chave tiradas das descrições das categorias na API Java
PALAVRAS = {
    "Eletrica": "tomada interruptor disjuntor chuveiro eletrico fiacao curto lampada ventilador luz energia",
    "Encanamento": "vazamento vazando torneira registro pia ralo entupido descarga caixa agua cano",
    "Limpeza": "faxina limpeza limpar vidros estofados organizacao sujo",
    "Pintura": "pintura pintar parede teto portao grade massa corrida tinta",
    "Montagem de moveis": "montagem montar desmontar guarda roupa cama rack estante moveis",
    "Pequenos reparos": "prateleira cortina suporte tv fechadura instalar",
    "Pedreiro": "reboco piso azulejo rejunte contrapiso muro rachadura reforma alvenaria",
    "Marceneiro": "madeira porta gaveta dobradica prateleira movel sob medida",
    "Mecanico": "carro moto bateria oleo freio pneu pega revisao",
}


def regras(descricao: str):
    palavras = set(normalizar(descricao).split())
    pontos = {c: len(palavras & set(p.split())) for c, p in PALAVRAS.items()}
    melhor = max(pontos, key=pontos.get)
    return melhor if pontos[melhor] > 0 else None


def ler_teste():
    with open(RAIZ / "dados" / "teste_final.csv", encoding="utf-8") as f:
        return [(l["descricao"], l["categoria"]) for l in csv.DictReader(f)]


def medir(nome, funcao, casos):
    certos_escopo = certos_fora = 0
    no_escopo = [c for c in casos if c[1] != FORA]
    fora = [c for c in casos if c[1] == FORA]
    erros, tempos = [], []
    for descricao, esperado in casos:
        inicio = time.perf_counter()
        previsto = funcao(descricao) or FORA
        tempos.append(time.perf_counter() - inicio)
        if previsto == esperado:
            if esperado == FORA:
                certos_fora += 1
            else:
                certos_escopo += 1
        else:
            erros.append({"descricao": descricao, "esperado": esperado, "previsto": previsto})
    total = certos_escopo + certos_fora
    linha = {
        "abordagem": nome,
        "acuracia_geral": round(total / len(casos), 4),
        "acuracia_no_escopo": round(certos_escopo / len(no_escopo), 4),
        "fora_do_escopo_detectado": f"{certos_fora}/{len(fora)}",
        "tempo_medio_ms": round(mean(tempos) * 1000, 1),
        "erros": erros,
    }
    print(f"  {nome:<26} geral {linha['acuracia_geral']:.1%} | no escopo {linha['acuracia_no_escopo']:.1%} | "
          f"fora detectado {linha['fora_do_escopo_detectado']} | {linha['tempo_medio_ms']} ms")
    return linha


def avaliar_rag(base: BaseDeConhecimento):
    """
    Leave-one-out no histórico: esconde cada serviço, recupera os 5 mais parecidos (sem ele) e compara o preço
    sugerido (mediana dos parecidos) com o valor que foi pago de verdade.
    Linha de base sem RAG: mediana da categoria inteira (o que se faria sem olhar a descrição).
    """
    erros_rag, erros_cat, erros_pct_rag, erros_pct_cat = [], [], [], []
    for alvo in base.servicos:
        parecidos = [s for s in base.recuperar(alvo.descricao, alvo.categoria, k=6) if s.id != alvo.id][:5]
        da_categoria = [s for s in base.servicos if s.categoria == alvo.categoria and s.id != alvo.id]
        f_rag = BaseDeConhecimento.faixa(parecidos) or BaseDeConhecimento.faixa(da_categoria)
        f_cat = BaseDeConhecimento.faixa(da_categoria)
        erros_rag.append(abs(f_rag.mediana - alvo.valor_pago))
        erros_cat.append(abs(f_cat.mediana - alvo.valor_pago))
        erros_pct_rag.append(erros_rag[-1] / alvo.valor_pago)
        erros_pct_cat.append(erros_cat[-1] / alvo.valor_pago)
    resultado = {
        "servicos_avaliados": len(base.servicos),
        "rag_erro_medio_reais": round(mean(erros_rag), 1), "rag_erro_medio_pct": round(mean(erros_pct_rag), 4),
        "sem_rag_erro_medio_reais": round(mean(erros_cat), 1), "sem_rag_erro_medio_pct": round(mean(erros_pct_cat), 4),
    }
    print(f"  Com RAG (mediana dos 5 parecidos): erra em média R$ {resultado['rag_erro_medio_reais']} "
          f"({resultado['rag_erro_medio_pct']:.1%})")
    print(f"  Sem RAG (mediana da categoria):    erra em média R$ {resultado['sem_rag_erro_medio_reais']} "
          f"({resultado['sem_rag_erro_medio_pct']:.1%})")
    return resultado


def main() -> None:
    casos = ler_teste()
    modelo = classificador.carregar()
    base = BaseDeConhecimento()
    triagem = Triagem(modelo, base)
    categorias = list(modelo.classes_)
    print(f"Teste final: {len(casos)} frases ({sum(c[1] == FORA for c in casos)} fora do escopo)\n")

    print("1. Classificação")
    tabela = [
        medir("Regras (palavras-chave)", regras, casos),
        medir("ML (TF-IDF + SVM)", lambda d: classificador.prever(modelo, d)[0], casos),
        medir("LLM sozinho (qwen2.5:3b)", lambda d: gerador.classificar(d, categorias)[1], casos),
        medir("Híbrido (ML + LLM)", lambda d: triagem.executar(d, gerar_orientacao=False)["categoria"], casos),
    ]

    print("\n2. RAG: faixa de preço")
    rag = avaliar_rag(base)

    print("\n3. Geração com checagem de valores")
    exemplos = []
    for descricao, esperado in [c for c in casos if c[1] != FORA][::6]:
        r = triagem.executar(descricao)
        exemplos.append({"descricao": descricao, "categoria": r["categoria"], "faixa": r["faixa"],
                         "origem": r["origem_orientacao"], "orientacao": r["orientacao"]})
    origens = {}
    for e in exemplos:
        origens[e["origem"]] = origens.get(e["origem"], 0) + 1
    print(f"  {len(exemplos)} respostas: {origens}")

    print("\n4. Situações de falha")
    falhas = []
    url_original = gerador.OLLAMA_URL
    gerador.OLLAMA_URL = "http://localhost:9"  # porta sem serviço: simula o Ollama fora do ar
    r = triagem.executar("a torneira da cozinha está pingando")
    gerador.OLLAMA_URL = url_original
    falhas.append({"caso": "Ollama fora do ar", "categoria": r["categoria"], "origem": r["origem_orientacao"],
                   "ok": r["categoria"] == "Encanamento" and r["origem_orientacao"] == "padrao-falha-llm"})
    r = triagem.executar("")
    falhas.append({"caso": "Texto vazio", "categoria": r["categoria"], "origem": r["origem_orientacao"],
                   "ok": r["categoria"] is None and r["origem_orientacao"] == "texto-insuficiente"})
    r = triagem.executar("quero pedir uma pizza de calabresa")
    falhas.append({"caso": "Fora do escopo (pizza)", "categoria": r["categoria"], "origem": r["origem_orientacao"],
                   "ok": r["categoria"] is None})
    contexto = "Faixa de preço justo: R$ 80 a R$ 150"
    falhas.append({"caso": "Checagem barra valor inventado",
                   "ok": not gerador.passa_na_checagem("Custa uns R$ 500.", contexto)
                   and gerador.passa_na_checagem("Fica entre R$ 80 e R$ 150.", contexto)})
    for f in falhas:
        print(f"  {'OK ' if f['ok'] else 'FALHOU'} {f['caso']}")

    RESULTADOS.mkdir(exist_ok=True)
    (RESULTADOS / "avaliacao_final.json").write_text(json.dumps(
        {"classificacao": tabela, "rag": rag, "geracao": exemplos, "falhas": falhas},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nResultados em resultados/avaliacao_final.json")


if __name__ == "__main__":
    main()
