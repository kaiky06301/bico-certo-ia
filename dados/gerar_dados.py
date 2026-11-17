"""
Gera as bases SIMULADAS do Bico Certo (semente fixa = sempre os mesmos arquivos).

- descricoes_treino.csv : descrições de problemas da casa com a categoria certa (treino do classificador)
- historico_servicos.csv : serviços já feitos no bairro com o valor pago (base de conhecimento do RAG)

As categorias e as faixas de preço são as mesmas da API Java (migrations V2 e V10).
Os conjuntos de avaliação NÃO são gerados aqui, foram escritos à mão com outras palavras:
- validacao_manual.csv : usado para diagnosticar erros e ajustar o vocabulário do treino;
- teste_final.csv      : congelado, usado UMA vez no fim (inclui frases fora do escopo do app).

Uso:  python dados/gerar_dados.py
"""
import csv
import random
from pathlib import Path

SEMENTE = 2026
PASTA = Path(__file__).parent

# categoria -> (preço mínimo, preço máximo) iguais à API Java
CATEGORIAS = {
    "Eletrica": (80, 350),
    "Encanamento": (70, 400),
    "Limpeza": (120, 300),
    "Pintura": (150, 900),
    "Montagem de moveis": (80, 300),
    "Pequenos reparos": (60, 250),
    "Pedreiro": (150, 1500),
    "Marceneiro": (100, 800),
    "Mecanico": (80, 600),
}

# problema -> complexidade (baixa/media/alta). Cada categoria tem vocabulário próprio.
PROBLEMAS = {
    "Eletrica": [
        ("o quadro de energia está esquentando", "media"),
        ("a energia da casa cai quando ligo o chuveiro", "media"),
        ("preciso trocar o interruptor e o espelho da tomada", "baixa"),
        ("quero instalar lustre, luminária e spots de led", "media"),
        ("a resistência do chuveiro queimou e só sai água gelada", "baixa"),
        ("a fiação está velha, esquentando e com cheiro de queimado", "alta"),
        ("preciso de um eletricista para instalar o padrão de energia", "alta"),
        ("a campainha e o interfone pararam de funcionar", "baixa"),
        ("a tomada da cozinha parou de funcionar", "baixa"),
        ("o disjuntor cai toda hora", "media"),
        ("o chuveiro elétrico queimou a resistência", "baixa"),
        ("saiu faísca do interruptor da sala", "media"),
        ("preciso instalar um ventilador de teto", "media"),
        ("a luz do quarto fica piscando", "baixa"),
        ("deu curto-circuito e metade da casa ficou sem energia", "alta"),
        ("quero trocar a fiação antiga do apartamento", "alta"),
        ("preciso instalar uma tomada nova para o ar-condicionado", "media"),
        ("a lâmpada do banheiro não acende mesmo trocando", "baixa"),
    ],
    "Encanamento": [
        ("o encanamento está com vazamento e a conta de água subiu", "media"),
        ("o esgoto está voltando com mau cheiro", "media"),
        ("a máquina de lavar está vazando pela mangueira", "baixa"),
        ("o sifão da pia quebrou", "baixa"),
        ("o tanque da lavanderia está entupido", "baixa"),
        ("a boia da caixa d'água quebrou", "baixa"),
        ("tem água minando do chão do banheiro", "alta"),
        ("preciso de um encanador para trocar a torneira", "baixa"),
        ("a pia da cozinha está entupida", "baixa"),
        ("está vazando água embaixo da pia", "media"),
        ("a descarga não para de correr água", "baixa"),
        ("o ralo do banheiro voltou a entupir", "baixa"),
        ("a torneira está pingando sem parar", "baixa"),
        ("o cano estourou dentro da parede", "alta"),
        ("a caixa d'água está transbordando", "media"),
        ("o registro do chuveiro não fecha direito", "media"),
        ("tem infiltração de água vindo do apartamento de cima", "alta"),
        ("o vaso sanitário está entupido", "baixa"),
    ],
    "Limpeza": [
        ("preciso de uma diarista para faxina quinzenal", "baixa"),
        ("a casa está muito suja depois de uma festa", "media"),
        ("quero limpar o fogão, o forno e a geladeira", "baixa"),
        ("preciso tirar o pó e a sujeira depois da reforma", "alta"),
        ("quero lavar o tapete e tirar manchas", "baixa"),
        ("preciso de faxina pesada no quintal e na garagem", "media"),
        ("quero higienizar cadeiras e poltronas estofadas", "media"),
        ("preciso deixar o apartamento limpo para entregar as chaves", "media"),
        ("preciso de uma faxina completa na casa", "media"),
        ("quero uma limpeza pós-obra no apartamento", "alta"),
        ("preciso limpar os vidros e as janelas", "baixa"),
        ("o sofá está muito sujo e precisa de higienização", "media"),
        ("preciso organizar e limpar a casa antes da mudança", "media"),
        ("quero uma diarista para a limpeza semanal", "baixa"),
        ("a cozinha está engordurada e precisa de limpeza pesada", "media"),
        ("preciso limpar o colchão e o tapete", "baixa"),
    ],
    "Pintura": [
        ("quero trocar a cor da parede do quarto", "media"),
        ("o muro está sujo e pichado e precisa de tinta", "media"),
        ("a tinta está bolhando por causa da umidade", "media"),
        ("preciso de um pintor para pintar portas e janelas", "media"),
        ("o teto da cozinha está encardido de gordura e precisa pintar", "media"),
        ("quero aplicar textura e grafiato na parede", "media"),
        ("preciso de retoque de pintura nas paredes riscadas", "baixa"),
        ("quero envernizar o deck e pintar a varanda", "media"),
        ("quero pintar a sala e o corredor", "media"),
        ("a parede está descascando e precisa de pintura", "media"),
        ("preciso pintar o portão de ferro que enferrujou", "baixa"),
        ("quero pintar a casa inteira por dentro", "alta"),
        ("o teto do banheiro está com mofo e manchado", "media"),
        ("preciso passar massa corrida e pintar o quarto", "media"),
        ("quero pintar a grade da varanda", "baixa"),
        ("preciso pintar a fachada da casa", "alta"),
    ],
    "Montagem de moveis": [
        ("comprei móveis pela internet e vieram desmontados", "media"),
        ("preciso de um montador para mesa e cadeiras", "baixa"),
        ("quero montar o armário de cozinha que veio nas caixas", "media"),
        ("preciso montar a cômoda e o criado-mudo", "baixa"),
        ("quero montar o beliche e a sapateira", "baixa"),
        ("preciso de montador de móveis para o quarto inteiro", "alta"),
        ("quero montar e instalar o painel e o home theater", "media"),
        ("preciso desmontar e montar o guarda-roupa na casa nova", "media"),
        ("comprei um guarda-roupa e preciso montar", "media"),
        ("preciso montar uma cama box com cabeceira", "baixa"),
        ("quero montar o rack e o painel da TV", "baixa"),
        ("preciso desmontar os móveis para a mudança", "media"),
        ("comprei uma estante desmontada na loja", "baixa"),
        ("preciso montar a cozinha modulada que chegou em caixas", "alta"),
        ("quero montar uma escrivaninha e uma cadeira", "baixa"),
        ("preciso montar o berço do bebê", "baixa"),
    ],
    "Pequenos reparos": [
        ("preciso de um marido de aluguel para vários ajustes", "media"),
        ("quero instalar persiana e cortina blackout", "baixa"),
        ("preciso pendurar a televisão na parede com suporte", "baixa"),
        ("quero instalar acessórios no banheiro, como toalheiro e papeleira", "baixa"),
        ("preciso furar a parede e colocar buchas e ganchos", "baixa"),
        ("quero instalar tela de proteção na janela", "media"),
        ("o trinco do portão está solto", "baixa"),
        ("preciso trocar a maçaneta e o miolo da fechadura", "baixa"),
        ("preciso instalar uma prateleira na parede", "baixa"),
        ("quero colocar a cortina e o varão", "baixa"),
        ("preciso fixar o suporte da TV na parede", "baixa"),
        ("a fechadura da porta emperrou e preciso trocar", "media"),
        ("quero instalar um varal de teto", "baixa"),
        ("preciso pendurar quadros e espelho", "baixa"),
        ("o puxador do armário caiu", "baixa"),
        ("quero instalar um olho mágico e uma trava na porta", "media"),
    ],
    "Pedreiro": [
        ("preciso de um pedreiro para quebrar uma parede", "alta"),
        ("a laje está com trinca e infiltração", "alta"),
        ("quero assentar porcelanato e cerâmica", "alta"),
        ("a calçada e o piso externo estão quebrados", "media"),
        ("quero construir uma churrasqueira de alvenaria", "alta"),
        ("preciso subir parede de tijolo e bloco", "alta"),
        ("o piso afundou e está oco", "media"),
        ("preciso de massa, cimento e reboco na parede", "media"),
        ("apareceu uma rachadura grande na parede", "alta"),
        ("preciso trocar o piso da cozinha", "alta"),
        ("o azulejo do banheiro está soltando", "media"),
        ("preciso refazer o rejunte do box", "baixa"),
        ("quero levantar um muro no quintal", "alta"),
        ("o reboco da parede está esfarelando", "media"),
        ("preciso fazer um contrapiso na lavanderia", "media"),
        ("quero fazer uma pequena reforma no banheiro", "alta"),
    ],
    "Marceneiro": [
        ("o móvel de madeira está com cupim", "alta"),
        ("a porta de madeira está arrastando e precisa de ajuste", "baixa"),
        ("preciso de um marceneiro para consertar a cadeira", "baixa"),
        ("o tampo da mesa de madeira está rachado", "media"),
        ("quero um painel de madeira ripado", "alta"),
        ("a madeira do armário inchou com a umidade", "media"),
        ("quero uma bancada e um nicho de madeira sob medida", "alta"),
        ("o rodapé e o batente de madeira estão soltos", "baixa"),
        ("a porta do quarto empenou e não fecha", "media"),
        ("a gaveta da cômoda saiu do trilho", "baixa"),
        ("a dobradiça do armário quebrou", "baixa"),
        ("o pé da mesa de madeira está bambo", "baixa"),
        ("quero uma prateleira de madeira sob medida", "media"),
        ("o guarda-roupa de madeira está com a porta caída", "media"),
        ("preciso consertar o cupim no batente da porta", "alta"),
        ("quero um móvel sob medida para a lavanderia", "alta"),
    ],
    "Mecanico": [
        ("o carro está fazendo barulho na suspensão", "media"),
        ("a moto não dá partida", "media"),
        ("a luz de óleo e de injeção acendeu no painel", "media"),
        ("o carro morreu e preciso de uma chupeta na bateria", "baixa"),
        ("o carro está saindo fumaça do escapamento", "alta"),
        ("a embreagem está dura e patinando", "alta"),
        ("o carro está puxando para o lado e o volante treme", "media"),
        ("preciso trocar os pneus e fazer alinhamento", "media"),
        ("o carro não pega de manhã", "media"),
        ("a bateria do carro arriou", "baixa"),
        ("preciso trocar o óleo da moto", "baixa"),
        ("o freio do carro está fazendo barulho", "media"),
        ("o pneu furou e não tenho estepe", "baixa"),
        ("a moto está falhando e morrendo no meio da rua", "media"),
        ("preciso de uma revisão básica no carro", "media"),
        ("o motor do carro está esquentando demais", "alta"),
    ],
}

ABERTURAS = ["", "olá, ", "bom dia, ", "oi, ", "preciso de ajuda: ", "urgente: ", "por favor, "]
FECHAMENTOS = ["", ".", " hoje.", " o quanto antes.", " alguém pode vir ver?", " moro no térreo.",
               " é no fim de semana?", " quanto fica mais ou menos?"]
BAIRROS = ["Capão Redondo", "Jardim Ângela", "Campo Limpo", "Jardim São Luís", "Centro", "Grajaú"]

# onde o valor cai dentro da faixa da categoria, por complexidade
POSICAO_NA_FAIXA = {"baixa": (0.00, 0.30), "media": (0.25, 0.65), "alta": (0.60, 1.00)}


def frase(rng: random.Random, problema: str) -> str:
    texto = rng.choice(ABERTURAS) + problema + rng.choice(FECHAMENTOS)
    return texto[0].upper() + texto[1:]


def valor(rng: random.Random, categoria: str, complexidade: str) -> int:
    minimo, maximo = CATEGORIAS[categoria]
    ini, fim = POSICAO_NA_FAIXA[complexidade]
    bruto = minimo + (maximo - minimo) * rng.uniform(ini, fim)
    return int(round(bruto / 5) * 5)  # prestador cobra valor "redondo"


def gerar() -> None:
    rng = random.Random(SEMENTE)

    with open(PASTA / "descricoes_treino.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # "grupo" = de qual modelo de frase veio; a validação cruzada usa isso para não testar
        # o modelo com variações da mesma frase que ele viu no treino (GroupKFold)
        w.writerow(["descricao", "categoria", "complexidade", "grupo"])
        for categoria, problemas in PROBLEMAS.items():
            for indice, (problema, complexidade) in enumerate(problemas):
                for _ in range(6):
                    w.writerow([frase(rng, problema), categoria, complexidade, f"{categoria}-{indice}"])

    with open(PASTA / "historico_servicos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "descricao", "categoria", "complexidade", "bairro", "valor_pago", "nota"])
        id_ = 1
        for categoria, problemas in PROBLEMAS.items():
            for problema, complexidade in problemas:
                for _ in range(rng.randint(3, 6)):
                    w.writerow([id_, frase(rng, problema), categoria, complexidade, rng.choice(BAIRROS),
                                valor(rng, categoria, complexidade), rng.choice([3, 4, 4, 5, 5, 5])])
                    id_ += 1


if __name__ == "__main__":
    gerar()
    print("Bases geradas em", PASTA)
