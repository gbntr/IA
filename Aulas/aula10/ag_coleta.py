import random
import copy

# Parâmetros do problema
NUM_BAIRROS = 12
CAPACIDADE_CAMINHAO = 8
VOLUME_POR_BAIRRO = [2, 1, 3, 2, 1, 3, 1, 2, 3, 1, 2, 1]

# Matriz de distâncias
MATRIZ_DISTANCIAS = [
    [0, 5, 9, 14, 7, 6, 12, 11, 8, 10, 13, 15],
    [5, 0, 4, 12, 6, 5, 11, 13, 9, 8, 14, 10],
    [9, 4, 0, 6, 10, 8, 12, 9, 7, 11, 13, 14],
    [14, 12, 6, 0, 8, 7, 9, 10, 12, 13, 5, 6],
    [7, 6, 10, 8, 0, 5, 8, 11, 10, 9, 12, 13],
    [6, 5, 8, 7, 5, 0, 6, 9, 8, 10, 11, 14],
    [12, 11, 12, 9, 8, 6, 0, 4, 7, 8, 10, 9],
    [11, 13, 9, 10, 11, 9, 4, 0, 3, 6, 7, 8],
    [8, 9, 7, 12, 10, 8, 7, 3, 0, 5, 9, 10],
    [10, 8, 11, 13, 9, 10, 8, 6, 5, 0, 4, 7],
    [13, 14, 13, 5, 12, 11, 10, 7, 9, 4, 0, 3],
    [15, 10, 14, 6, 13, 14, 9, 8, 10, 7, 3, 0]
]

# Parâmetros do Algoritmo Genético
TAMANHO_POPULACAO = 100
GERACOES = 500
TAXA_MUTACAO = 0.1
TAXA_CROSSOVER = 0.8
NUM_SEPARADORES = 4  # Permite até 5 viagens (pois precisamos de no mínimo 3)

# Representação:
# Bairros são de 0 a 11. Separadores são representados pelos números 12, 13, 14, 15.
# O cromossomo é uma permutação dos números de 0 a 15.

def criar_cromossomo():
    cromossomo = list(range(NUM_BAIRROS + NUM_SEPARADORES))
    random.shuffle(cromossomo)
    return cromossomo

def criar_populacao(tamanho):
    return [criar_cromossomo() for _ in range(tamanho)]

def calcular_aptidao(cromossomo):
    viagens = []
    viagem_atual = []
    
    # Decodificar cromossomo em viagens
    for gene in cromossomo:
        if gene < NUM_BAIRROS:
            viagem_atual.append(gene)
        else:
            if viagem_atual:
                viagens.append(viagem_atual)
                viagem_atual = []
    if viagem_atual:
        viagens.append(viagem_atual)
        
    distancia_total = 0
    penalidade_capacidade = 0
    
    for viagem in viagens:
        # Calcular volume da viagem e checar capacidade
        volume_viagem = sum(VOLUME_POR_BAIRRO[b] for b in viagem)
        if volume_viagem > CAPACIDADE_CAMINHAO:
            penalidade_capacidade += (volume_viagem - CAPACIDADE_CAMINHAO) * 50 # Peso da penalidade
            
        # Calcular distância percorrida na viagem
        if len(viagem) > 1:
            for i in range(len(viagem) - 1):
                distancia_total += MATRIZ_DISTANCIAS[viagem[i]][viagem[i+1]]
                
    # Penalizar número excessivo de viagens (mínimo necessário é 3)
    penalidade_viagens = max(0, len(viagens) - 3) * 20
    
    # Queremos minimizar o custo, então a aptidão é o inverso do custo total (ou usar o valor negativo)
    custo_total = distancia_total + penalidade_capacidade + penalidade_viagens
    
    # Adicionar um pequeno valor para evitar divisão por zero
    return 1.0 / (custo_total + 1e-6), custo_total, distancia_total, viagens

def selecao_torneio(populacao, aptidoes, k=3):
    selecionados = []
    for _ in range(2):
        participantes = random.sample(list(zip(populacao, aptidoes)), k)
        vencedor = max(participantes, key=lambda x: x[1])[0]
        selecionados.append(vencedor)
    return selecionados[0], selecionados[1]

def crossover_pmx(pai1, pai2):
    # Partially Mapped Crossover (PMX) para manter a permutação válida
    tamanho = len(pai1)
    ponto1, ponto2 = sorted(random.sample(range(tamanho), 2))
    
    filho1 = [-1] * tamanho
    filho2 = [-1] * tamanho
    
    # Copiar o segmento entre os pontos
    filho1[ponto1:ponto2] = pai1[ponto1:ponto2]
    filho2[ponto1:ponto2] = pai2[ponto1:ponto2]
    
    # Função auxiliar para mapeamento
    def preencher_filho(filho, pai, segmento_pai, segmento_outro_pai):
        for i in range(ponto1, ponto2):
            elemento = segmento_outro_pai[i - ponto1]
            if elemento not in filho:
                pos = pai.index(elemento)
                while ponto1 <= pos < ponto2:
                    pos = pai.index(segmento_pai[pos - ponto1])
                filho[pos] = elemento
        
        for i in range(tamanho):
            if filho[i] == -1:
                filho[i] = pai[i]
        return filho

    filho1 = preencher_filho(filho1, pai2, pai1[ponto1:ponto2], pai2[ponto1:ponto2])
    filho2 = preencher_filho(filho2, pai1, pai2[ponto1:ponto2], pai1[ponto1:ponto2])
    
    return filho1, filho2

def mutacao_swap(cromossomo):
    # Troca de posição de dois genes
    if random.random() < TAXA_MUTACAO:
        i, j = random.sample(range(len(cromossomo)), 2)
        cromossomo[i], cromossomo[j] = cromossomo[j], cromossomo[i]
    return cromossomo

def executar_ag():
    populacao = criar_populacao(TAMANHO_POPULACAO)
    melhor_solucao = None
    melhor_aptidao = -1
    melhor_custo = float('inf')
    melhor_viagens = []
    
    for geracao in range(GERACOES):
        resultados_aptidao = [calcular_aptidao(ind) for ind in populacao]
        aptidoes = [r[0] for r in resultados_aptidao]
        
        # Encontrar melhor da geração
        for i, (aptidao, custo, dist, viagens) in enumerate(resultados_aptidao):
            if aptidao > melhor_aptidao:
                melhor_aptidao = aptidao
                melhor_solucao = populacao[i]
                melhor_custo = custo
                melhor_viagens = viagens
                
        nova_populacao = []
        
        # Elitismo: manter o melhor
        nova_populacao.append(melhor_solucao)
        
        while len(nova_populacao) < TAMANHO_POPULACAO:
            pai1, pai2 = selecao_torneio(populacao, aptidoes)
            
            if random.random() < TAXA_CROSSOVER:
                filho1, filho2 = crossover_pmx(pai1, pai2)
            else:
                filho1, filho2 = copy.deepcopy(pai1), copy.deepcopy(pai2)
                
            filho1 = mutacao_swap(filho1)
            filho2 = mutacao_swap(filho2)
            
            nova_populacao.append(filho1)
            if len(nova_populacao) < TAMANHO_POPULACAO:
                nova_populacao.append(filho2)
                
        populacao = nova_populacao
        
        if (geracao + 1) % 100 == 0:
            print(f"Geração {geracao + 1} - Menor custo encontrado: {melhor_custo}")
            
    return melhor_solucao, melhor_custo, melhor_viagens

if __name__ == "__main__":
    print("Iniciando Algoritmo Genético para Coleta de Resíduos...")
    melhor_cromossomo, custo_final, viagens_finais = executar_ag()
    
    print("\n--- Resultado Final ---")
    print(f"Cromossomo: {melhor_cromossomo}")
    print(f"Custo Total (Distância + Penalidades): {custo_final}")
    print(f"Número de Viagens: {len(viagens_finais)}")
    
    distancia_real = 0
    for i, viagem in enumerate(viagens_finais):
        vol = sum(VOLUME_POR_BAIRRO[b] for b in viagem)
        dist = 0
        if len(viagem) > 1:
            for j in range(len(viagem) - 1):
                dist += MATRIZ_DISTANCIAS[viagem[j]][viagem[j+1]]
        distancia_real += dist
        nomes_bairros = [f"B{b}" for b in viagem]
        print(f"  Viagem {i+1}: Rota {nomes_bairros} | Volume: {vol} m³ | Distância: {dist} km")
        
    print(f"Distância Percorrida Real: {distancia_real} km")
