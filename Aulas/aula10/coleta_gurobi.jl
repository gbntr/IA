using JuMP
using Gurobi

# --- Dados do Problema ---
const NUM_BAIRROS = 12
const CAPACIDADE_CAMINHAO = 8
const VOLUME_POR_BAIRRO = [2, 1, 3, 2, 1, 3, 1, 2, 3, 1, 2, 1]

# Matriz de distâncias (Convertida para sintaxe Julia)
const MATRIZ_DISTANCIAS = [
    0  5  9 14  7  6 12 11  8 10 13 15;
    5  0  4 12  6  5 11 13  9  8 14 10;
    9  4  0  6 10  8 12  9  7 11 13 14;
   14 12  6  0  8  7  9 10 12 13  5  6;
    7  6 10  8  0  5  8 11 10  9 12 13;
    6  5  8  7  5  0  6  9  8 10 11 14;
   12 11 12  9  8  6  0  4  7  8 10  9;
   11 13  9 10 11  9  4  0  3  6  7  8;
    8  9  7 12 10  8  7  3  0  5  9 10;
   10  8 11 13  9 10  8  6  5  0  4  7;
   13 14 13  5 12 11 10  7  9  4  0  3;
   15 10 14  6 13 14  9  8 10  7  3  0
]

# --- Inicialização do Modelo ---
# O Gurobi encontrará a solução ótima matemática
model = Model(Gurobi.Optimizer)

# --- Variáveis de Decisão ---
# x[i, j] = 1 se o caminhão viaja do bairro i para o bairro j
@variable(model, x[1:NUM_BAIRROS, 1:NUM_BAIRROS], Bin)

# u[i] = Variável auxiliar para carga acumulada (Eliminação de Sub-rotas MTZ)
# Assumimos o Bairro 1 como o Depósito (ponto inicial/final de cada viagem)
@variable(model, VOLUME_POR_BAIRRO[i] <= u[i=2:NUM_BAIRROS] <= CAPACIDADE_CAMINHAO)

# --- Função Objetivo ---
# Minimizar a distância total percorrida
@objective(model, Min, sum(MATRIZ_DISTANCIAS[i, j] * x[i, j] for i in 1:NUM_BAIRROS, j in 1:NUM_BAIRROS))

# --- Restrições ---

# 1. Cada bairro (exceto o depósito) deve ser visitado exatamente uma vez
for j in 2:NUM_BAIRROS
    @constraint(model, sum(x[i, j] for i in 1:NUM_BAIRROS if i != j) == 1)
end

# 2. Cada bairro (exceto o depósito) deve ter uma saída exatamente uma vez
for i in 2:NUM_BAIRROS
    @constraint(model, sum(x[i, j] for j in 1:NUM_BAIRROS if i != j) == 1)
end

# 3. Equilíbrio de fluxo no depósito (O número de saídas deve ser igual ao de entradas)
@constraint(model, sum(x[1, j] for j in 2:NUM_BAIRROS) == sum(x[i, 1] for i in 2:NUM_BAIRROS))

# 4. Restrição de Miller-Tucker-Zemlin (MTZ) - Controla a capacidade e evita sub-ciclos
for i in 2:NUM_BAIRROS
    for j in 2:NUM_BAIRROS
        if i != j
            @constraint(model, u[i] + VOLUME_POR_BAIRRO[j] <= u[j] + CAPACIDADE_CAMINHAO * (1 - x[i, j]))
        end
    end
end

# --- Execução do Solver ---
optimize!(model)

# --- Exibição dos Resultados ---
if termination_status(model) == MOI.OPTIMAL
    println("\n=== Resultado Otimizado (Gurobi) ===")
    println("Distância Total Mínima: ", objective_value(model), " km")
    
    # Reconstrução das rotas
    for j in 2:NUM_BAIRROS
        if value(x[1, j]) > 0.5
            print("\nRota: Depósito (B1)")
            atual = j
            while true
                print(" -> B$atual")
                proximo = 0
                for k in 1:NUM_BAIRROS
                    if k != atual && value(x[atual, k]) > 0.5
                        proximo = k
                        break
                    end
                end
                if proximo == 1
                    println(" -> Depósito (B1)")
                    break
                else
                    atual = proximo
                end
            end
        end
    end
else
    println("Não foi possível encontrar uma solução ótima.")
end
