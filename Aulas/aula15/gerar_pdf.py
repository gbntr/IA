from fpdf import FPDF

IMG = "/home/gbntr/.claude/image-cache/d7237b7c-4699-45dc-aef0-12a655b98259/1.png"
OUT = "guilherme_ramos_aula15Pratico.pdf"

pdf = FPDF()
pdf.set_margins(20, 20, 20)
pdf.add_page()

# Título
pdf.set_font("Helvetica", "B", 14)
pdf.cell(0, 10, "Aula 15 - Aprendizado Supervisionado (Parte 2)", ln=True, align="C")
pdf.set_font("Helvetica", "", 11)
pdf.cell(0, 7, "Guilherme Ramos", ln=True, align="C")
pdf.ln(4)

# Dataset
pdf.set_font("Helvetica", "B", 12)
pdf.cell(0, 8, "Dataset", ln=True)
pdf.set_font("Helvetica", "", 11)
pdf.multi_cell(0, 6,
    "Wine Dataset (UCI - https://archive.ics.uci.edu/dataset/109/wine)\n"
    "178 amostras, 13 atributos quimicos, 3 classes de vinho.\n"
    "Divisao: 70% treino (124 amostras) e 30% teste (54 amostras), aleatoria com random_state=42.\n"
    "Pre-processamento: normalizacao com StandardScaler (media 0, desvio padrao 1), "
    "necessaria para convergencia adequada do SVM, Perceptron e MLP."
)
pdf.ln(3)

# SVM
pdf.set_font("Helvetica", "B", 12)
pdf.cell(0, 8, "1. SVM - Support Vector Machine", ln=True)
pdf.set_font("Helvetica", "", 11)
pdf.multi_cell(0, 6,
    "Parametros utilizados:\n"
    "  - kernel='rbf': kernel de funcao de base radial, mapeia os dados para um espaco de "
    "maior dimensao permitindo separacao nao-linear entre as classes.\n"
    "  - C=1.0: parametro de regularizacao. Controla o trade-off entre maximizar a margem e "
    "minimizar erros de classificacao. Valor maior = menos regularizacao.\n"
    "  - gamma='scale': define o raio de influencia de cada amostra de suporte. Com 'scale', "
    "gamma = 1 / (n_features * X.var()), se ajusta automaticamente ao dataset.\n"
    "  - random_state=42: semente para reproducibilidade.\n\n"
    "Resultado: 53/54 acertos | Acuracia: 98.15%"
)
pdf.ln(3)

# Perceptron
pdf.set_font("Helvetica", "B", 12)
pdf.cell(0, 8, "2. Perceptron", ln=True)
pdf.set_font("Helvetica", "", 11)
pdf.multi_cell(0, 6,
    "Parametros utilizados:\n"
    "  - max_iter=1000: numero maximo de passagens pelo dataset de treino (epocas). "
    "Garante que o algoritmo tenha iteracoes suficientes para convergir.\n"
    "  - eta0=0.1: taxa de aprendizado. Controla o tamanho do passo na atualizacao dos pesos. "
    "Valor moderado evita oscilacao e divergencia.\n"
    "  - tol=1e-3: criterio de parada. O treinamento encerra quando a reducao de loss entre "
    "epocas consecutivas for menor que este valor.\n"
    "  - random_state=42: semente para reproducibilidade.\n\n"
    "Resultado: 53/54 acertos | Acuracia: 98.15%"
)
pdf.ln(3)

# MLP
pdf.set_font("Helvetica", "B", 12)
pdf.cell(0, 8, "3. MLP - Multi-Layer Perceptron", ln=True)
pdf.set_font("Helvetica", "", 11)
pdf.multi_cell(0, 6,
    "Parametros utilizados:\n"
    "  - hidden_layer_sizes=(100, 50): duas camadas ocultas com 100 e 50 neuronios "
    "respectivamente. Permite aprender representacoes hierarquicas dos dados.\n"
    "  - activation='relu': funcao de ativacao ReLU (Rectified Linear Unit) nas camadas ocultas. "
    "Eficiente e evita o problema de gradiente desvanecente.\n"
    "  - solver='adam': otimizador Adam, adaptativo por parametro. Eficiente para datasets "
    "de tamanho pequeno/medio e converge rapido.\n"
    "  - alpha=0.0001: forca da regularizacao L2 (weight decay), penaliza pesos grandes "
    "para reduzir overfitting.\n"
    "  - learning_rate_init=0.001: taxa de aprendizado inicial do Adam.\n"
    "  - max_iter=500: numero maximo de epocas de treinamento.\n"
    "  - random_state=42: semente para reproducibilidade.\n\n"
    "Resultado: 54/54 acertos | Acuracia: 100.00%"
)
pdf.ln(4)

# Print da saída
pdf.set_font("Helvetica", "B", 12)
pdf.cell(0, 8, "Print da saida do programa", ln=True)
pdf.ln(2)
pdf.image(IMG, x=20, w=170)

pdf.output(OUT)
print(f"PDF gerado: {OUT}")
