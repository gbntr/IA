from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Carrega o dataset Wine (UCI dataset/109/wine)
data = load_wine()
X, y = data.data, data.target

# Divisão aleatória: 70% treino, 30% teste
# random_state=42 garante reprodutibilidade
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Normalização: média 0 e desvio padrão 1
# Necessário para SVM, Perceptron e MLP convergirem bem
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- SVM ---
# kernel='rbf': kernel radial basis function (não-linear)
# C=1.0: parâmetro de regularização (penalidade por erro de classificação)
# gamma='scale': gamma = 1 / (n_features * X.var()), escala automaticamente
# random_state=42: reprodutibilidade
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)
svm_hits = int(svm_acc * len(y_test))

# --- Perceptron ---
# max_iter=1000: número máximo de épocas de treinamento
# eta0=0.1: taxa de aprendizado inicial
# tol=1e-3: critério de convergência (diferença mínima de loss entre épocas)
# random_state=42: reprodutibilidade
perceptron = Perceptron(max_iter=1000, eta0=0.1, tol=1e-3, random_state=42)
perceptron.fit(X_train, y_train)
perceptron_pred = perceptron.predict(X_test)
perceptron_acc = accuracy_score(y_test, perceptron_pred)
perceptron_hits = int(perceptron_acc * len(y_test))

# --- MLP (Multi-Layer Perceptron) ---
# hidden_layer_sizes=(100, 50): duas camadas ocultas com 100 e 50 neurônios
# activation='relu': função de ativação ReLU nas camadas ocultas
# solver='adam': otimizador Adam (adaptativo, eficiente para datasets pequenos)
# alpha=0.0001: regularização L2 para evitar overfitting
# learning_rate_init=0.001: taxa de aprendizado inicial do Adam
# max_iter=500: número máximo de épocas
# random_state=42: reprodutibilidade
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42
)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_acc = accuracy_score(y_test, mlp_pred)
mlp_hits = int(mlp_acc * len(y_test))

total = len(y_test)

print("=" * 50)
print("  Dataset: Wine (UCI) | 178 amostras, 13 atributos, 3 classes")
print(f"  Treino: {len(X_train)} amostras | Teste: {total} amostras")
print("=" * 50)
print(f"\n  SVM (RBF, C=1.0, gamma=scale)")
print(f"    Acertos: {svm_hits}/{total} | Acurácia: {svm_acc*100:.2f}%")
print(f"\n  Perceptron (eta0=0.1, max_iter=1000)")
print(f"    Acertos: {perceptron_hits}/{total} | Acurácia: {perceptron_acc*100:.2f}%")
print(f"\n  MLP (camadas=[100,50], relu, adam, alpha=0.0001)")
print(f"    Acertos: {mlp_hits}/{total} | Acurácia: {mlp_acc*100:.2f}%")
print("=" * 50)
