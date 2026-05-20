from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def main():
    print("Aula 14 - Aprendizado Supervisionado")
    print("Nome: Guilherme Ramos\n")

    wine = fetch_ucirepo(id=109)
    X = wine.data.features
    y = wine.data.targets.values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "KNN (k=5)":       KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes":     GaussianNB(),
        "Decision Tree":   DecisionTreeClassifier(random_state=42),
    }

    print(f"Amostras de treino: {len(X_train)} | Amostras de teste: {len(X_test)}\n")

    for name, model in models.items():
        # KNN é sensível à escala, os demais também se beneficiam da normalização
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acertos = (y_pred == y_test).sum()
        acuracia = accuracy_score(y_test, y_pred)
        print(f"{name}:")
        print(f"  Acertos: {acertos}/{len(y_test)}")
        print(f"  Acurácia: {acuracia * 100:.2f}%\n")

if __name__ == "__main__":
    main()
