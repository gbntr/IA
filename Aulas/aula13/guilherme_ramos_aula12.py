import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo

# Configurações de visualização
sns.set_theme(style="whitegrid")

def main():
    print("Iniciando exercício de pré-processamento de dados...")
    print("Nome: Guilherme Ramos\n")

    # 1) Carregar o dataset Wine
    # Fetch dataset 
    wine = fetch_ucirepo(id=109) 
    
    # Data (as pandas dataframes) 
    X = wine.data.features 
    y = wine.data.targets 
    
    # Concatenar para algumas análises
    df = pd.concat([X, y], axis=1)
    
    # a) Boxplot analisando média, desvio padrão e outliers (antes da normalização)
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=X)
    plt.xticks(rotation=45)
    plt.title("Boxplot das Features (Original)")
    plt.tight_layout()
    plt.savefig("boxplot_original.png")
    print("a) Boxplot original gerado (boxplot_original.png).")
    
    # Análise rápida de estatísticas
    stats = X.describe()
    print("\nEstatísticas das Features (Original):")
    print(stats.loc[['mean', 'std']])
    print("\nObservação sobre Outliers: Features como 'Proline' e 'Magnesium' possuem escalas muito diferentes, "
          "o que dificulta a visualização de outliers em outras variáveis sem normalização.")

    # b) Aplicar normalização (StandardScaler) e gerar novo boxplot
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=X_scaled)
    plt.xticks(rotation=45)
    plt.title("Boxplot das Features (Normalizado - StandardScaler)")
    plt.tight_layout()
    plt.savefig("boxplot_normalizado.png")
    print("\nb) Normalização aplicada e novo boxplot gerado (boxplot_normalizado.png).")

    # c) Plotar distribuição de elementos em cada classe
    plt.figure(figsize=(8, 6))
    sns.countplot(x='class', data=y)
    plt.title("Distribuição de Elementos por Classe")
    plt.tight_layout()
    plt.savefig("distribuicao_classes.png")
    print("c) Distribuição de classes gerada (distribuicao_classes.png).")

    # d) Matriz de correlação e identificação de maiores correlações
    plt.figure(figsize=(12, 10))
    corr_matrix = X.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de Correlação")
    plt.tight_layout()
    plt.savefig("matriz_correlacao.png")
    print("d) Matriz de correlação gerada (matriz_correlacao.png).")
    
    # Identificar as maiores correlações (excluindo a diagonal)
    corr_unstacked = corr_matrix.abs().unstack()
    corr_sorted = corr_unstacked.sort_values(ascending=False)
    # Pegar as correlações mais altas que não sejam 1.0 (auto-correlação)
    top_corr = corr_sorted[corr_sorted < 1.0].head(10)
    print("\nMaiores correlações encontradas:")
    print(top_corr)

    # e) Gráfico 2D aplicando PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
    pca_df['class'] = y.values
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='class', palette='viridis', data=pca_df, s=100)
    plt.title("Visualização PCA 2D")
    plt.tight_layout()
    plt.savefig("pca_2d.png")
    print("\ne) Gráfico PCA 2D gerado (pca_2d.png).")
    
    print("\nExercício concluído com sucesso!")

if __name__ == "__main__":
    main()
