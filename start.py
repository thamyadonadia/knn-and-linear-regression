import pandas as pd
import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('diabetes.csv')
mae_knn = []
mae_linear_regression = []

for i in range(30): 
    shuffle_data = data.sample(frac=1, random_state=i)

    # Separar os dados em treino (70%), validação (10%) e teste (20%)
    train_data, remaining_data = train_test_split(shuffle_data, train_size=0.7, random_state=42)
    valid_data, test_data = train_test_split(remaining_data, test_size=0.67, random_state=42)

    # Separar as features (X) e os rótulos (y) dos dados de treino
    x_train = train_data.drop('Glucose', axis=1) 
    y_train = train_data['Glucose'] 

    # Separar as features (X) e os rótulos (y) dos dados de validação
    x_valid = valid_data.drop('Glucose', axis=1) 
    y_valid = valid_data['Glucose']  
    
    # Separar as features (X) e os rótulos (y) dos dados de teste
    x_test = test_data.drop('Glucose', axis=1)
    y_test = test_data['Glucose']

    # Para escolher o k com menor MAE
    lower_mae_knn = 100
    best_k = 0

    for k in range(1, 11):
        # Criar o regressor KNN com um número de vizinhos ajustável (k)
        knn = KNeighborsRegressor(n_neighbors=k)

        # Treinar o regressor KNN
        knn.fit(x_train, y_train)

        # Fazer previsões nos dados de validação
        y_pred = knn.predict(x_valid)

        # Calcular o erro quadrático médio (MAE) das previsões
        mae = mean_absolute_error(y_valid, y_pred)  

        if mae < lower_mae_knn: 
            lower_mae_knn = mae
            best_k = k
        

    # Aplicando KNN no conjunto de teste
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    mae_knn.append(mean_absolute_error(y_test, y_pred))

    #TODO: precisa verificar qual parâmetro será ajustado para validar on conjunto de validação
    
    # Criando uma instância de polynomail_features
    polynomial_features = PolynomialFeatures(degree=2)
    x_poly_train = polynomial_features.fit_transform(x_train)
    x_poly_test = polynomial_features.fit_transform(x_test)
    
    # Aplicando Regressão Linear no conjunto de teste
    linear_regression = LinearRegression()
    linear_regression.fit(x_poly_train, y_train) # Treinar o regressor de regressão linear
    y_pred = linear_regression.predict(x_poly_test) # Fazer previsões nos dados de teste
    mae_linear_regression.append(mean_absolute_error(y_test, y_pred))  # Adicionar o MAE à lista

# Cálculo da média (estimativa pontual) e o desvio padrão do MAE
mae_knn_mean = np.mean(mae_knn)
mae_knn_std = np.std(mae_knn) # mae_std = np.std(mae_knn, ddof=1) -  Usamos ddof=1 para calcular o desvio padrão amostral

# Calcular o intervalo de confiança com nível de confiança de 95%
alpha = 0.05  # Nível de confiança de 95%
df = 29  # Graus de liberdade

t_critical = stats.t.ppf(1 - alpha / 2, df)  # Valor crítico da distribuição t de Student
lower_bound, upper_bound = stats.t.interval(1 - alpha, df, loc=mae_knn_mean, scale=mae_knn_std / np.sqrt(30))

print("Intervalo de confiança KNN (95%):", (lower_bound, upper_bound))
print("Estimativa pontual KNN: ", mae_knn_mean)

# Cálculo da média (estimativa pontual) e o desvio padrão do MAE
mae_linear_regression_mean = np.mean(mae_linear_regression)
mae_linear_regression_std = np.std(mae_linear_regression) # mae_std = np.std(mae_knn, ddof=1) -  Usamos ddof=1 para calcular o desvio padrão amostral

t_critical = stats.t.ppf(1 - alpha / 2, df)  # Valor crítico da distribuição t de Student
lower_bound, upper_bound = stats.t.interval(1 - alpha, df, loc=mae_linear_regression_mean, scale=mae_linear_regression_std / np.sqrt(30))

print("\nIntervalo de confiança Regressão Linear (95%):", (lower_bound, upper_bound))
print("Estimativa pontual Regressão Linear: ", mae_linear_regression_mean)