import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generowanie przykładowych danych
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Dzielimy dane na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tworzenie obiektu regresji liniowej
model = LinearRegression()

# Trenowanie modelu
model.fit(X_train, y_train)

# Predykcja na zestawie testowym
y_pred = model.predict(X_test)

# Wizualizacja wyników
plt.scatter(X_test, y_test, color='blue', label='Prawdziwe wartości')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Przewidywana linia')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regresja liniowa')
plt.legend()
plt.show()

# Obliczanie i wyświetlanie błędu średniego kwadratowego oraz R^2
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Średni błąd kwadratowy (MSE): {mse:.2f}')
print(f'Wskaźnik R^2: {r2:.2f}')
