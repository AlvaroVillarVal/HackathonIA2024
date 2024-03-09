from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Path to the CSV file
import matplotlib.pyplot as plt

csv_file = 'RetoIA\datasetIA.csv'

# Read the CSV file using pandas
data = pd.read_csv(csv_file,sep=';')
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
columnas = ["Tasa aciertos","Recompensa media", "Puntuación maxima", "Tiempo de respuesta min", "Tiempo de respuesta maximo", "Distancia al jugador de los objetos.", "Ratio de aparición de los objetos en el juego.", "Ratio de recompensa de los objetos.", "Tamaño de los objetos."]
data["Recompensa media"]= (data["Recompensa minima"]+data["Recompensa maxima"])/2
data["Tasa aciertos"]=data["Aciertos"]/(data["Aciertos"]+data["Fallos"])
columns_to_drop = ["Numero de piezas", "Tiempo total de la prueba","Trial values", "Aciertos","Fallos","Recompensa minima", "Recompensa maxima"]  # Lista de columnas a eliminar
data = data.drop(columns=columns_to_drop)  # Eliminar las columnas del DataFrame
data = data.dropna()

scaler = MinMaxScaler()
# Assuming 'data' is your DataFrame
data_normalized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
tasa_aciertos=0
data_normalized1=data_normalized.copy()
data_normalized2=data_normalized.copy()
data_normalized3=data_normalized.copy()
data_normalized1["Tasa aciertosobjetivo"]=data_normalized1["Tasa aciertos"]
X1 = data_normalized1.drop(["Ratio de aparición de los objetos en el juego.","Distancia al jugador de los objetos.","Ratio de recompensa de los objetos.","Tamaño de los objetos."], axis=1)  # Use all columns except targets as features
y1 = data_normalized1[["Distancia al jugador de los objetos.","Ratio de aparición de los objetos en el juego.","Ratio de recompensa de los objetos.","Tamaño de los objetos."]]  # Use 'target1', 'target2', 'target3', 'target4' columns as labels

# Split the data into training set and test set
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.1, random_state=42)

# Create MLPRegressor object
mlp1 = MLPRegressor(hidden_layer_sizes=(300,300, 300), max_iter=1000)

# Train the model
mlp1.fit(X1_train, y1_train)

# Use the model to make predictions
predictions = mlp1.predict(X1_test)
print(predictions)
# Calculate Mean Absolute Error
mae = mean_absolute_error(y1_test, predictions)
print(f"Mean Absolute Error: {mae}")

# Calculate Mean Squared Error
mse = mean_squared_error(y1_test, predictions)
print(f"Mean Squared Error: {mse}")

# Calculate R-squared
r2 = r2_score(y1_test, predictions)
print(f"R-squared: {r2}")

X2 = data_normalized2.drop(["Tasa aciertos","Recompensa media","Puntuación maxima", "Tiempo de respuesta min", "Tiempo de respuesta maximo"], axis=1)  # Use all columns except targets as features
y2 = data_normalized2[[ "Puntuación maxima", "Tiempo de respuesta min", "Tiempo de respuesta maximo","Recompensa media"]]  # Use 'target1', 'target2', 'target3', 'target4' columns as labels

# Split the data into training set and test set
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.1, random_state=42)

# Create MLPRegressor object
mlp2 = MLPRegressor(hidden_layer_sizes=(30,30, 30), max_iter=1000)

# Train the model
mlp2.fit(X2_train, y2_train)

# Use the model to make predictions
predictions = mlp2.predict(X2_test)
print(predictions)
# Calculate Mean Absolute Error
mae = mean_absolute_error(y2_test, predictions)
print(f"Mean Absolute Error: {mae}")

# Calculate Mean Squared Error
mse = mean_squared_error(y2_test, predictions)
print(f"Mean Squared Error: {mse}")

# Calculate R-squared
r2 = r2_score(y2_test, predictions)
print(f"R-squared: {r2}")


X3 = data_normalized3.drop(["Tasa aciertos"], axis=1)  # Use all columns except targets as features
y3 = data_normalized3["Tasa aciertos"]  # Use 'target1', 'target2', 'target3', 'target4' columns as labels

# Split the data into training set and test set
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.1, random_state=42)

# Create MLPRegressor object
mlp3 = MLPRegressor(hidden_layer_sizes=(30,30, 30), max_iter=1000)

# Train the model
mlp3.fit(X3_train, y3_train)

# Use the model to make predictions
predictions = mlp3.predict(X3_test)
print(predictions)
# Calculate Mean Absolute Error
mae = mean_absolute_error(y3_test, predictions)
print(f"Mean Absolute Error: {mae}")

# Calculate Mean Squared Error
mse = mean_squared_error(y3_test, predictions)
print(f"Mean Squared Error: {mse}")

# Calculate R-squared
r2 = r2_score(y3_test, predictions)
print(f"R-squared: {r2}")

for i in range(0,100):
    single_row_normalized2 = single_row = X1.iloc[[9000+i]].copy()
    tasa_aciertos=0
    print("\n")
    objetivomin=0.65
    objetivomax=0.65
    tasa_aciertos_old=-1111111111111111111111
    while tasa_aciertos < 0.55 or tasa_aciertos > 0.75:
       

        # Create a single row DataFrame for prediction

        # Normalize the single row data
        # Get the first row as a DataFrame and create a copy
    
        
        # # Make prediction for the single row
        prediction = mlp1.predict(single_row_normalized2)
        dist = prediction[0][0]
        aparicion = prediction[0][1]
        recompensa = prediction[0][2]
        tamaño = prediction[0][3]
        listaprediccion = [dist,aparicion, recompensa, tamaño]
    
        prediction_df = pd.DataFrame([listaprediccion], columns=X2.columns)
        predicción = mlp2.predict(prediction_df)
        listaprediccion2 = [predicción[0][0],predicción[0][1],predicción[0][2]]
        listaprediccion3=listaprediccion2.copy()
        listaprediccion2+=listaprediccion
        listaprediccion2.append(predicción[0][3])
        single_row_normalized1 = pd.DataFrame([listaprediccion2], columns=X3.columns)
        tasa_aciertos= mlp3.predict(single_row_normalized1)
        listaprediccion3.append(predicción[0][3])
        listaprediccion3.append(tasa_aciertos[0])
        listaprediccion3.append(0.65)
        single_row_normalized2 = pd.DataFrame([listaprediccion3], columns=X1.columns)
        print(tasa_aciertos)
        if abs(tasa_aciertos-tasa_aciertos_old)<0.0001:
            break
        else:
            tasa_aciertos_old=tasa_aciertos
            



    
    

