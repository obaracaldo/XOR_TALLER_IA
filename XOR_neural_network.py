# Se importan las librerias necesarias
import numpy as np

# Se define la funcion de activacion que se utilizara, en este caso la funcion sigmoide
def sigmoid(x):
    return 1 / (1+np.exp(-x))

# Datos de entrenamiento para la red
X = np.array([[0, 0, 1, 1, 1], [0, 1, 0, 1, 1]]) #dos arreglos binarios
Y = np.array([[0, 1, 1, 0, 0]]) #xor de los dos arreglos

# Se definen los parametros de la red neuronal
Neurons_H_Layers = 2 # numero de neuronas en la capa oculta
input = X.shape[0] # cantidad de datos de entrada
output = Y.shape[0] # 1 cantidad de datos de salida


#Se inicializan los pesos y los bias de forma aleatoria
#matriz 2x2 con entradas aleatorias
W1 = np.random.randn(Neurons_H_Layers,input) #cada fila son los pesos de un neurona
#matriz 1x2 con entradas aleatorias
W2 = np.random.randn(output,Neurons_H_Layers) #cada posicion tiene los pesos de las salidas de la capa anterior
#matriz 2x1 con entradas aleatorias
b1 = np.random.randn(Neurons_H_Layers, 1) #cada entrada sera el bias de cada nurona en la capa oculta
#bias de la capa de salida
b2 = np.random.randn(output, 1)

epoch = 100000
learningRate = 0.01

for i in range(epoch): #ciclo de entrenamiento (100000 iteraciones)

    
    #------------------------------------------------------------------------------#
    #                             Forward propagation                              #
    #------------------------------------------------------------------------------#
    # Se realiza el forward propagation
    Z1 = np.dot(W1, X) + b1 #se multiplica cada entrada con sus respectivos pesos en cada neurona y se suma su respectivo bias
    OUT_LAYER1 = sigmoid(Z1) #se aplica la funcion de activacion a ambas salidas

    Z2 = np.dot(W2,OUT_LAYER1) + b2 #se multiplican las dos salidas con sus respectivos pesos y se suma un bias
    OUT_LAYER2 = sigmoid(Z2) #se aplica la funcion de activacion

    #------------------------------------------------------------------------------#
    #                             back propagation                                 #
    #------------------------------------------------------------------------------#
    # Se realiza back propagation teniendo como funcion de error el error cuadratico medio
    # A continuacion se calcula el gradiente, las derivadas segun los pesos, las salidas y los bias para cada capa 
    total_datos = X.shape[1]
    d_Z2 = OUT_LAYER2 - Y
    d_W2 = np.dot(d_Z2, OUT_LAYER1.T) / total_datos
    d_b2 = np.sum(d_Z2, axis=1, keepdims=True)

    d_A1 = np.dot(W2.T, d_Z2)
    d_Z1 = np.multiply(d_A1, OUT_LAYER1 * (1 - OUT_LAYER1))
    d_W1 = np.dot(d_Z1, X.T) / total_datos
    d_b1 = np.sum(d_Z1, axis=1, keepdims=True) / total_datos


    #------------------------------------------------------------------------------#
    #                             Actualizacion parametros                         #
    #------------------------------------------------------------------------------#
    #se actualizan los pesos y los bias segun los gradientes encontrados y la tasa de aprendizaje
    W1 = W1 - learningRate * d_W1
    W2 = W2 - learningRate * d_W2
    b1 = b1 - learningRate * d_b1
    b2 = b2 - learningRate * d_b2


#------------------------------------------------------------------------------#
#                                  Prediccion                                  #
#------------------------------------------------------------------------------#

#Prueba de la red neuronal
X = np.array([[1, 0, 0, 0, 1], [0, 1, 0, 1, 1]]) #dato de entrada, dos listas binarias

#se realiza el forward propagation una vez con los parametros ya entrenados
Z1 = np.dot(W1, X) + b1 
OUT_LAYER1 = sigmoid(Z1)

Z2 = np.dot(W2,OUT_LAYER1) + b2
OUT_LAYER2 = sigmoid(Z2)

#la salida de la red sera la prediccion de la xor de las dos listas binarias
prediction = (OUT_LAYER2 > 0.5) * 1.0 #se aproxima a 1 si el resultado predicho es suficientemente grande
print(f'XOR entre {X[0]} y {X[1]}')
print(F'LA PREDICCION ES {prediction}')
