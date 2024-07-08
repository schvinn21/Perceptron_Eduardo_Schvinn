import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plotar(grupo1, grupo2, entradas, bias, pesos):
    plt.scatter(grupo1[:, 0], grupo1[:, 1], color='red', label='Grupo 1')
    plt.scatter(grupo2[:, 0], grupo2[:, 1], color='blue', label='Grupo 2')

    x_min, x_max = min(entradas[:, 0]), max(entradas[:, 0])
    y_min, y_max = min(entradas[:, 1]), max(entradas[:, 1])

    valores_x = np.linspace(x_min, x_max, 200)
    valores_y = -(pesos[0] * valores_x + bias) / pesos[1]

    plt.plot(valores_x, valores_y, label='Linha de Decisão', color='green')

    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.legend()
    plt.grid(True)
    plt.show()

def gera_grupo1(mean, cov, size=50, seed=None):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, cov, size=size)

def gera_grupo2(mean, cov, size=100, seed=None):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, cov, size=size)

def altera_pesos(saida_percep):
    global pesos
    for i in range(len(entradas)):
        if saida_percep[i] != rotulos[i]:
            pesos +=  taxa_de_aprendizado * (rotulos[i] - saida_percep[i])* entradas[i]

def funcao_de_ativacao(soma_ponderada):
    if soma_ponderada > bias :
        return 1
    else :
        return 0

def perceptron(x_entradas, w_pesos):
    saida_percep = []
    for x in x_entradas:
        produto_x_pesos = np.dot(x, w_pesos)
        print("produto dos pesos e entradas:", produto_x_pesos)
        saida_percep.append(funcao_de_ativacao(produto_x_pesos))
    return np.array(saida_percep)


mean1 = [2, 3] ; cov1 = [[1, 0.5], [0.5, 1]]

mean2 = [4, 2] ; cov2 =  [[1, 0.5], [0.5, 1]]

seed = 10 

grupo1 = gera_grupo1(mean1, cov1, seed=seed)
grupo2 = gera_grupo2(mean2, cov2, seed=seed)

entradas = np.vstack((grupo1, grupo2))

rotulos = np.hstack((np.zeros(len(grupo1)), np.ones(len(grupo2))))

pesos = np.zeros(entradas.shape[1])
#print(pesos)
bias = 1
taxa_de_aprendizado = 0.01

epocas = 500
for _ in range(epocas):
    saida = perceptron(entradas, pesos)
    altera_pesos(saida)
    print("pesos:", pesos)
    
pesos_treinados = pesos

#print("saida final:", saida)
plotar = plotar(grupo1, grupo2, entradas, bias, pesos)

'''

Geração dos dados para teste.

'''



#dados de teste tilizando uma outra semente.

mean1_teste = [2, 3] ; cov1_teste = [[1, 0.5], [0.5, 1]]

mean2_teste = [4, 2] ; cov2_teste = [[1, 0.5], [0.5, 1]]

seed_teste = 40


def plotar_teste(grupo1_teste, grupo2_teste, entradas_teste , bias , pesos_treinados):
    plt.scatter(grupo1_teste[:, 0], grupo1_teste[:, 1], color='red', label='Grupo 1')
    plt.scatter(grupo2_teste[:, 0], grupo2_teste[:, 1], color='blue', label='Grupo 2')

    x_min_teste, x_max_teste = min(entradas_teste[:, 0]), max(entradas_teste[:, 0])
    y_min_teste, y_max_teste = min(entradas_teste[:, 1]), max(entradas_teste[:, 1])

    valores_x = np.linspace(x_min_teste, x_max_teste, 200)
    valores_y = -(pesos_treinados[0] * valores_x + bias) / pesos_treinados[1]

    plt.plot(valores_x, valores_y, label='Linha de Decisão', color='green')

    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.legend()
    plt.grid(True)
    plt.show()

def gera_grupo1_teste(mean, cov, size=50, seed=None):
    rng = np.random.default_rng(seed_teste)
    return rng.multivariate_normal(mean, cov, size=size)

def gera_grupo2_teste(mean, cov, size=100, seed=None):
    rng = np.random.default_rng(seed_teste)
    return rng.multivariate_normal(mean, cov, size=size)

def aplicar_rede(pesos_treinados,t_entradas):
    saida_teste= []
    for t in t_entradas:
        produto_teste = np.dot(t,pesos_treinados)
        print("produto dos pesos e entradas do teste:", produto_teste)
        saida_teste.append(funcao_de_ativacao(produto_teste))
    return np.array(saida_teste)
    


grupo1_teste = gera_grupo1_teste(mean1_teste, cov1_teste, seed=seed_teste)
grupo2_teste = gera_grupo2_teste(mean2_teste, cov2_teste, seed=seed_teste)

entradas_teste = np.vstack((grupo1_teste, grupo2_teste))

rotulos_teste = np.hstack((np.zeros(len(grupo1_teste)), np.ones(len(grupo2_teste))))

plotar_teste = plotar_teste(grupo1_teste, grupo2_teste, entradas_teste , bias , pesos_treinados)

teste = aplicar_rede(pesos_treinados,entradas_teste)
print("saida do teste:" , teste)
print("rotulo do teste : " , rotulos_teste)

#matriz de confusão do teste

y_verdeiros = rotulos_teste
y_previsoes = teste
cm = confusion_matrix(y_verdeiros, y_previsoes)

tn, fp, fn, tp = cm.ravel()
precisao = tp/(tp + fp)
print("a precisão é:", precisao*100)

acuracia = (tp + tn)/(tp + tn + fp + fn)
print("a acuracia é:", acuracia*100)

# Função para plotar a matriz de confusão
def plot_confusion_matrix(cm, classes, title='Matriz de Confusão', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Rótulos Previstos')
    plt.ylabel('Rótulos Verdadeiros')
    plt.tight_layout()

# Definindo os nomes das classes
classes = ['Classe 0', 'Classe 1']  # Substitua com os nomes das suas classes

# Plotando a matriz de confusão
plot_confusion_matrix(cm, classes)
plt.show()