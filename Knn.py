import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Lectura de documentos CSV de entrenamiento
signo_ok = pd.read_csv("Click_ok.csv")
signo_ok_2 = pd.read_csv("Click_ok2.csv")
signo_ok_3 = pd.read_csv("Click_ok3.csv")
signo_ok_4 = pd.read_csv("Click_ok4.csv")

# Longitud de las filas de las caracteristicas
dis1 = len(signo_ok)
dis2 = len(signo_ok_2)
dis3 = len(signo_ok_3)
dis4 = len(signo_ok_4)

# Se halla el CSV con menor longitud en las filas
dis_t = [dis1, dis2, dis3,dis4]
dis_min = min(dis_t)
index_min = dis_t.index(dis_min)

# Recorte de las caracteristicas segun el CSV con menor longitud en las filas
if index_min == 0:
    signo_ok_2 = signo_ok_2.drop(range(dis1, dis2), axis=0)
    signo_ok_3 = signo_ok_3.drop(range(dis1, dis3), axis=0)
    signo_ok_4 = signo_ok_4.drop(range(dis1, dis4), axis=0)
elif (index_min == 1):
    signo_ok     = signo_ok.drop(range(dis2, dis1), axis=0)
    signo_ok_3 = signo_ok_3.drop(range(dis2, dis3), axis=0)
    signo_ok_4 = signo_ok_4.drop(range(dis2, dis4), axis=0)
elif (index_min == 2):
    signo_ok     = signo_ok.drop(range(dis3, dis1), axis=0)
    signo_ok_2 = signo_ok_2.drop(range(dis3, dis2), axis=0)
    signo_ok_4 = signo_ok_4.drop(range(dis3, dis4), axis=0)
else:
    signo_ok     = signo_ok.drop(range(dis4, dis1), axis=0)
    signo_ok_2 = signo_ok_2.drop(range(dis4, dis2), axis=0)
    signo_ok_3 = signo_ok_3.drop(range(dis4, dis3), axis=0)

# Concatenacion de los CSV de entrenamiento
sig_t = pd.concat([signo_ok, signo_ok_2, signo_ok_3, signo_ok_4], axis=0)

# Se escoge los X y Y, de entrenamiento y testeo
X_train, X_test, y_train, y_test = train_test_split(sig_t.iloc[:, :-1], sig_t.iloc[:, -1], random_state=0,
                                                    test_size=0.20)

np.save("sample_ref3", X_train)
np.save("sample2_ref3", y_train)
np.save("sample3_ref3", X_test)

## Normalizacion de datos
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)


## Verificacion de parametros adecuados segun los datos de entrenamiento
k_range = range(1, int(np.sqrt(len(y_train))))
dis=['manhattan','chebyshev', 'minkowski']

MCC=[]
F1=[]
distancia=[]
ki=[]

#Se encuentran las metricas
for i in dis:
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors = k,weights='distance',metric=i, metric_params=None,algorithm='brute')
        knn.fit(X_train, y_train)
        y_pred=knn.predict(X_test)
        ## Metricas
        MCC.append(matthews_corrcoef(y_test,y_pred))
        F1.append(f1_score(y_test,y_pred,average='micro'))
        distancia.append(i)
        ki.append(k)

## Metricas de evaluacion para KNN
print("########################################################################"+"\n")
maximo_MCC = MCC.index(max(MCC))
print("Con knn: Segun MCC({}) el mejor k es {} y la distancia es {}:".format(max(MCC),ki[maximo_MCC],distancia[maximo_MCC]))
maximo_F1  = F1.index(max(F1))
print("Con knn: Segun F1({}) el mejor k es {} y la distancia es {}:".format(max(F1),ki[maximo_F1],distancia[maximo_F1]))
print(classification_report(y_test, y_pred))
