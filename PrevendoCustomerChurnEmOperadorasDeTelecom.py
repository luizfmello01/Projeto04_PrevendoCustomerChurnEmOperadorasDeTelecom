# Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import shapiro
from scipy.stats import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
# !pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# !pip install xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import pickle

# Config pandas
pd.set_option("max_columns", 1000)


# In[2]:


# Datasets
treino_raw = pd.read_csv("../Datasets/telecom_treino.csv", sep = ",")
teste_raw = pd.read_csv("../Datasets/telecom_teste.csv", sep = ",")


# ## 3.0 - Análise Exploratória de Dados

# In[3]:


# Primeiras linhas do dataset
treino_raw.head()


# A coluna "Unnamed: 0" pode ser removida do dataset porque é um índice e para o treinamento do modelo a coluna não tem relevância.<br/>
# As colunas categóricas e TARGET estão com o tipo de dados em String, será realizado um LabelEncoder (Transformação de categorias em valores numéricos) para melhor performance do modelo.

# In[4]:


# Dimensões do dataset
treino_raw.shape


# O dataset tem um bom número de registro, porém tem que ter cuidado ao realizar remoção de registros para não afetar a performance do modelo.

# In[5]:


# Tipo de dado de cada atributo
treino_raw.dtypes


# Conforme visto anteriormente, o dataset tem algumas variáveis com o tipo Object (Strings) que representam variáveis categóricas, devem ser transformadas para números.

# In[6]:


# Valores missing
treino_raw.isna().sum()


# O dataset não contem valores missing.

# In[7]:


# Valores unicos de cada atributo
for i in treino_raw.columns:
    print(i, "-> ", treino_raw[i].nunique(), sep="")


# O dataset tem muitos valores unicos, o que indica que temos bastantes variáveis numéricas, pode-se fazer quantization das variáveis númericas para uma melhor performance do modelo

# In[8]:


# Separar atributos númericos e categóricos
variaveis_numericas = []
variaveis_categoricas = []

for i in treino_raw.columns:
    if ( treino_raw[i].dtype == 'O' ):
        variaveis_categoricas.append(i)
    else:
        variaveis_numericas.append(i)


# In[9]:


# Retirar o atributo "Unnamed: 0" da lista de variaveis numericas
variaveis_numericas.remove("Unnamed: 0")


# ### Variáveis numéricas

# In[10]:


# Sumário estatístico das variáveis numéricas
treino_raw[variaveis_numericas].describe()


# A média e mediana das variáveis numéricas estão muito aproximadas, o desvio padrão está com valor baixo, indica que os dados estão próximos da média.

# In[11]:


# Gráfico de dispersão entre as variáveis total_day_minutes e total_day_charge
fig = plt.figure(figsize=(10,6))
sns.scatterplot(data = treino_raw[variaveis_numericas], x = "total_day_minutes", y = "total_day_charge")
plt.title("Relação de total_day_minutes e total_day_charge")
plt.show()


# De acordo com o scatterplot, temos uma correlação positiva das variáveis, conforme cresce os minutos falados no dia, sobe o custo.

# In[12]:


# Gráfico de dispersão entre as variáveis total_eve_minutes e total_eve_charge
fig = plt.figure(figsize=(10,6))
sns.scatterplot(data = treino_raw[variaveis_numericas], x = "total_eve_minutes", y = "total_eve_charge")
plt.title("Relação de total_eve_minutes e total_eve_charge")
plt.show()


# De acordo com o scatterplot, temos uma correlação positiva das variáveis, conforme cresce os minutos falados na vespera, sobe o custo, porém o custo é menor do que os minutos falados no dia e custo do dia.

# In[13]:


# Histograma
treino_raw[variaveis_numericas].hist(figsize=(16,10))
plt.show()


# De acordo com os histogramas, as variáveis númericas estão aparentemente em uma distribuição normal com exceção da variável "number_vmail_messages"

# In[14]:


# Boxplot
treino_raw[variaveis_numericas].plot(kind='box', layout=(5,3), subplots=True, figsize=(20,15))
plt.show()


# De acordo com os boxplots, todas as variáveis numéricas tem valores outliers, a maioria dos outliers estão concentrados no 1º e 3º quartil.

# In[15]:


# Teste de hipótese para validar distribuição gaussiana
# H0 -> É uma distribuição gaussiana
# Ha -> Não é uma distribuição gaussiana
def teste_gaussiano(nome_variavel, dados_variavel):
    alpha = 0.05
    stat, p = shapiro(dados_variavel)
    
    if p > alpha:
        print("Variável", nome_variavel, "está em uma distribuição gaussiana (Falha ao rejeitar H0)")
        print("\tP-Value: %.3f" % p)
    else:
        print("Variável", nome_variavel, "não está em uma distribuição gaussiana (Rejeitar H0)")
        print("\tP-Value: %.3f" % p)
    print()


# In[16]:


for i in variaveis_numericas:
    teste_gaussiano(i, treino_raw[i])


# Algumas variáveis não estão em uma distribuição normal, será realizado normalização nas variáveis para melhor performance do modelo preditivo.

# In[17]:


# Correlação das variáveis numéricas
figure = plt.figure(figsize=(16,10))
sns.heatmap(treino_raw[variaveis_numericas].corr(),
            cmap='BrBG', vmin=-1, vmax=1, annot=True)
plt.show()


# O dataset tem poucas variáveis correlacionadas, porém as correlações existentes são fortes.

# In[18]:


# Simetria das variáveis numéricas
treino_raw[variaveis_numericas].skew()


# As variáveis estão simétricas, algumas com uma leve distorção para a cauda esquerda e outras para a cauda direita, mas o valor da distorção é muito baixo.

# ### Variáveis categóricas

# In[19]:


# Barplot de cada variável agrupado pela variável TARGET (Churn)
for i in treino_raw[variaveis_categoricas].drop("churn", axis = 1).columns:
    pd.crosstab(treino_raw[i], treino_raw["churn"]).plot(kind = "bar",
                                                         stacked = True,
                                                         figsize = (12,6),
                                                         title = i)


# In[20]:


# Countplot para contar os valores das variáveis categóricas
for i in variaveis_categoricas:
    if ( i == "state" ):
        val_figsize = (16,6)
    else:
        val_figsize = (10,6)
        
    fig = plt.figure(figsize=val_figsize)
    sns.countplot(x = i, data = treino_raw)
    plt.show()


# De acordo com os gráfico de barras, os estados com mais registros são WV e MN, o código de área com mais registro é area_code_415, a maioria dos registros não tem plano internacional e não tem plano de correio de voz. A variável TARGET (churn) está desbalanceada, tem mais de 2500 registros com churn no e aproximadamente 500 registros com churn yes, para melhor performance do modelo preditivo, precisa-se balancear a variável churn.

# In[21]:


# Label Encoder das variáveis categóricas para realizar análise de correlação
treino_raw_le = treino_raw.copy()

for i in variaveis_categoricas:
    print("Realizando label encoder da variável", i)
    le = LabelEncoder()
    le.fit(treino_raw[i])
    treino_raw_le[i] = le.transform(treino_raw_le[i])


# In[22]:


# Exibir primeiras linhas do dataset depois do labelencoder das variáveis categóricas
treino_raw_le.head()


# In[23]:


# Análise de correlação com Spearman
sns.heatmap(treino_raw_le[variaveis_categoricas].corr(method="spearman"),
            cmap='BrBG', vmin=-1, vmax=1, annot=True)
plt.show()


# Realizado LabelEncoder nas variáveis para realizar análise de correlação. A correlação das variáveis preditoras com a variável target é fraca, muito próxima de zero, a variável preditora que tem mais correlação com a variável target é International_Plan.

# ## 4.0 - Manipulação de dados

# In[24]:


# Cópia do dataset original para efetuar as manipulações
treino_munging = treino_raw.copy()


# Declarar função para estruturar e manipular os dados do dataset para treinar o modelo de machine learning.<br/>
# As técnicas utilizadas dentro da função "estruturar_dados" foram criadas de acordo com as necessidades descobertas na análise exploratória.

# In[25]:


# Declarar e Treinar LabelEncoder para cada variável categorica
le = []

# Adicionar cada variável categorica no LabelEncoder
for i in variaveis_categoricas:
    le.append((i, LabelEncoder()))
    
# Treinar LabelEncoder de cada variável categórica
for var, modelo in le:
    modelo.fit(treino_munging[var])
    print("Concluído LabelEncoder da variável", var)


# In[26]:


# Função para realizar a manipulação dos dados
standard_scaler = StandardScaler()

def tratar_dados(dataset):
    # Remover variável "Unnamed: 0" do dataset
    try:
        dataset = dataset.drop(["Unnamed: 0"], axis = 1)
        print("Concluído remoção da variável 'Unnamed: 0'")
    except:
        print("O dataset não tem a variável 'Unnamed: 0'")
    
    # Transformar variáveis categóricas que estão em formato de texto para número     
    for var, modelo in le:
        try:   
            dataset[var] = modelo.fit_transform(dataset[var])
            print("Concluído LabelEncoder da variável", var)
        except:
            print("O dataset não tem a variável", var)
        
    # Normalizar as variáveis numéricas  
    dataset[variaveis_numericas] = standard_scaler.fit_transform(X = dataset[variaveis_numericas], y = dataset["churn"])
    print("Concluído normalização das variáveis numéricas")
        
    return dataset

# Função para inverter o label encoder das variáveis categóricas e normalização das variáveis numéricas
def inverter_dados(dataset):  
    # Transformar variáveis categóricas que estão em formato de texto para número     
    for var, modelo in le:
        try:   
            dataset[var] = modelo.inverse_transform(dataset[var])
            print("Concluído inversão de LabelEncoder da variável", var)
        except:
            print("O dataset não tem a variável", var)
        
    # Normalizar as variáveis numéricas
    dataset[variaveis_numericas] = standard_scaler.inverse_transform(X = dataset[variaveis_numericas])
        
    return dataset


# In[27]:


# Utilizar a função para realizar a manipulação dos dados (Remover variável "Unnamed: 0", tranformar variáveis categóricas que 
# estão em formato texto para número e normalizar as variáveis númericas)
treino_munging = tratar_dados(treino_munging)


# In[28]:


# Exibir primeiras linhas do dataset manipulado
treino_munging.head()


# O dataset está com as variáveis categóricas transformadas para número e as variáveis numéricas estão normalizadas, mas será que essas variáveis numéricas estão mesmo normalizadas? Vamos plotar histograma para confirmar.

# In[29]:


# Histograma das variáveis numéricas
treino_munging[variaveis_numericas].hist(figsize=(15,10))
plt.show()


# As variáveis númericas com exceção da variável "number_vmail_messages" estão em formato de distribuição normal.

# In[30]:


# Boxplot das variáveis numéricas
treino_munging[variaveis_numericas].plot(kind='box', layout=(5,3), subplots=True, figsize=(20,15))
plt.show()


# De acordo com os graficos exibidos acima, as variáveis númericas do dataset tem muitos outliers, será realizado remoção dos outliers.

# In[31]:


# Função para remover outlier da variável com método do desvio padrão
def remover_outlier(valor):
    # Calcular estatística de média e desvio padrão
    media, desvio_padrao = np.mean(valor), np.std(valor)
    
    # Identificar outliers
    valor_corte = desvio_padrao * 2.5
    baixo, alto = media - valor_corte, media + valor_corte
    valor = np.where(valor > alto, np.nan, valor)
    valor = np.where(valor < baixo, np.nan, valor)
    
    return valor


# In[32]:


# Remover outliers do dataset utilizando a função criada a cima e gravar em uma nova variável #

# Copiar dataset e gravar em uma nova variável
treino_munging_no_outliers = treino_munging.copy()

# Percorrer cada variável numérica e retirar outlier de cada variável.
for i in variaveis_numericas:
    treino_munging_no_outliers[i] = remover_outlier(treino_munging_no_outliers[i])
    
# A função remover_outlier deixa os valores outliers como nulos, vamos remover os valores nulos do dataset
treino_munging_no_outliers = treino_munging_no_outliers.dropna()


# In[33]:


# Boxplot das variáveis numéricas
treino_munging_no_outliers[variaveis_numericas].plot(kind='box', layout=(5,3), subplots=True, figsize=(20,15))
plt.show()


# De acordo com os gráficos acima, foram removidos os outliers do dataset.

# In[34]:


# Exibir primeiras linhas do dataset
treino_munging_no_outliers.head()


# In[35]:


# Dimensões do dataset
treino_munging_no_outliers.shape


# O dataset continua com uma boa quantidade de registros para o treinamento do modelo de machine learning, foram removidos 514 registros ao eliminar os outliers das variáveis numéricas.

# ## 5.0 - Feature Selection (Seleção de variáveis)

# In[36]:


# Utilizar o método SelectKBest do SKLearn com o método estatistico f_classif (ANOVA) para selecionas as melhores variáveis para o modelo de machine learning
predict = treino_munging_no_outliers.drop(["churn"], axis = 1)
target = treino_munging_no_outliers["churn"]

kbest = SelectKBest(f_classif, k=15)
kbest.fit_transform(
    X = predict, 
    y = target)
print("Concluído treinamento do SelectKBest com método ANOVA")


# In[37]:


# Criar dataframe com o resultado da seleção de variáveis
resultado_fs = pd.DataFrame({
    'Variavel': predict.columns,
    'Selecionado': kbest.get_support(),
    'Score': kbest.scores_
}).sort_values("Score", ascending = False).reset_index().drop("index", axis = 1)


# In[38]:


# 8 variáveis com mais score para o treinamento do modelo
variaveis_predict = resultado_fs.iloc[0:8].Variavel.values


# In[39]:


# Manter somente as variáveis que foram selecionadas na lista de variáveis numéricas
variaveis_numericas_novas = []

for i in variaveis_numericas.copy():
    if ( i in variaveis_predict ):
        variaveis_numericas_novas.append(i)
        
variaveis_numericas = variaveis_numericas_novas.copy()


# In[40]:


# Exibir variaveis preditoras selecionadas
variaveis_predict


# As variaveis exibidas acima serão utilizadas para o treinamento do modelo de machine learning porque são as variáveis que tiveram mais score na seleção de variáveis

# ## 6.0 - Preparar dataset de treino e teste para o treinamento

# In[41]:


# Primeiras linhas do dataset de teste
teste_raw.head()


# In[42]:


# Tratar o dataset de teste
teste_munging = tratar_dados(teste_raw)


# In[43]:


# Manter somente variáveis da feature selection (seleção de variáveis) nos datasets e separar dados de treino e de teste #

# Dataset de treino
x_treino = treino_munging_no_outliers[variaveis_predict]
y_treino = treino_munging_no_outliers["churn"]

# Dataset de teste
x_teste = teste_munging[variaveis_predict]
y_teste = teste_munging["churn"]


# In[44]:


# Primeiras linhas do dataset de teste sem a variável target após o tratamento dos dados
x_teste.head()


# In[45]:


# Balancear dataset de treino
x_treino_balanceado, y_treino_balanceado = SMOTE().fit_resample(x_treino, y_treino)


# In[46]:


# Dimensões do dataset de treino balanceado
print("Dimensões das variáveis preditoras:", x_treino_balanceado.shape)
print("Quantidade de registros da variável target:", y_treino_balanceado.count())


# Dataset está com 4940 registros, com 8 variáveis preditoras e uma variável target após o balanceamento da classe target.

# ## 7.0 - Treinamento do Modelo de Machine Learning

# In[47]:


def obter_modelos():
    modelos = []
    modelos.append( ("Naive Bayes", GaussianNB()) )
    modelos.append( ("Regressão Logística", LogisticRegression()) )
    modelos.append( ("KNN", KNeighborsClassifier()) )
    modelos.append( ("SVM Classificador", SVC()) )
    
    return modelos


# In[48]:


# Treinar modelos de regressão logistica e naive bayes com a técnica de cross validation
modelos = obter_modelos()
k = KFold(n_splits=10)

for nome, modelo in modelos:
    cv = cross_validate(estimator = modelo, 
                        X = x_treino_balanceado, 
                        y = y_treino_balanceado,
                        cv = k,
                        scoring=['accuracy', 'recall', 'precision'])
    print( "%s:\n\tAcurácia: %.3f \n\tRecall: %.3f \n\tPrecisão: %.3f \n" % 
          ( nome, 
            np.mean(cv["test_accuracy"]),
            np.mean(cv["test_recall"]),
            np.mean(cv["test_precision"])) )


# Os modelos que apresentaram melhores resultados foram KNN e SVM, vamos realizar treinamento desses modelos individualmente para serem avaliados com dados de teste.

# In[49]:


# Treinamento do modelo KNN
modelo_knn_v1 = KNeighborsClassifier()
modelo_knn_v1.probability=True
modelo_knn_v1.fit(X = x_treino_balanceado, y = y_treino_balanceado)
print("Concluído treinamento do algoritmo KNN")


# In[50]:


# Treinamento do modelo SVM Classifier
modelo_svm_v1 = SVC()
modelo_svm_v1.probability=True
modelo_svm_v1.fit(X = x_treino_balanceado, y = y_treino_balanceado)
print("Concluído treinamento do algoritmo SVM")


# ## 8.0 - Avaliação do Modelo de Machine Learning

# Métricas escolhida para avalição do modelo: Recall 

# In[51]:


# Avaliação do modelo KNN com os dados de teste
previsao_knn_v1 = modelo_knn_v1.predict(x_teste)
print( "Avaliação do modelo KNN:\n" )
print( classification_report(y_teste, previsao_knn_v1) )


# In[52]:


# Avaliação do modelo SVM com os dados de teste
previsao_svm_v1 = modelo_svm_v1.predict(x_teste)
print( "Avaliação do modelo SVM:\n" )
print( classification_report(y_teste, previsao_svm_v1) )


# O modelo SVM foi superior na métrica recall, esse modelo é o escolhido para receber otimização de hiperparametros.

# ## 9.0 - Otimização do Modelo de Machine Learning

# In[53]:


# Treinar modelo Random Forest para realizar otimização
modelo_rfc_v1 = RandomForestClassifier(n_estimators=1000)
modelo_rfc_v1.fit(x_treino_balanceado, y_treino_balanceado)
print("Concluído treinamento do algoritmo Random Forest")


# In[54]:


# Avaliação do modelo Random Forest com os dados de teste
previsao_rfc_v1 = modelo_rfc_v1.predict(x_teste)
print( "Avaliação do modelo Random Forest Classifier:\n" )
print( classification_report(y_teste, previsao_rfc_v1) )


# O modelo Random Forest não teve um recall superior ao modelo SVM.

# In[55]:


# Treinar modelo XGBoost para realizar otimização
modelo_xgb_v1 = XGBClassifier(n_estimators=500)
modelo_xgb_v1.fit(x_treino_balanceado, y_treino_balanceado)
print("Concluído treinamento do algoritmo XGBoost")


# In[56]:


# Avaliação do modelo XGBoost com os dados de teste
previsao_xgb_v1 = modelo_xgb_v1.predict(x_teste)
previsao_xgb_v1 = [round(value) for value in previsao_xgb_v1]
print( "Avaliação do modelo XGBoost Classifier:\n" )
print( classification_report(y_teste, previsao_xgb_v1) )


# O modelo XGBoost não teve um recall superior ao modelo SVM.

# In[57]:


# Otimizar hiperparametros do modelo SVM com GridSearchCV #

# Parametros do Grid
param_grid = {'C': [0.1], 
              'kernel': ['rbf'],
              'gamma': ['scale'],
              'tol': [0.001],
              'class_weight': [{0:1.0, 1:1.10}, {0:1.0, 1:1.12}]
             }

# Treinar GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=KFold(n_splits=6), scoring='recall')
grid.fit(x_treino_balanceado, y_treino_balanceado)


# In[58]:


# Exibir melhores parametros encontrados com o GridSearchCV
print("Melhores parametros:")
print(grid.best_params_)


# In[59]:


pd.DataFrame(grid.cv_results_).sort_values(["rank_test_score"]).head()


# In[60]:


# Avaliar modelo treinado com GridSearchCV utilizando os dados de teste
grid_previsoes = grid.predict(x_teste)
print( "Matriz de confusão:\n" )
print( confusion_matrix(y_teste, grid_previsoes) )
print( "\nRelatório de classificação:\n" )
print( classification_report(y_teste, grid_previsoes) )


# O modelo treinado com os hyperparametros ('C': 0.1, 'class_weight': {0: 1.0, 1: 1.12}, 'gamma': 'scale', 'kernel': 'rbf', 'tol': 0.001) será o modelo escolhido para entrega final do projeto.
# Modelo teve uma queda de 3% de recall para a classe 0 (No), porém teve um aumento de 5% de recall para a classe 1 (Yes).

# ## 10.0 - Salvar o Modelo de Machine Learning para Entrega Final do Projeto

# In[61]:


# Treinamento do modelo final
modelo_svm_final = SVC(C=0.1, class_weight={0:1.0, 1:1.12}, gamma='scale', kernel='rbf', tol=0.001)
modelo_svm_final.probability = True
modelo_svm_final.fit(x_treino_balanceado, y_treino_balanceado)
print( "Treinamento do Modelo SVM (Versão final) realizado com sucesso" )


# In[62]:


# Prever resultado de Churn e a probabilidade para os dados de teste
previsao_modelo_svm_final = modelo_svm_final.predict(x_teste)
previsao_prob_modelo_svm_final = modelo_svm_final.predict_proba(x_teste)
df_previsoes = pd.DataFrame({
    'churn': pd.Series(previsao_modelo_svm_final, dtype=np.int32),
    'Probabilidade_ChurnNo': pd.Series(np.round(previsao_prob_modelo_svm_final.transpose()[0], 2), dtype=np.float32),
    'Probabilidade_ChurnYes': pd.Series(np.round(previsao_prob_modelo_svm_final.transpose()[1], 2), dtype=np.float32)
})


# In[63]:


# Dataset de teste com a previsão e probabilidade de churn
teste_resultado = inverter_dados(x_teste.join(df_previsoes))
teste_resultado.head()


# In[64]:


# Salvar modelo de machine learning
nome_arquivo = "../modelo/modelo_svm_final.sav"
pickle.dump(modelo_svm_final, open(nome_arquivo, 'wb'))


# ## FIM.
