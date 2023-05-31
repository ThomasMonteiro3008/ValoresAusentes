#!/usr/bin/env python
# coding: utf-8

# 
# ## <font color='blue'>Projeto </font>
# ## <font color='blue'>Análise de Dados Para Campanhas de Marketing de Instituições Financeiras</font>

# ## Instalando e Carregando os Pacotes

# In[1]:


# Versão da Linguagem Python
from platform import python_version
print('Versão da Linguagem Python Usada Neste Jupyter Notebook:', python_version())


# In[2]:


# Para atualizar um pacote, execute o comando abaixo no terminal ou prompt de comando:
# pip install -U nome_pacote

# Para instalar a versão exata de um pacote, execute o comando abaixo no terminal ou prompt de comando:
# !pip install nome_pacote==versão_desejada

# Depois de instalar ou atualizar o pacote, reinicie o jupyter notebook.

# Instala o pacote watermark. 
# Esse pacote é usado para gravar as versões de outros pacotes usados neste jupyter notebook.
# !pip install -q -U watermark


# In[3]:


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[4]:


# Versões dos pacotes usados neste jupyter notebook
get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-a "Data Science Academy" --iversions')


# ## Carregando os Dados

# In[5]:


# Carrega o dataset
df = pd.read_csv("dados/dataset.csv") 


# In[6]:


# Shape
df.shape


# In[7]:


# Amostra
df.head()


# ## Análise Exploratória

# In[8]:


# Info
df.info()


# In[9]:


# Temos valores nulos?
df.isna().any()


# In[10]:


# Temos valores nulos?
df.isna().sum()


# In[11]:


# Não usaremos a coluna ID. Vamos removê-la.
df.drop(["customerid"], axis = 1, inplace = True)


# In[12]:


# Colunas
df.columns


# > Exercício 1: A coluna "jobedu" parece ter duas informações. Vamos separar em duas colunas.

# In[13]:


df.head()


# In[14]:


# Fazemos o split da coluna jobedu e criamos a coluna job com o primeiro elemento antes da vírgula
df['job'] = df["jobedu"].apply(lambda x:x.split(",")[0])


# In[15]:


df.head()


# In[16]:


# Fazemos o split da coluna jobedu e criamos a coluna education com o segundo elemento antes da vírgula
df['education'] = df["jobedu"].apply(lambda x:x.split(",")[1])


# In[17]:


df.head()


# In[18]:


# Drop da coluna "jobedu" 
df.drop(["jobedu"], axis = 1, inplace = True)


# In[19]:


df.head()


# ## Tratamento de Valores Ausentes

# > Vamos primeiro tratar a variável que representa a idade.

# In[20]:


# Valores ausentes no dataframe
df.isna().any()


# In[21]:


# Valores ausentes da variável age
df.age.isnull().sum()


# In[22]:


# Calcula o percentual de valores ausentes na variável age
df.age.isnull().mean()*100


# Como o percentual é baixo não podemos eliminar a coluna. Podemos então eliminar os registros com valores ausentes (nesse caso perderíamos 20 linhas no dataset) ou podemos aplicar imputação. Vamos usar a segunda opção.

# In[23]:


# Histograma
df.age.plot(kind = "hist")
plt.title("Histograma da Variável Idade\n")
plt.show()


# In[24]:


# Boxplot
sns.boxplot(df.age)
plt.title("Boxplot da Variável Idade\n")
plt.show()


# In[25]:


# Vamos verificar qual é a média de idade.
df.age.mean()


# In[26]:


# Vamos verificar qual é a mediana, valor do meio da distribuição quando os dados estão ordenados.
df.age.median()


# In[27]:


# Vamos verificar qual é a moda, o valor que aparece com mais frequência.
df.age.mode()


# > Exercício 2: Vamos imputar os valores ausentes da variável age com uma medida de tendência central. Escolha uma das medidas, aplique a imputação e justifique sua escolha. Deixamos a variável como float ou como int? Se convertemos, fazemos isso antes ou depois da imputação?

# In[28]:


# Vamos preencher com a moda pois são poucos valores ausentes e assim alteramos muito pouco o padrão nos dados.
df.age.fillna("32", inplace = True)


# In[29]:


# Agora convertemos para int
df.age = df.age.astype("int")


# In[30]:


# Tipo da variável
df.age.dtypes


# In[31]:


# Média
df.age.mean()


# In[32]:


# Mediana
df.age.median()


# In[33]:


# Percentual de valores ausentes
df.age.isnull().mean()*100


# ## Tratamento de Valores Ausentes

# > Vamos agora tratar a variável que representa o mês.

# In[34]:


# Valores ausentes no dataframe
df.isna().any()


# In[35]:


# Valores ausentes na variável
df.month.isnull().sum()


# In[36]:


# Percentual de valores ausentes
df.month.isnull().mean()*100


# Como o percentual é menor que 30% não podemos eliminar a coluna. Podemos então eliminar os registros com valores ausentes (nesse caso perderíamos 50 linhas no dataset) ou podemos aplicar imputação. Vamos usar a segunda opção.

# In[37]:


# Tipo da variável
df.month.dtypes


# In[38]:


# Categorias da variável
df.month.value_counts()


# > Exercício 3: Vamos imputar os valores ausentes da variável month. Escolha uma estratégia e aplique no dataset.

# In[39]:


# Vamos imputar com a moda, o valor mais frequente da variável, pois são poucos registros
df.month.mode()


# In[40]:


# Imputação com a moda
df.month.fillna("may, 2017", inplace = True)


# In[41]:


# Valores ausentes tratados com sucesso
df.month.isnull().sum()


# ## Tratamento de Valores Ausentes

# > Vamos agora tratar a variável que representa o salário.

# In[42]:


# Valores ausentes no dataframe
df.isna().any()


# In[43]:


# Valores ausentes na variável
df.salary.isnull().sum()


# In[44]:


# Calcula o percentual de valores ausentes na variável salary
df.salary.isnull().mean()*100


# Como o percentual é baixo não podemos eliminar a coluna. Podemos então eliminar os registros com valores ausentes (nesse caso perderíamos 26 linhas no dataset) ou podemos aplicar imputação. Vamos usar a segunda opção.
# 
# Mas espere. Vamos checar algo aqui.

# In[45]:


df.head()


# Existe salário igual a zero? Não. O valor zero é provavelmente um outlier (confirmar com a área de negócio).

# In[46]:


# Histograma
df.salary.plot(kind = "hist")
plt.title("Histograma da Variável Salário\n")
plt.show()


# In[47]:


# Boxplot
sns.boxplot(df.salary)
plt.title("Boxplot da Variável Salário\n")
plt.show()


# In[48]:


# Vamos verificar qual é a média de idade.
df.salary.mean()


# In[49]:


# Vamos verificar qual é a mediana.
df.salary.median()


# In[50]:


# Vamos verificar qual é a moda.
df.salary.mode()


# > Exercício 4: Vamos imputar os valores ausentes da variável salary com uma medida de tendência central. Precisamos também tratar os valores iguais a zero. Escolha sua estratégia, aplique a imputação e justifique sua escolha. 

# In[51]:


# Vamos preencher com a mediana pois os dados parecem assimétricos (nesse caso a média não pode ser usada) 
# e o valor mais frequente está muito abaixo da média e da mediana (por isso não usaremos a moda)
df.salary.fillna("60000", inplace = True)


# In[52]:


df.head()


# In[53]:


# Histograma (vai gerar erro)
df.salary.plot(kind = "hist")
plt.title("Histograma da Variável Salário\n")
plt.show()


# In[54]:


# Tipo da variável
df.salary.dtypes


# In[55]:


# Convertemos para o tipo float
df.salary = df.salary.astype("float")


# In[56]:


# Tipo da variável
df.salary.dtypes


# In[57]:


# Histograma
df.salary.plot(kind = "hist")
plt.title("Histograma da Variável Salário\n")
plt.show()


# In[58]:


# Boxplot
sns.boxplot(df.salary)
plt.title("Boxplot da Variável Salário\n")
plt.show()


# In[59]:


# Registros para cada salário
df.salary.value_counts()


# In[60]:


# Replace do zero pela mediana
df['salary'] = df['salary'].replace(0, df['salary'].median())


# In[61]:


# Registros para cada salário
df.salary.value_counts()


# In[62]:


# Histograma
df.salary.plot(kind = "hist")
plt.title("Histograma da Variável Salário\n")
plt.show()


# In[63]:


# Boxplot
sns.boxplot(df.salary)
plt.title("Boxplot da Variável Salário\n")
plt.show()


# In[64]:


# Calcula o percentual de valores ausentes na variável salary
df.salary.isnull().mean()*100


# In[65]:


df.isna().any()


# ## Tratamento de Valores Ausentes

# > Vamos agora tratar a variável que representa a resposta (variável alvo).

# In[66]:


df.head()


# In[67]:


# Valores ausentes
df.response.isnull().sum()


# In[68]:


# Calcula o percentual
df.response.isnull().mean()*100


# Como o percentual é baixo (e a variável é o alvo da nossa análise) não podemos eliminar a coluna. Podemos então eliminar os registros com valores ausentes (nesse caso perderíamos 30 linhas no dataset) ou podemos aplicar imputação.

# > Exercício 5: Escolha sua estratégia, aplique e justifique sua escolha. 

# In[69]:


# Não devemos aplicar imputação na variável de estudo (variável resposta ou variável alvo)
# Vamos dropar os registros
df.dropna(subset = ["response"], inplace = True)


# In[70]:


# Verifca valores NA
df.isnull().sum()


# ## Tratamento de Valores Ausentes

# > Vamos agora tratar a variável pdays.

# In[71]:


# Valores ausentes
df.pdays.isnull().sum()


# In[72]:


# Describe
df.pdays.describe()


# -1 indica valor ausente

# In[73]:


# Vamos fazer relace de -1 por NaN
df.pdays = df.pdays.replace({-1.0:np.NaN})


# In[74]:


# Valores ausentes
df.isnull().sum()


# In[75]:


# Calcula o percentual
df.pdays.isnull().mean()*100


# > Exercício 6: Escolha sua estratégia, aplique e justifique sua escolha. 

# In[76]:


# Drop da coluna "pdays" pois tem mais de 30% dos valores ausentes
df.drop(["pdays"], axis = 1, inplace = True)


# In[77]:


# Valores ausentes
df.isnull().sum()


# ## Conclusão e Análise dos Dados

# ### Análise Univariada

# In[78]:


# Proporção da variável de estado civil
df.marital.value_counts(normalize = True)


# In[79]:


# Plot
df.marital.value_counts(normalize = True).plot(kind = "barh")
plt.title("Proporção da variável de estado civil\n")
plt.legend()
plt.show()


# In[80]:


# Proporção da variável de job
df.job.value_counts(normalize = True)


# In[81]:


# Plot
plt.figure(figsize = (10,6))
df.job.value_counts(normalize = True).plot(kind = "barh")
plt.title("Proporção da variável de job\n", fontdict = {'fontsize': 20, 'fontweight' : 5, 'color' : 'Green'})
plt.legend()
plt.show()


# In[82]:


# Proporção da variável de education
df.education.value_counts(normalize = True)


# In[83]:


# Plot
plt.figure(figsize = (10,6))
df.education.value_counts(normalize = True).plot(kind = "pie")
plt.title("Proporção da variável de education\n", fontdict = {'fontsize': 20, 'fontweight' : 5, 'color' : 'Green'})
plt.legend()
plt.legend(bbox_to_anchor=(1.31,0.4))
plt.show()


# In[84]:


# Proporção da variável response
df.response.value_counts(normalize = True)


# In[89]:


# Plot
plt.figure(figsize = (10,6))
df.response.value_counts(normalize = True).plot(kind = "pie")
plt.title("Proporção da variável response\n", fontdict = {'fontsize': 20, 'fontweight' : 5, 'color' : 'Green'})
plt.legend()
plt.show()


# ## Análise Multivariada

# In[90]:


# Scatter Plot
sns.scatterplot(df["balance"], df["salary"])
plt.title("Scatter Plot Entre Saldo e Salário\n", fontdict = {'fontsize': 20, 'fontweight' : 5, 'color' : 'Green'})
plt.show()


# In[91]:


# Scatter Plot
sns.scatterplot(df["balance"], df["age"])
plt.title("Scatter Plot Entre Saldo e Idade\n", fontdict = {'fontsize': 20, 'fontweight' : 5, 'color' : 'Green'})
plt.show()


# In[92]:


# Pair Plot
sns.pairplot(df[["salary","balance","age"]])
plt.show()


# In[93]:


# Calcula a correlação
res = df[["salary", "balance", "age"]].corr()


# In[94]:


# Mapa de Correlação
plt.figure(figsize = (10,5))
sns.heatmap(res, annot = True, cmap = "Reds")
plt.title("Mapa de Correlação\n", fontdict = {'fontsize': 20, 'fontweight' : 5, 'color' : 'Green'})
plt.show()


# ### Numérico x Categórico

# In[95]:


# Agrupa o salário pela variável resposta e calcula a média
df.groupby(by = ["response"])["salary"].mean()


# In[96]:


# Agrupa o salário pela variável resposta e calcula a mediana
df.groupby(by = ["response"])["salary"].median()


# In[97]:


# Boxplot
plt.figure(figsize = (10,5))
sns.boxplot(df["response"], df["salary"])
plt.title("Salário x Resposta\n", fontdict = {'fontsize': 20, 'fontweight' : 5, 'color' : 'Green'})
plt.show()


# In[98]:


# Agrupa educação por salário e calcula a média
df.groupby(by = ["education"])["salary"].mean()


# In[99]:


# Cria a variável response_flag como tipo numérico onde response "yes"= 1, "no"= 0
df["response_flag"] = np.where(df["response"] == "yes",1,0)
df.head()


# In[100]:


# Mapa de correlação
res1 = df.pivot_table(index = "education", columns = "marital", values = "response_flag", aggfunc = "mean")
sns.heatmap(res1, annot = True, cmap = "RdYlGn")
plt.title("Education vs Marital vs Response Flag\n", fontdict = {'fontsize': 20, 'fontweight' : 5, 'color' : 'Green'})
plt.show()


# # Fim
