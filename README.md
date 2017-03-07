{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Titanic: Machine Learning from Disaster\n",
    "##### Prevendo os sobreviventes com a utilização de Regressão Linear\n",
    "---\n",
    "Solução para a competição \"Titanic: Machine Learning from Disaster\" utilizando Regressão Linear.\n",
    "\n",
    "A análise abaixo é de caráter exploratório, apenas para aprender o básico da importação e tratamento de dados no Python, além de demonstrar que o método de Regressão Linear pode ser poderoso e ao mesmo tempo de simples implementação. \n",
    "\n",
    "*Carlos Melo*\n",
    "<a href=\"https://www.linkedin.com/in/carlosfab/\"><img src=\"https://brand.linkedin.com/etc/designs/linkedin/katy/global/clientlibs/img/default-share.png\" width=150></a>\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Definição do problema\n",
    "\n",
    "O naufrágio do Titanic foi um dos maiores desastres na história. Ocorrido em 15 de abril de 1912, após uma colisão com um *iceberg*, o navio afundou. O saldo final da tragédia foi um total de 1502 mortos de um total de 2224 pessoas (incluindo passageiros e tripulação). Um dos motivos para o grande número de vítimas (cerca de 68%) diz respeito à quantidade de botes bem inferior ao mínimo que seria necessário.\n",
    "\n",
    "Nessa situação, apesar do fator \"sorte\" ter feito parte na sobrevivência dos passageiros do Titanic, existiram grupos de pessoas que tinham uma maior probabilidade de escaparem da morte (como por exemplo mulheres, crianças e passageiros viajando na primeira-classe), como informa a [descrição do próprio desafio](https://www.kaggle.com/c/titanic).\n",
    "\n",
    "Dentro desse contexto, o site [Kaggle](http://www.kaggle.com) fornece então um conjunto de dados aos competidores do desafio, contendo algumas informações (nome, idade, sexo, sobreviveu(sim/não), etc.) sobre 891 passageiros. É esperado que o competidor conduza uma análise e crie um modelo capaz de prever, com a maior acurácia possível, a condição de sobrevivência para os passageiros restantes (*dataset* de teste).\n",
    "\n",
    "Um vídeo com a descrição do acidente do Titanic pode ser visto no link abaixo.\n",
    "\n",
    "<a href=\"https://www.youtube.com/watch?v=9xoqXVjBEF8\n",
    "\" target=\"_blank\"><img src=\"https://img.youtube.com/vi/9xoqXVjBEF8/0.jpg\" \n",
    "alt=\"TITANIC sinking\" width=\"240\" height=\"180\" border=\"10\" /></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Aquisição dos dados\n",
    "\n",
    "[O Kaggle fornece os dados](https://www.kaggle.com/c/titanic/data) no formato'.csv'. Os mesmos são divididos em:\n",
    "\n",
    "* conjunto de treinamento (train.csv): para ser utilizado para construir os modelos de *machine learning*. Neste arquivo, é conhecido se cada passageiro sobreviveu ou não.\n",
    "* conjunto de teste (test.csv): para verificar o quão bom o modelo construido é com dados não observados até então. Neste aqruivo, não é fornecida a condição de sobrevivência para cada passageiro. O modelo construído é que deve fazer essa previsão.\n",
    "\n",
    "Será utilizado, primariamente, a biblioteca *Pandas* para se trabalhar com os dados do Titanic."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Análise de dados\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "# Visualização\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "sns.set()\n",
    "%matplotlib inline\n",
    "\n",
    "# Classe de Regressão Linear\n",
    "from sklearn.linear_model import LinearRegression\n",
    "\n",
    "# Validação cruzada\n",
    "from sklearn.cross_validation import KFold\n",
    "\n",
    "# Warnings\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": false,
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Importando banco de dados de treinamento e teste como Dataframes\n",
    "train_data = pd.read_csv('data/train.csv')\n",
    "test_data = pd.read_csv('data/test.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Preparação e limpeza dos dados\n",
    "** Informações e visualização dos dados importados**\n",
    "\n",
    "Inicialmente, vamos ver as primeiras 5 entradas e as 5 últimas entradas das variáveis, a fim de gerar um conhecimento inicial de como o Dataframe está estruturado."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": false,
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>887</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>Montvila, Rev. Juozas</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>211536</td>\n",
       "      <td>13.00</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>888</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Graham, Miss. Margaret Edith</td>\n",
       "      <td>female</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>112053</td>\n",
       "      <td>30.00</td>\n",
       "      <td>B42</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>889</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Johnston, Miss. Catherine Helen \"Carrie\"</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>W./C. 6607</td>\n",
       "      <td>23.45</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>890</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Behr, Mr. Karl Howell</td>\n",
       "      <td>male</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>111369</td>\n",
       "      <td>30.00</td>\n",
       "      <td>C148</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>891</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Dooley, Mr. Patrick</td>\n",
       "      <td>male</td>\n",
       "      <td>32.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>370376</td>\n",
       "      <td>7.75</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass                                      Name  \\\n",
       "886          887         0       2                     Montvila, Rev. Juozas   \n",
       "887          888         1       1              Graham, Miss. Margaret Edith   \n",
       "888          889         0       3  Johnston, Miss. Catherine Helen \"Carrie\"   \n",
       "889          890         1       1                     Behr, Mr. Karl Howell   \n",
       "890          891         0       3                       Dooley, Mr. Patrick   \n",
       "\n",
       "        Sex   Age  SibSp  Parch      Ticket   Fare Cabin Embarked  \n",
       "886    male  27.0      0      0      211536  13.00   NaN        S  \n",
       "887  female  19.0      0      0      112053  30.00   B42        S  \n",
       "888  female   NaN      1      2  W./C. 6607  23.45   NaN        S  \n",
       "889    male  26.0      0      0      111369  30.00  C148        C  \n",
       "890    male  32.0      0      0      370376   7.75   NaN        Q  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data.tail()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Significado das variáveis**\n",
    "\n",
    "Com as 5 primeiras entradas impressas na tela, pode-se ter então uma noção de como os dados estão estruturados, bem como quais são as variáveis mais importantes para o início da investigação. Segue o significado de cada coluna do DataFrame:\n",
    "\n",
    "* *PassengerId:* ID do passageiro\n",
    "* *Survived:* Sobreviveu (== 0) ou Faleceu (== 1)\n",
    "* *Pclass:* Classe do passageiro (1st, 2nd, 3rd)\n",
    "* *Name:* Nome do passageiro\n",
    "* *Sex:* Sexo do passageiro\n",
    "* *Age:* Idade\n",
    "* *SibSp:* Número de irmãos/cônjuge a bordo (define as relações familiares)\n",
    "* *Parch:* Número de pais/filhos a bordo (crianças viajando com babás apenas, recebem *parch* = 0)\n",
    "* *Ticket:* Número do *ticket* de embarque\n",
    "* *Fare:* Tarifa paga pelo passageiro\n",
    "* *Cabin:* Número da cabine\n",
    "* *Embarked:* Porto de embarque\n",
    "\n",
    "A seguir, serão olhadas algumas outras informações sobre as variáveis, a fim de identificar o tipo de variável, a contagem do número de entradas para cada uma (para auxiliar a encontrar dados faltando ou fragmentados), valores mínimos e máximos, médias, etc.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "________________________________________\n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 12 columns):\n",
      "PassengerId    891 non-null int64\n",
      "Survived       891 non-null int64\n",
      "Pclass         891 non-null int64\n",
      "Name           891 non-null object\n",
      "Sex            891 non-null object\n",
      "Age            714 non-null float64\n",
      "SibSp          891 non-null int64\n",
      "Parch          891 non-null int64\n",
      "Ticket         891 non-null object\n",
      "Fare           891 non-null float64\n",
      "Cabin          204 non-null object\n",
      "Embarked       889 non-null object\n",
      "dtypes: float64(2), int64(5), object(5)\n",
      "memory usage: 83.6+ KB\n",
      "________________________________________\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>714.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>257.353842</td>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>14.526497</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>223.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>668.500000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       PassengerId    Survived      Pclass         Age       SibSp  \\\n",
       "count   891.000000  891.000000  891.000000  714.000000  891.000000   \n",
       "mean    446.000000    0.383838    2.308642   29.699118    0.523008   \n",
       "std     257.353842    0.486592    0.836071   14.526497    1.102743   \n",
       "min       1.000000    0.000000    1.000000    0.420000    0.000000   \n",
       "25%     223.500000    0.000000    2.000000         NaN    0.000000   \n",
       "50%     446.000000    0.000000    3.000000         NaN    0.000000   \n",
       "75%     668.500000    1.000000    3.000000         NaN    1.000000   \n",
       "max     891.000000    1.000000    3.000000   80.000000    8.000000   \n",
       "\n",
       "            Parch        Fare  \n",
       "count  891.000000  891.000000  \n",
       "mean     0.381594   32.204208  \n",
       "std      0.806057   49.693429  \n",
       "min      0.000000    0.000000  \n",
       "25%      0.000000    7.910400  \n",
       "50%      0.000000   14.454200  \n",
       "75%      0.000000   31.000000  \n",
       "max      6.000000  512.329200  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print('_'*40)\n",
    "train_data.info()\n",
    "print('_'*40)\n",
    "train_data.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "O método .info() retorna um resumo conciso do DataFrame, permitindo identificar o tipo da variável em cada coluna. Já o método .describe() retorna diferentes características das colunas numéricas do DataFrame.\n",
    "\n",
    "Com isso, é possível identificar que as colunas que possuem valores nulos ou em branco (ou seja, seus dados precisam ser corrigidos) são:\n",
    " * ~~'Cabin'~~\n",
    " * 'Age'\n",
    " * 'Embarked'\n",
    " \n",
    " Como existem poucos valores lançados para a categoria 'Cabin', além de não se enxergar inicialmente nenhuma influênca, preferiu-se desconsiderar esta coluna.\n",
    " \n",
    "Na coluna 'Age', os valores NaN serão substituidos pela sua mediana, o que parece ser mais razoável que excluir uma linha inteira que pode conter outros dados importantes. Na coluna 'Embarked', as lacunas serão substituidas pelo valor de maior frequência."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": false,
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['S' 'C' 'Q' nan]\n",
      "S    644\n",
      "C    168\n",
      "Q     77\n",
      "Name: Embarked, dtype: int64\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "count    891.000000\n",
       "mean      29.361582\n",
       "std       13.019697\n",
       "min        0.420000\n",
       "10%       16.000000\n",
       "25%       22.000000\n",
       "50%       28.000000\n",
       "75%       35.000000\n",
       "99%       65.000000\n",
       "max       80.000000\n",
       "Name: Age, dtype: float64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Verificando a frequência dos valores de 'Embarked'\n",
    "print(train_data['Embarked'].unique())\n",
    "print(train_data['Embarked'].value_counts())\n",
    "# Substituindo os valores nulos de 'Embarked' por 'S'\n",
    "train_data.loc[train_data['Embarked'].isnull(), 'Embarked'] = 'S'\n",
    "\n",
    "# Substituindo os valores nulos de 'Age' pela mediana da coluna\n",
    "train_data.loc[train_data['Age'].isnull(), 'Age'] = train_data['Age'].median()\n",
    "\n",
    "# Verificando 'Age' com o método .describe(), agora sem dados nulos (NaN)\n",
    "train_data['Age'].describe(percentiles=[.1, .25, .5, .75, .99])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Análise, identificação de padrões e exploração dos dados\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Inicialmente, vamos considerar a hipótese de que crianças e mulheres teriam tido mais chances de sobreviver, uma vez que provavelmente foram as primeiras a serem embarcadas nos botes.\n",
    "\n",
    "Para isso, vamos ver a porcentagem de sobreviventes na categoria 'Sex', analisar o histograma da distribuição das idades, classificar todos os passageiros em 3 grupos (Crianças, adultos e idosos) para vermos a porcentagem de sobreviventes em cada uma delas, e por último analisar se há diferença entre os passageiros das diferentes classes no navio."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\neb3e2OXMzEyNjIzIcRwFg0H5fL7YWl5enoLB4Ji3OZfCQt85185XYaHvon6HAwC4NCSyVWOJ+zNn\nr9erUCgUuxyNRmOR/eRaKBSSz+cb8zYAAGBsceM8e/Zs7d+/X5IUCARUWloaWyspKdHAwICGhoYU\nDofV09OjWbNmjXkbAAAwNo/ruu5YV4hGo2poaNCbb74p13XV1NSk119/XcPDw/L7/ers7NQjjzwi\n13VVVVWl22677ay3KSkpuVgzAQCQ0uLGGQAAXFychAQAAGOIMwAAxqTNW6g//jn3oUOHlJ2drcbG\nRhUXFyd7W+fllVde0YMPPqi2tjYNDAyovr5eHo9Hn//853XvvfcqIyM1vpeKRCJav369jh49qnA4\nrBUrVmjmzJkpOc/o6Kg2bNigw4cPy+PxaNOmTbrssstSchZJeuedd/TNb35T27dvl+M4KTuHJN1y\nyy2xX9mcNm2ali9fnrLzbNu2TZ2dnYpEIqqpqdHVV1+dkrPs2rVLTz/9tCTpww8/1MGDB/Wzn/1M\nTU1NKTdLJBJRfX29jh49qoyMDN1///0X92vGTRPPPvusu3btWtd1Xffll192ly9fnuQdnZ/HH3/c\nvfnmm91Fixa5ruu6d9xxh3vgwAHXdV33nnvucX//+98nc3vnpaOjw21sbHRd13UHBwfda6+9NmXn\n2bt3r1tfX++6ruseOHDAXb58ecrOEg6H3e985zvuDTfc4L711lspO4fruu4HH3zgfv3rXz/jY6k6\nz4EDB9w77rjDHR0ddYPBoPvwww+n7Cz/X0NDg7tjx46UnWXv3r3uqlWrXNd13RdeeMG98847L+os\n9r99GaexzmSWCoqKirRly5bY5ddee01XX321JGnOnDn605/+lKytnbf58+fru9/9riTJdV1lZmam\n7DzXX3+97r//fknSsWPHNHHixJSdpbW1VdXV1ZoyZYqk1H6MvfHGG3r//fe1dOlS3X777QoEAik7\nzwsvvKDS0lKtXLlSy5cv13XXXZeys3zs1Vdf1VtvvSW/35+ys0yfPl2jo6OKRqMKBoNyHOeizpI2\nL2t/mrOSWTJv3jwdOXIkdtl1XXk8HkkfnXntvffeS9bWzlteXp6kjz4nq1at0l133aXW1taUncdx\nHK1du1Z79+7Vww8/rK6urpSbZdeuXbr88stVUVGhxx9/XFJqP8YmTJiguro6LVq0SG+//baWLVuW\nsvMMDg7q2LFjeuyxx3TkyBGtWLEiZWf52LZt27Ry5UpJqfs4y83N1dGjR3XjjTdqcHBQjz32mLq7\nuy/aLKlRrnFIt7OS/f+fY4RCIU2cODGJuzl/x48f18qVK3XrrbdqwYIFeuCBB2JrqThPa2ur1qxZ\no8WLF+vDDz+MfTxVZvnFL34hj8ejF198UQcPHtTatWv1r3/9K7aeKnN8bPr06SouLpbH49H06dM1\nadIkvfbaa7H1VJpn0qRJmjFjhrKzszVjxgxddtll+uc//xlbT6VZJOndd9/V4cOH9eUvf1lS6v5f\n9uSTT+qrX/2qVq9erePHj+tb3/qWIpFIbP1Cz5I2L2un21nJvvjFL+qll16SJO3fv19f+tKXkryj\n8Tt16pSWLl2qu+++WwsXLpSUuvPs3r1b27ZtkyTl5OTI4/HoyiuvTLlZnnrqKf30pz9VW1ubvvCF\nL6i1tVVz5sxJuTk+1tHRoZaWFknSiRMnFAwG9ZWvfCUl5ykvL9fzzz8v13V14sQJvf/++7rmmmtS\nchZJ6u7u1jXXXBO7nKpf+xMnToz97Yj8/HyNjIxc1FnS5iQk6XBWsiNHjuj73/++du7cqcOHD+ue\ne+5RJBLRjBkz1NjYqMzMzGRvcVwaGxv1u9/9TjNmzIh97Ac/+IEaGxtTbp7h4WGtW7dOp06d0sjI\niJYtW6aSkpKU/dxIUm1trRoaGpSRkZGyc4TDYa1bt07Hjh2Tx+PRmjVrVFBQkLLz/OhHP9JLL70k\n13X1ve99T9OmTUvZWX7yk5/IcRwtWbJEklL2/7JQKKT169fr5MmTikQiuv3223XllVdetFnSJs4A\nAKSLtHlZGwCAdEGcAQAwhjgDAGAMcQYAwBjiDACAMcQZSCNXXHHFWT9eX1+vXbt2jfs4u3btUn19\nfaK2BeA8EWcAAIxJ3fNbAjgn13XV0tKiP/zhD5oyZYpGR0djJ+x/6KGH9OKLL+r06dMqKCjQli1b\nVFhYqN27d2vr1q3yer367Gc/q9zcXEnSX//6VzU3N+uDDz5QQUGBNm3apM997nPJHA9IezxzBtLQ\ns88+q9dff12//vWvtXnzZv3973+XJA0MDOhvf/ubduzYoWeffVZFRUXas2ePTpw4oQcffFBPPfWU\n2tvbY+epD4fD2rBhg3784x/r6aef1re//W3dc889yRwNuCTwzBlIQ3/+8591ww03KCsrS5dffrnm\nzJkjSSouLtbatWv185//XIcPH1YgEFBRUZFefvllzZo1S5MnT5YkLViwQAcOHNDbb7+tf/zjH1qx\nYkXs2MFgMCkzAZcS4gykIY/Ho2g0Grv88V9o6+vr0+rVq7VkyRLNmzdPGRkZsT/pd7brR6NRTZs2\nTb/85S8lSaOjozp16tRFnAS4NPGyNpCGrrnmGj3zzDMKh8M6ffq0nn/+eUkf/cWgq6++WjU1NZo5\nc6a6uro0Ojqq8vJyvfLKKzpx4oSi0ah++9vfSpJmzJih06dPq6enR9JHf3pyzZo1SZsLuFTwzBlI\nQ9dff71effVV3XzzzZo8eXLsL7TddNNNuvPOO7VgwQJlZWXpiiuu0JEjRzR58mRt2LBBS5YsUU5O\njmbOnClJys7O1ubNm/XDH/5QH374obxer1pbW5M5GnBJ4K9SAQBgDC9rAwBgDHEGAMAY4gwAgDHE\nGQAAY4gzAADGEGcAAIwhzgAAGEOcAQAw5v8AnjKEg55MeawAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x11527da58>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Histograma de idades\n",
    "sns.set()\n",
    "plt.hist(train_data['Age'], normed=True, alpha=.8)\n",
    "plt.xlabel(\"Idade\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['male' 'female']\n",
      "        Survived\n",
      "Sex             \n",
      "female  0.742038\n",
      "male    0.188908\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x1150ac940>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(train_data['Sex'].unique())\n",
    "print(train_data[['Sex', \"Survived\"]].groupby(['Sex']).mean())\n",
    "plot = sns.barplot(x='Sex', y='Survived', data=train_data, alpha=.8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "           Survived\n",
      "Age_Group          \n",
      "Adulto     0.365059\n",
      "Criança    0.539823\n",
      "Idoso      0.090909\n"
     ]
    },
    {
     "data": {
      "image/png": "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\nYbrvvvtOgwcPrjDtdMcnnD3C2wDLly9X79699dZbb/2mfmbNmvU7VQSTHDx4UA888IAefvhhZWdn\na/Hixdq2bZsWLVpU4XXTpk2T0+kMUZUAzgThfY5bv3692rRpo+TkZL388suSpA0bNmjAgAG6/fbb\nlZeXJ+nUT72DBw/Wd999F2jPmjVLBw4cUFpamsrKyjR27FglJydr0KBBWrFiRc1uFGrU6tWrdfXV\nV+viiy+WdPyGSZMnT1br1q01aNAgDRkyRMuWLVNiYqKOHj2qbdu26Y477lBKSor69eunTz75RJJ0\nww03aMKECXK73Ro5cqSOHTumI0eOaMyYMXK73RowYIAKCwvl8Xh0//3364477lDfvn31yiuvhHDr\nUdMqOz5J0pw5cwLPs5gyZYqk43fiHDx4sIYMGaI777xTHo+H41OQLLvDGn4fS5Ys0aBBg9SuXTs5\nnU5t2rRJjz/+uKZPn662bdvq73//e1D9/PWvf9XChQuVlpamhQsXqkmTJpo6dao8Ho8GDBigLl26\nqEmTJhZvDULhp59+UuvWrStMa9CggcLDw3X06FEtWbJEkjR9+nRJ0ldffaXx48crLi5Oubm5Wrp0\nqTp16qRvv/1W8+fPV4sWLZScnKzPPvtMhYWFatWqlaZNm6Zdu3bpww8/VHh4uPr06aMbbrhBe/bs\n0bBhwzRkyJAa326ERmXHp61bt+rtt9/WokWL5HA4NHr0aK1Zs0YfffSRevXqpZSUFOXn5+vgwYPK\nz8/n+BQEwvscduDAARUUFOiXX35Rdna2PB6PFi5cqJ9//llt27aVdPxmON98880py1b3C8AdO3bo\n2muvlXT8bncxMTH69ttv+eOopVq2bKkvvviiwrRvv/1WH3/8cWA/Olnz5s01c+ZM1atXTyUlJYHb\nHEdFRalFixaSpBYtWujo0aPauXOnrr/+eklS27Zt1bZtW+3Zs0fz58/Xu+++K5fLpfLycou3EOeS\nyo5PO3fu1OWXX67w8HBJ0pVXXqnt27drxIgRev7555WSkqILLrhAl112GcenIHHa/By2fPlyDRw4\nUHPmzNFLL72kxYsXa+3atYqMjNSOHTskSZ999pkkKSIiQvv27dOxY8d08ODBCqfMTzgR6DExMdqw\nYYOk4/eg37Ztmy666KIa2irUtO7du+uDDz4IfMgrKytTVlaWoqKiFBZ26iHgySef1H333afJkycr\n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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x1151148d0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def check_age(age):\n",
    "    \"\"\" Identifica o grupo etário do qual o passageiro faz parte.\n",
    "    \n",
    "    :param age: <float> representando a idade do passageiro\n",
    "    :return: <string> com o grupo etário equivalente\n",
    "    \"\"\"\n",
    "    if age < 18:\n",
    "        return 'Criança'\n",
    "    elif (age >= 18 ) and (age < 65):\n",
    "        return 'Adulto'\n",
    "    else:\n",
    "        return 'Idoso'\n",
    "\n",
    "train_data['Age_Group'] = train_data['Age'].apply(check_age)\n",
    "print(train_data[['Age_Group', 'Survived']].groupby(['Age_Group']).mean())\n",
    "\n",
    "plot = sns.barplot(x='Age_Group', y='Survived', data = train_data, alpha=.8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\nuX+X67uyqPCqbm8LdKT4ez+P9QiIM+NKimI9Agxi2Zm32+1Wa2trZDkUCsnpdHa4rrW1tV3MAQBA\n5yyLd2Zmpurq6iRJXq9XaWlpkXWpqanau3evDh06JL/fr1dffVWjRo2yahQAAOKKLRwOh6144FAo\npIULF+qNN95QOBxWSUmJdu7cKZ/PJ4/Ho5qaGi1ZskThcFg5OTnKy8uzYgwAAOKOZfEGAADW4CIt\nAAAYhngDAGAY4h1nXn/9deXn58d6DMSBQCCguXPnasaMGZo2bZq2bNkS65FguKNHj+qOO+5Qbm6u\nrr32Wr3xxhuxHslYlv2eN3re7373O23YsEGJiYmxHgVxYMOGDRo4cKDKy8t16NAhXXPNNbr88stj\nPRYMVltbK0lau3attm/frvvuu0/Lli2L8VRm4sw7jgwZMkQPPvhgrMdAnJg4caJ++tOfSpLC4bAc\nDkeMJ4LpJkyYoHvuuUeStG/fPp155pkxnshcnHnHkSuvvFLvv/9+rMdAnOjXr5+kY1dL/MlPfqJb\nb701xhMhHjidTv385z/X5s2b9dvf/jbW4xiLM28Andq/f78KCgo0depUTZkyJdbjIE4sWrRIzz//\nvObPny+fzxfrcYxEvAF06KOPPtL111+vuXPnatq0abEeB3Fg/fr1euSRRyRJiYmJstlsstvJUHfw\npwagQw8//LD++9//aunSpcrPz1d+fr7a2tpiPRYMdsUVV2jnzp3Ky8vTj370I915553q06dPrMcy\nEldYAwDAMJx5AwBgGOINAIBhiDcAAIYh3gAAGIZ4AwBgGK6wBvQC77//viZOnKjU1FTZbDYFAgEN\nGjRIpaWlGjx48An3X7dunV5++WWVlZXFYFoAJ8OZN9BLDBo0SE8//bTWr1+vZ555Runp6ZHrTAMw\nC2feQC914YUXqqamRn/9619VVlamcDisc845R/fee2+7+23cuFErV65UW1ubPvnkExUXF2v06NFa\nuXKlnnrqKdntdo0cOVJFRUXatWuXFixYoGAwqISEBJWWluprX/tabJ4gEMc48wZ6oUAgoI0bN2rk\nyJG6/fbbtWjRIlVVVWn48OF66qmnIvcLhUJau3atHn74YW3YsEE33nijHn30UQWDQT3yyCN68skn\ntW7dOtlsNn344YdavXq1rrvuOq1bt075+fnyer0xfJZA/OLMG+glDhw4oKlTp0qS/H6/Ro4cqRkz\nZmjXrl06//zzJUm33XabpGPveUuS3W7XkiVLVFNToz179ujll1+W3W6X0+nUqFGjNG3aNF1++eXK\ny8vTl7/8ZY0dO1ZFRUV68cUXNX78eF155ZWxebJAnCPeQC9x/D3vT9u1a1e75ebmZrW2tkaWW1tb\nlZOTo6lTp2r06NEaPny4/vCHP0iSli5dKq/Xq7q6Ot1www1avHixJk6cqFGjRqm2tlarV6/W1q1b\nVVxcbP2TA3oZ4g30Yuedd54OHjyot956S1//+tf1+9//XpKUkpIiSXr33Xdlt9t18803S5Luvvtu\nHT16VAcPHtSMGTP05JNPatSoUfrggw+0e/duPfbYY5o8ebJyc3OVmpqq0tLSmD03IJ4Rb6AXS0hI\nUHl5uX72s58pEAhoyJAh+vWvf63nn39ekjRixAidf/75mjRpkvr06aPRo0dr3759Ouuss5Sbm6tp\n06YpMTFRZ599tr773e9q9OjRuuuuu7R06VI5HA7Nmzcvxs8QiE98qxgAAIbh0+YAABiGeAMAYBji\nDQCAYYg3AACGId4AABiGeAMAYBjiDQCAYYg3AACG+f8P1NnBF3DX0QAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x114f32630>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot = sns.barplot(x='Pclass', y='Survived', data = train_data, alpha=.8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "          Survived\n",
      "Embarked          \n",
      "C         0.553571\n",
      "Q         0.389610\n",
      "S         0.339009\n"
     ]
    }
   ],
   "source": [
    "print(train_data[['Embarked', 'Survived']].groupby(['Embarked']).mean())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "** Análise preliminar**\n",
    "\n",
    "* A categoria 'Survived' é binária. Ou seja, 1 caso o passageiro tenha sobrevivido; 0 caso tenha falecido.\n",
    "* Cerca de 62% dos passageiros do *dataset* de treino não sobreviveram, ficando próximo à porcentagem do total de 68% (1502 não-sobreviventes / 2224 passageiros).\n",
    "* Cerca de 75% dos passageiros têm até 35 anos.\n",
    "* Menos de 1% dos passageiros têm mais de 65 anos.\n",
    "* Alinhado com a hipótese inicial, mulheres e crianças sobreviveram proporcionalmente mais.\n",
    "* Passageiros que embarcaram pelo Porto 'C' sobreviveram mais que 'Q' e 'S'."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "**Convertendo *Sex* e *Embarked* para valores numéricos**\n",
    "\n",
    "Como as colunas *Sex* e *Embarked* parecem ser bastante informativas, iremos convertê-las para valores numéricos. Assim, o algoritmo de *machine learning* poderá utilizá-las para fazer previsões."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# 'Sex'        male = 0   female = 1\n",
    "train_data.loc[train_data['Sex'] == 'male', 'Sex'] = 0\n",
    "train_data.loc[train_data['Sex'] == 'female', 'Sex'] = 1\n",
    "\n",
    "# 'Embarked'   'S' = 0    'C' = 1    'Q' = 2\n",
    "train_data.loc[train_data['Embarked'] == 'S', 'Embarked'] = 0\n",
    "train_data.loc[train_data['Embarked'] == 'C', 'Embarked'] = 1\n",
    "train_data.loc[train_data['Embarked'] == 'Q', 'Embarked'] = 2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Previsões com *Machine Learning*\n",
    "\n",
    "A Regressão Linear pode ser definida como uma equação que visa estimar o valor esperado de uma variável *y*, dados alguns valores de outras variáveis *x*. A mesma considera que as relações existentes entre as variáveis dependentes e as de entradas são lineares, do tipo *y = mx + b*. Neste caso, o valor de *y* é o que queremos prever, *m* é um coeficiente angular, *x* é o valor de uma dada coluna*, e *b* é uma constante. \n",
    "\n",
    "Apesar desse modelo de previsão ser poderoso, tem alguns pontos negativos, como por exemplo o fato de não conseguir captar as não-lineariedades que possam existir no *dataset* de treino, e também o fato de não apresentar probabilidades de sobrevivência (apresenta apenas o valor binário).\n",
    "\n",
    "**Validação Cruzada**\n",
    "A validação cruzada servirá para treinarmos o algoritmo em um conjunto de dados diferentes daqueles em que faremos a previsão. Este passo é considerado crítico para se evitar um *overfitting* do modelo (ou seja, evitar um *fit* no ruído).\n",
    "\n",
    "Utilizaremos o Método k-fold para a validação cruzada, quando dividiremos os dados em 3 partes (*folds*) e procederemos da seguinte maneira:\n",
    "* Combinar as duas primeiras partes, treinar o modelo e fazer a previsão na terceira parte;\n",
    "* Combinar a primeira e terceira parte, treinar o modelo e fazer a previsão na segunda; e\n",
    "* Combinar a segunda e a terceira parte, treinar o modelo e fazer a previsão na primeira.\n",
    "\n",
    "Faremos uso da biblioteca (excelente!) *scikit-learn*, que facilita muito o fluxo de trabalho."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "len(predictions) : 3\n",
      "5 primeiras entradas da variável predictions:  [ 0.09445602  0.96588768  0.54180569  0.92949341  0.03164796]\n"
     ]
    }
   ],
   "source": [
    "# Colunas que serão utilizadas para fazer a previsão\n",
    "predictors = ['Sex', 'Age', 'Pclass', 'Embarked']\n",
    "\n",
    "# Inicialização da classe do algoritmo\n",
    "algorithm = LinearRegression()\n",
    "folds = KFold(train_data.shape[0], n_folds=3, random_state=0)\n",
    "# Visualizar a estrutura da variável folds\n",
    "# for train_index, test_index in folds:\n",
    "#     print(\"Train: \", train_index, \"Test: \", test_index)\n",
    "\n",
    "predictions = []\n",
    "for train_index, test_index in folds:\n",
    "    train_predictors = train_data[predictors].iloc[train_index, :]\n",
    "    train_target = train_data['Survived'].iloc[train_index]\n",
    "    # Treinando o algoritmo utilizando os predictos e o target (survived?)\n",
    "    algorithm.fit(train_predictors, train_target)\n",
    "    # Fazendo previsões\n",
    "    test_predictions = algorithm.predict(train_data[predictors].iloc[test_index, :])\n",
    "    predictions.append(test_predictions)\n",
    "    \n",
    "# Como criamos 3 conjuntos de teste, o output é uma list de len() = 3. Para cara item da lista, \n",
    "# é armazenado um valor para cada passageiro do conjunto de teste\n",
    "print(\"len(predictions) :\", len(predictions))\n",
    "print(\"5 primeiras entradas da variável predictions: \", predictions[0][0:5])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Como os únicos valores para a coluna 'Survived' são 0 ou 1, iremos transformar as previsões de cada passageiro seguindo o critério:\n",
    "* ( > 0.5 ) = 1\n",
    "* ( <= 0.5 ) = 0\n",
    "\n",
    "Com isso, basta calcular quantos acertos o nosso algoritmo teve e calcular a acurácia do modelo.\n",
    "\n",
    "Apesar de 78% de acertos estar longe do ideal, a aplicação de métodos de Regressão Linear foi capaz de capturar e transmitir para a equação algumas correlações importantes. Como próximo passo, vamos processar a variável de teste do modelo (idêntico aos passos que utilizamos no *dataset* de treino), enviar para o Kaggle e verificar a nossa taxa de acerto."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Acurácia: 78.34%\n"
     ]
    }
   ],
   "source": [
    "predictions = np.concatenate(predictions)\n",
    "# Transformando as previsões em valores binários\n",
    "predictions[predictions > 0.5] = 1\n",
    "predictions[predictions <= 0.5] = 0\n",
    "\n",
    "# Calcular a acurácia da previsão do algoritmo\n",
    "accuracy = sum(predictions[predictions == train_data['Survived']]) / predictions.shape[0]\n",
    "print('Acurácia: {0:.2f}%'.format(100*accuracy))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Gerando um arquivo para enviar ao Kaggle\n",
    "\n",
    "Seguiremos os mesmo passos de limpeza de dados e conversões numéricas para algumas variáveis.\n",
    "\n",
    "Ao final, geraremos um arquivo .csv para submeter ao Kaggle, e verificar então como nosso algoritmo foi no *dataset* de teste."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Preenchendo os valores NaN com a mediana da idade\n",
    "test_data.loc[test_data['Age'].isnull(), 'Age'] = test_data['Age'].median()\n",
    "\n",
    "# Conversão dos valores da coluna 'Sex'\n",
    "# 'Sex'        male = 0   female = 1\n",
    "test_data.loc[test_data['Sex'] == 'male', 'Sex'] = 0\n",
    "test_data.loc[test_data['Sex'] == 'female', 'Sex'] = 1\n",
    "\n",
    "# Conversão dos valores da coluna 'Embarked'\n",
    "# 'Embarked'   'S' = 0    'C' = 1    'Q' = 2\n",
    "test_data.loc[test_data['Embarked'] == 'S', 'Embarked'] = 0\n",
    "test_data.loc[test_data['Embarked'] == 'C', 'Embarked'] = 1\n",
    "test_data.loc[test_data['Embarked'] == 'Q', 'Embarked'] = 2\n",
    "\n",
    "# Agora, treinaremos o algoritmo com o dataset de treino completo (sem dividir com KFold)\n",
    "algorithm = LinearRegression()\n",
    "algorithm.fit(train_data[predictors], train_data['Survived'])\n",
    "# Previsões em cima do dataset de teste\n",
    "predictions = algorithm.predict(test_data[predictors])\n",
    "\n",
    "# Coneversão do resultado para valores binários\n",
    "predictions[predictions > 0.5] = int(1)\n",
    "predictions[predictions <= 0.5] = int(0)\n",
    "\n",
    "# Gerando o arquivo csv para envio \n",
    "# São necessárias apenas duas colunas no arquivo: 'PassengerId' e 'Survived'\n",
    "kaggle = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})\n",
    "# Convertendo a coluna de float para int (caso contrário, o site dará erro)\n",
    "kaggle['Survived'] = kaggle['Survived'].astype(dtype='int64')\n",
    "\n",
    "kaggle.to_csv('kaggle.csv', index=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Resultado final\n",
    "\n",
    "Após o envio para o Kaggle, foi computado um *Score* de 75% de acertos, alinhado com a acurácia que o modelo havia apresentado no *dataset* de treino.\n",
    "\n",
    "<a href=\"https://ibb.co/iyVhBF\"><img src=\"https://preview.ibb.co/hTdr5a/Screenshot_2017_03_07_01_15_52.png\" alt=\"Screenshot 2017 03 07 01 15 52\" border=\"0\" /></a>\n",
    "\n",
    "Como foi dito no começo, a Regressão Linear apresenta bons resultados e é simples de implementar. Há, no entanto, outros algoritmos melhores, como o *Random Forest*, que tem um melhor potencial para elevar a taxa de acerto.\n",
    "\n",
    "Mas como o carater desta análise era apenas epxloratório, creio que serviu para o propósito de aprendizado (meu, principalmente). Quaisquer dúvidas ou sugestões serão muito bem vindas. Obrigado!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python [conda root]",
   "language": "python",
   "name": "conda-root-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
