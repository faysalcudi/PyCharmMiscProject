


#########  ANA PROBLEM ################
#Kırmızı şarapların kimyasal özellikleri kullanılarak kalite puanı tahmin edilebilir mi?
#########  1.PROBLEM ################
#Hangi özellikler şarap kalitesini en çok etkiliyor?
#########  2. PROBLEM ################
#Kaliteyi tahmin eden model kurulabilir mi?
#########  ANA PROBLEM ################
#Uçucu asitlik (volatile acidity) yüksek olan şaraplar genelde daha düşük kaliteye mi sahip?
#(Ortalama karşılaştırması ve scatter plot ile gösterilir.)
#########  4.problem ################
#Şarap kalitesini tahmin eden en iyi makine öğrenmesi modeli hangisi?
#########  5.problem ################
#Hangi üç özellik kaliteyi en iyi tahmin ediyor?
#(Makine öğrenmesi modeli ile feature importance çıkarılır.)

###
"""What are the correlations between all chemical properties?
sight: Strong correlations exist between density-alcohol (-0.50), 
fixed acidity-pH (-0.68), and citric acid-fixed acidity (0.67), revealing chemical relationships."""
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

"""What's the relationship between density and alcohol content?
Insight: Strong negative correlation between alcohol and density;
 higher alcohol wines have lower density, with better quality wines clustering together."""

"""pH seviyeleri farklı kaliteli şaraplar arasında nasıl dağılım gösterir?"""

"""How does citric acid content vary with wine quality?"""


import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.simplefilter(action='ignore', category=Warning)

# GÖREV 1: KEŞİFCİ VERİ ANALİZİ
           # Adım 1: Genel resmi inceleyiniz.
           # Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
           # Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
           # Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)
           # Adım 5: Aykırı gözlem analizi yapınız.
           # Adım 6: Eksik gözlem analizi yapınız.
           # Adım 7: Korelasyon analizi yapınız.


_df= pd.read_csv("/Users/macpro/PyCharmMiscProject/Miuul_Proje/winequality-red.csv")
df=_df.copy()
df.head(5)
#Orijinal Değişken Adı	Türkçe Karşılığı
#fixed acidity	Sabit asitlik
#volatile acidity	Uçucu asitlik
#citric acid	Sitrik asidi
#residual sugar	Kalıntı şeker
#chlorides	Klorür (tuz oranı)
#free sulfur dioxide	Serbest kükürt dioksit
#total sulfur dioxide	Toplam kükürt dioksit
#density	Yoğunluk
#pH	pH değeri
#sulphates	Sülfatlar
#alcohol	Alkol oranı
#quality	Şarap kalitesi


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA ########################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)



def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols
num_cols
cat_but_car
df.info()


##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

#1 ve 2 cok dusuk kalite oldugu için veri setinde yok
#9 ve 10 ise cok yuksek kalite oldugu için veri setinde yok
"""
         quality      Ratio
quality                    
5            681  42.589118
6            638  39.899937
7            199  12.445278
4             53   3.314572
8             18   1.125704
3             10   0.625391
"""



##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################
"""
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

# x bizim degişkenlerimiz
# y ise bu degişkenlerden kaç tane oldugu gosteriyor
for col in num_cols:
    num_summary(df, col, plot=True)
"""


def num_summary_all(dataframe, numerical_cols):
    # Tüm sayısal değişkenlerin describe tablolarını yan yana yazdırmak için
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    summary_df = pd.DataFrame()

    for col in numerical_cols:
        desc = dataframe[col].describe(quantiles).T
        desc.name = col
        summary_df = pd.concat([summary_df, desc], axis=1)

    print(summary_df.T)  # Transpose ile değişkenler satır olur, daha okunur

    # Grafikler için subplot ayarı
    n = len(numerical_cols)
    n_cols = 3  # bir satırda 3 grafik
    n_rows = (n // n_cols) + (1 if n % n_cols != 0 else 0)

    plt.figure(figsize=(n_cols * 5, n_rows * 4))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        dataframe[col].hist(bins=20, edgecolor='black')
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.tight_layout()
    plt.show()


# Çağırmak için:
num_summary_all(df, num_cols)

##################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################
# bu koddaki amacımız her bir sayısal degişkenin kaliteye göre ortalamasını bulmaktır

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "quality", col)

#volatile acidity (uçucu asitlik) kalite arttıkça azalma eğiliminde (örneğin kalite 3’te 0.88, kalite 7’de 0.40 civarı). Bu iyi, çünkü uçucu asitlik genelde negatif kalite göstergesidir.
#alcohol (alkol oranı) kalite arttıkça yükseliyor (kalite 3: 9.95, kalite 8: 12.09).
#citric acid (sitrik asit) ve diğer bazı bileşenler kalite ile beraber artıyor.
#Genel çıkarım:
#Şarap kalitesi yükseldikçe bazı kimyasal özelliklerde belirgin değişiklikler oluyor.
#Bu tip analiz, hangi değişkenlerin kaliteyle ilişkili olduğunu anlamak için faydalı.


##################################
# KORELASYON
##################################


df.corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

# TotalChargers'in aylık ücretler ve tenure ile yüksek korelasyonlu olduğu görülmekte

df.corrwith(df["quality"]).sort_values(ascending=False)



##################################
# GÖREV 2: FEATURE ENGINEERING
##################################

##################################
# EKSİK DEĞER ANALİZİ
##################################
df.isnull().sum()

##eksik degerimiz bulunmamaktadır


##################################
# AYKIRI DEĞER ANALİZİ
##################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# Analiz fonksiyonu
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

""" aykırı degerlere ulaştık
fixed acidity False
volatile acidity False
citric acid False
residual sugar True
chlorides True
free sulfur dioxide False
total sulfur dioxide True
density False
pH False
sulphates True
alcohol False
"""

for col in num_cols:  # num_cols = sayısal değişkenlerin listesi
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)


####################################################################
#What's the relationship between density and alcohol content?
# Create a scatter plot with regression line
plt.figure(figsize=(12, 8))
plt.scatter(df['alcohol'], df['density'], c=df['quality'], cmap='RdYlBu_r',
           alpha=0.7, s=60, edgecolors='black', linewidth=0.5)

# Add regression line
z = np.polyfit(df['alcohol'], df['density'], 1)
p = np.poly1d(z)
plt.plot(df['alcohol'].sort_values(), p(df['alcohol'].sort_values()), "red", linewidth=2, linestyle='--')

plt.colorbar(label='Quality Rating')
plt.title('Relationship Between Alcohol Content and Wine Density', fontsize=16, fontweight='bold')
plt.xlabel('Alcohol Content (%)', fontsize=12)
plt.ylabel('Density (g/cm³)', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


########linkcode###########
#How does alcohol content relate to wine quality?
##########
plt.figure(figsize=(12, 8))
sns.violinplot(data=df, x='quality', y='alcohol', palette='Reds', inner='box')
plt.title('Alcohol Content Distribution Across Wine Quality Ratings', fontsize=16, fontweight='bold')
plt.xlabel('Quality Rating', fontsize=12)
plt.ylabel('Alcohol Content (%)', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

#linkcode
#How do pH levels distribute across different quality wines?
fig, axes = plt.subplots(len(df['quality'].unique()), 1, figsize=(12, 12), sharex=True)
colors = ['#8B0000', '#A0522D', '#B22222', '#CD5C5C', '#DC143C', '#F08080']

for i, quality in enumerate(sorted(df['quality'].unique())):
    data = df[df['quality'] == quality]['pH']
    axes[i].hist(data, bins=20, alpha=0.7, color=colors[i], edgecolor='black', linewidth=0.8)
    axes[i].set_ylabel(f'Quality {quality}', rotation=0, labelpad=20)
    axes[i].grid(alpha=0.3)

plt.suptitle('pH Distribution Across Wine Quality Levels', fontsize=16, fontweight='bold', y=0.98)
plt.xlabel('pH Level', fontsize=12)
plt.tight_layout()
plt.show()


#What's the multivariate relationship between key quality predictors?¶
key_vars = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid', 'quality']
plt.figure(figsize=(14, 10))
sns.pairplot(df[key_vars], hue='quality', palette='Reds', diag_kind='hist',
            plot_kws={'alpha': 0.7}, diag_kws={'alpha': 0.7})
plt.suptitle('Multivariate Analysis of Key Wine Quality Predictors', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

#How do multiple chemical properties cluster wines into quality groups?¶
# Create a radar chart for average properties by quality
#bu bolume 5 tane aykırı degeri true olan bolumu koyalım bu kaliteyi ne boyutta degiştiriyomuş
from math import pi

# Calculate means by quality
quality_means = df.groupby('quality').mean()
properties = ['fixed acidity', 'volatile acidity', 'citric acid', 'alcohol', 'sulphates']

# Normalize data for radar chart
quality_means_norm = quality_means[properties].div(quality_means[properties].max())

fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))
colors = ['#8B0000', '#A0522D', '#B22222', '#CD5C5C', '#DC143C', '#F08080']

angles = [n / float(len(properties)) * 2 * pi for n in range(len(properties))]
angles += angles[:1]

for i, quality in enumerate(quality_means_norm.index):
    values = quality_means_norm.loc[quality].values.tolist()
    values += values[:1]

    ax.plot(angles, values, 'o-', linewidth=2, label=f'Quality {quality}', color=colors[i])
    ax.fill(angles, values, alpha=0.25, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(properties)
ax.set_title('Wine Quality Profile by Chemical Properties', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
plt.tight_layout()
plt.show()


#################################################
####### BASE MODEL  #############

def label_encoder(dataframe, col_name):
    le = LabelEncoder()
    dataframe[col_name] = le.fit_transform(dataframe[col_name])
    return dataframe, le

label_encoder(df,'quality')

y = df["quality"]
X = df.drop(["quality"], axis=1)

models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

#merhabbabababadgfdgdf