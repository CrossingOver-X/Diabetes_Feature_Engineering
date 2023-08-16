#Feature Engineering

#İş Problemi

#Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin  edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir. Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir

#Veri Seti Hikayesi

#Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır. ABD'dekiArizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir.Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.


#Görev 1 : Keşifçi Veri Analizi

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

#Görev 1 : Keşifçi Veri Analizi

df = pd.read_csv("datasets/diabetes.csv")


#Pregnancies: Hamilelik sayısı
#Glucose: Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
#Blood Pressure: Kan Basıncı (Küçük tansiyon) (mm Hg)
#SkinThickness: Cilt Kalınlığı
#Insulin: 2 saatlik serum insülini (mu U/ml)
#DiabetesPedigreeFunction:diyabetin aile geçmişiyle ilişkisini değerlendirmek için kullanılan bir matematiksel formüldür. Bu fonksiyon, bir kişinin aile üyelerindeki diyabetin şiddetini ve yaygınlığını göz önünde bulundurarak bir tahmin yapmaya çalışır.
#DPF'nin yüksek olması, kişinin aile geçmişinde diyabetin daha yaygın ve şiddetli olduğunu gösterir ve bu nedenle kişinin diyabet riskinin artabileceğini düşündürebilir.
#BMI: Vücut kitle endeksi
#Age: Yaş (yıl)
#Outcome: Hastalığa sahip (1) ya da değil (0)

#Adım1:Genel resmi inceleyiniz.
df.head(10)
df.info
df.dtypes
df.shape
df.isnull().sum()
df.describe().T

#Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin ve BMI değişkenlerinin min değerlerinin 0 olduğunu görüyoruz.Pregnancies d dışındaki diğer değişkenlerin hiçbiri 0 olamaz.


#Adım2:  Numerik ve kategorik değişkenleri yakalayınız

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
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

#Observations: 768
#Variables: 9
#cat_cols: 1
#num_cols: 8
#cat_but_car: 0
#num_but_cat: 1

#Adım3: Numerik ve kategorik değişkenlerin analizini yapınız.

cat_cols #: Outcome
num_cols #: ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

#Numerik değişken analizi
def num_analysis(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

num_analysis(df, num_cols)


#Kategorik değişken analizi

def cat_analysis(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))


cat_analysis(df, "Outcome") #:   Outcome  Ratio
                             #0      500 65.104
                             #1      268 34.896

#Adım 4: Hedef değişken(outcome) analizi yapınız. Kategorik değişkenlere(outcome) göre hedef değişkenin(outcome) ortalaması, hedef değişkene(outcome) göre numerik('Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age') değişkenlerin ortalaması)

def target_analysis_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_analysis_num(df, "Outcome", col)


#Adım 5: Aykırı gözlem analizi yapınız.

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    if low_limit < dataframe[col_name].min():
        low_limit = dataframe[col_name].min()
    return low_limit, up_limit

outlier_thresholds(df, "Pregnancies")
outlier_thresholds(df, "Glucose")

low, up = outlier_thresholds(df, "Pregnancies")

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
check_outlier(df, "Pregnancies")

#Adım 6: Eksik gözlem analizi yapınız.
#Pregnancies ve Outcome dışında veriler 0 olamaz.

zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
zero_columns


#Adım 7: Korelasyon analizi yapınız

df.corr()


#Görev 2 :

#Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb. değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0 olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik değerlere işlemleri uygulayabilirsiniz.

for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

df.isnull().sum()

df.head(10)

for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()

df[zero_columns].describe().T

#Adım 2: Yeni değişkenler oluşturunuz.

df.loc[(df["BMI"] < 16), "BMI_Level"] ="overweak"
df.loc[(df["BMI"] >= 16) & (df["BMI"] < 18.5) , "BMI_Level"] ="weak"
df.loc[(df["BMI"] >= 18.5) & (df["BMI"] < 25) , "BMI_Level"] ="normal"
df.loc[(df["BMI"] >= 25) & (df["BMI"] < 30) , "BMI_Level"] ="overweight"
df.loc[(df["BMI"] >= 30) & (df["BMI"] < 35) , "BMI_Level"] ="1st_Obese"
df.loc[(df["BMI"] >= 35) & (df["BMI"] < 45) , "BMI_Level"] ="2nd_Obese"
df.loc[(df["BMI"] >= 45), "BMI_Level"] ="3rd_Obese"


def categorize_age(age):
    if age >= 65:
        return "Old"
    elif age >= 35:
        return "Middle"
    else:
        return "Young"


df["Age_Category"] = df["Age"].apply(categorize_age)


df.head(5)


#Adım 3: Encoding işlemlerini gerçekleştiriniz.
df = pd.get_dummies(df,drop_first=True)

df.head()

#Adım 4: Numerik değişkenler için standartlaştırma yapınız.
num_cols
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()


#Adım 5: Model oluşturunuz

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)







