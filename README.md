# Diabetes Hastalığı Tahmini ve Özellik Mühendisliği

Bu proje, Pima Indian kadınları üzerinde yapılan bir diyabet araştırması veri setini kullanarak, kişilerin diyabet hastası olup olmadığını tahmin edebilecek bir makine öğrenmesi modeli oluşturmayı amaçlamaktadır. Ayrıca veri seti üzerinde eksik veri analizi, aykırı değer analizi, özellik mühendisliği ve model oluşturma gibi adımlar da gerçekleştirilmektedir.

# Veri Seti ve İş Problemi

Veri seti, ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan bir diyabet araştırmasının parçasıdır. Arizona eyaletinde yaşayan 21 yaş ve üzerindeki Pima Indian kadınları üzerinde gerçekleştirilen bu araştırma, kişilerin diyabet durumlarını belirlemeyi amaçlamaktadır.

# Analiz Adımları
# Keşifçi Veri Analizi (EDA):

Veri setinin genel görünümü incelenir.
Veri tipleri, eksik değerler, aykırı değerler ve istatistiksel özetlemeler kontrol edilir.
Numerik ve kategorik değişkenler ayrıştırılır.
Hedef değişkenin dağılımı ve sınıf oranlarına bakılır.
Numerik değişkenlerin istatistiksel analizi yapılır.

# Veri Ön İşleme ve Özellik Mühendisliği:

Eksik değerler, hamilelik ve sonuç dışındaki değişkenlerde bulunan 0 değerleri ilgili istatistiksel ölçümlerle doldurulur.
Veri setine yeni özellikler eklenir:
BMI seviyeleri kategorisi oluşturulur.
Yaş kategorisi oluşturulur.
Kategorik değişkenler için "One-Hot Encoding" uygulanır.
Numerik değişkenler standartlaştırılır.

# Model Oluşturma ve Değerlendirme:

Veri seti bağımsız değişkenler (X) ve hedef değişken (y) olarak ayrıştırılır.
Veri seti eğitim ve test kümelerine ayrılır.
RandomForestClassifier kullanılarak bir makine öğrenmesi modeli oluşturulur.
Oluşturulan modelin performansı accuracy metriği kullanılarak değerlendirilir.

# Sonuçlar

Bu proje, Pima Indian kadınları üzerinde gerçekleştirilen diyabet araştırması veri seti üzerinde bir dizi analiz ve işlemi içermektedir. Eksik veri analizi, aykırı değer analizi, özellik mühendisliği ve makine öğrenmesi modeli oluşturma adımlarıyla veri seti hazırlanmış ve model performansı değerlendirilmiştir.

Bu projenin detaylı açıklamaları ve kodun tamamı "Diabetes_FeatureEngineering.py" dosyasında bulunmaktadır.
