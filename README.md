# Duygu Analizi

İnternet üzerinden yapılan yorumların duygu analizi yapılması amacıyla gerçekleştirilecek proje, sosyal medya platformlarında paylaşılan metinleri veya ürün yorumlarını anlama ve yorumlama hedefine yönelik bir çalışma önermektedir. Bu proje, internette kullanıcılar tarafından paylaşılan metin verilerini toplamayı, işlemeyi ve bu metinlerdeki duygu durumlarını belirlemeyi amaçlamaktadır.

Bu sayede, kullanıcıların paylaşımlarının genel duygu durumunu değerlendirebilecek, örneğin olumlu, olumsuz veya nötr olarak sınıflandırabilecek bir sistem geliştirilmesi planlanmaktadır. Proje, işletmelerin müşteri memnuniyetini artırmak, ürün geliştirmek ve pazarlama stratejilerini optimize etmek için değerli bir bilgi kaynağı sağlayacaktır.

# Sentiment Analysis

The project, which will be realized with the aim of analyzing the sentiment of online comments, proposes a study aimed at understanding and interpreting texts or product reviews shared on social media platforms. This project aims to collect and process text data shared by users on the Internet and identify the sentiment of these texts.

In this way, it is planned to develop a system that can evaluate the overall sentiment of users' posts, for example, classifying them as positive, negative or neutral. The project will provide a valuable source of information for businesses to improve customer satisfaction, develop products and optimize their marketing strategies.

### Genel Hatlarıyla Algoritma Adımları:
**1-** Gelen yorumlar ilk olarak get_normalizasyon fonksiyonuna girerek normalize edilir.

**2-** Normalize edilmiş yorumlar get_kelimeanaliz ile sıfat, zarf ve fiil olarak ayrılır.

**3-** Ayrılan kelimeler hash tablosuna atılır sayılır ve kaydedilir.

**3.5-** Eğer hash tablosuna eklenirke kelime pozitif ve negatif txt dosyalarında yok ise kullanıcıya kelimenin hangi txt ye ait olduğu sorulur buna göre sıralanır.

**4-** Hash tablosundaki değerler -1 ile 1 aralığına alınır.

**5-** Daha sonra yorumun gerekli kelimeleri için verilen değerler toplanarak cümlenin skoru belirlenir.

**6-** Cümlenin sonucu - ise olumsuz, + ise olumlu bir yorum olduğunu kaydediyoruz.

### Ön Koşullar (Prerequisites)

Projeyi, herhangi bir Jupyter Notebook platformunda çalıştırmak mümkündür.

```
Google Colab -> colab.google
Jupyter Notebook -> jupyter.org
```

### Gerekli Kütüphaneler

Projenin Çalışması için gerekli kütüphane zemberek kütüphanesidir:

```
!pip install zemberek-python
```

Gerekli importlar:

```
import re
import csv
import string
import itertools
import numpy as np
import pandas as pd
import random

from zemberek import (
    TurkishSpellChecker,
    TurkishSentenceNormalizer,
    TurkishMorphology,
)
```

Zemnerek için gerekli değişken tanımlamaları:

```
morphology = TurkishMorphology.create_with_defaults()
normalizer = TurkishSentenceNormalizer(morphology)
```

Projenin çalışabilmesi için [Drive(Sözlük)](https://github.com/enescidem/Sentiment-Analysis/tree/main/Drive(S%C3%B6zl%C3%BCk)) klasöründeki dosyaları kendi Drive'ınıza Sözlük adı altında bir klasör oluşturarak yüklemeniz gerekir.
Drive'ı projeye entegre etme:
```
from google.colab import drive
drive.mount('/content/drive')
```

## Fonksiyonlar Ve İşlevleri

### Drive içerisindeki pozitif.txt ve negatif.txt dosyalarında kelime ekleme, silme ve arama işlemlerini yapmak için aşağıdaki fonksiyonları kullanabilirsiniz:
```
def kelime_ekle(dosya_adı, kelime):
def kelime_sil(dosya_adı, silinecek_kelime):
def kelime_arama(dosya_adı, aranan_kelime):
```
---

### Normalizasyon fonksiyonu:
```
def get_normalizasyon(example):
```
![image](https://github.com/enescidem/Sentiment-Analysis/assets/92892867/c0aaac0f-63ab-4ca3-a40b-130913981712)

---
### Kelime Analiz fonksiyonu:
```
def get_kelimeanaliz(example):
```
![image](https://github.com/enescidem/Sentiment-Analysis/assets/92892867/4c3b2cac-57f0-4014-b9ca-f06265643dcd)

---
### Sorgulanan kelimenin pozitif veya negatif txt dosyalarında olup olmadığını kontrol eden bir sistem:
```
def pozitif_mi(kelime):
def negatif_mi(kelime):
```
---
### Hash tablosunun fonksiyonları:
```
def hash_func(kelime, boyut):
def linear_probe(index, step, boyut):
def kelime_cikar(metin):
def kelime_agirligi(metin_listesi, tablo_boyutu=5000):
def ara_tablo(kelime, frekans_tablosu, tablo_boyutu=5000):
def print_frekans_tablosu(frekans_tablosu):
def en_yuksek_frekansli_kelime(frekans_tablosu):
def en_dusuk_frekansli_kelime(frekans_tablosu):
def katsayı_hesapla(deger, min_deger, max_deger):
def normalize_frekans_tablosu(tablo):
```
1. **hash_func(kelime, boyut)**: Bir kelimenin ASCII değerlerinin toplamını verilen boyuta göre mod alarak bir hash değeri döndürür.
2. **linear_probe(index, step, boyut)**: Çakışma durumunda lineer arama ile yeni indeks hesaplar.
3. **kelime_cikar(metin)**: Metin içindeki kelimeleri çıkarır ve noktalama işaretleri ve rakamları kaldırarak bir liste döndürür.
4. **kelime_agirligi(metin_listesi, tablo_boyutu)**: Metin listesindeki kelimeleri pozitif veya negatif olarak sınıflandırıp ağırlıklarını hesaplayarak hash tablosuna ekler.
5. **ara_tablo(kelime, frekans_tablosu, tablo_boyutu)**: Bir kelimenin frekans tablosunda kaç kez geçtiğini bulur.
6. **print_frekans_tablosu(frekans_tablosu)**: Frekans tablosundaki kelimeleri ve frekanslarını yazdırır.
7. **en_yuksek_frekansli_kelime(frekans_tablosu)**: Frekans tablosundaki en yüksek frekansa sahip kelimeyi bulur.
8. **en_dusuk_frekansli_kelime(frekans_tablosu)**: Frekans tablosundaki en düşük frekansa sahip kelimeyi bulur.
9. **katsayı_hesapla(deger, min_deger, max_deger)**: Bir değeri min ve max değerlere göre normalize eder.
10. **normalize_frekans_tablosu(tablo)**: Frekans tablosundaki tüm frekansları normalize eder.

Örnek Bir Hash Tablosu([Daha Ayrıntılı](https://github.com/enescidem/Sentiment-Analysis/blob/main/Drive(S%C3%B6zl%C3%BCk)/hash_table.csv)):
![HashTable](https://github.com/enescidem/Sentiment-Analysis/assets/92892867/9b62d762-949a-4191-b035-b030510c50ab)

---
### Hash tablosunu Drive'a kaydetme ve geri yükleme fonksiyonları:
```
def save_to_csv(frekans_tablosu, dosya_adı):
def load_from_csv(dosya_adı, tablo_boyutu):
```
---


## Bir Veri Seti Verdiğimizde([verilen veri seti](https://github.com/enescidem/Sentiment-Analysis/blob/main/Drive(Sözlük)/veri_seti.csv)) Algoritmanın Verdiği Sonuçlar:
![image](https://github.com/enescidem/Sentiment-Analysis/assets/92892867/48a2d958-493b-4419-a39e-6601b54676df)
Bu tabloda görülen sütunları teker teker açıklayalım:

**Görüş:** Bu sütun veri setinde bulunan yorumların normalizasyondan geçirilmiş halidir.

**label:** Bu sütun veri setindeki yorumların gerçek(doğru) sonuçlarını gösterir.

**Kelimeler:** Bu sütun normalizasyondan geçmiş cümlelerden bulunan sıfat ve zarf kelimeleridir.

**Toplam Ağırlık:** Bu sütun yorumun tüm fonksiyonlardan geçtikten sonraki aldığı sonuçtur. - değeri olumsuz + değeri olumludur.

**binary_sonuçlar:** Bu sütun Toplam Ağırlıkta bulduğumuz değerleri 0 ve 1 olmaz üzere ayırıyor. -1 de veriyoruz bu değer Toplam Ağırlığı 0 olan sonuçlar için.

**Doğruluk:** Bu sütun label ile binary_sonuçlar sütununu karşılaştırıp eşit olan değerlere True farklı olan değerlere ise False değeri yazdırıyor.


Burada bulunan sonuçlarda fiiler yoktur. Fiiller için oluçturulan fonksiyonlar tam olarak doğru çalışmadığı için es geçilmiştir.

### Tabloya göre modelin doğruluk oranı şu şekildedir: ###

![image](https://github.com/enescidem/Sentiment-Analysis/assets/92892867/8d495c25-eb6a-4f84-aba5-68cc88c4e027)

### Tek Bir Cümlenin Adım Adım Analizi: ###

![image](https://github.com/enescidem/Sentiment-Analysis/assets/92892867/36850038-576c-412b-8be1-c85c616a563c)
