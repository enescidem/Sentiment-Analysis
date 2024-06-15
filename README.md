# Duygu Analizi

İnternet üzerinden yapılan yorumların duygu analizi yapılması amacıyla gerçekleştirilecek proje, sosyal medya platformlarında paylaşılan metinleri veya ürün yorumlarını anlama ve yorumlama hedefine yönelik bir çalışma önermektedir. Bu proje, internette kullanıcılar tarafından paylaşılan metin verilerini toplamayı, işlemeyi ve bu metinlerdeki duygu durumlarını belirlemeyi amaçlamaktadır.

Bu sayede, kullanıcıların paylaşımlarının genel duygu durumunu değerlendirebilecek, örneğin olumlu, olumsuz veya nötr olarak sınıflandırabilecek bir sistem geliştirilmesi planlanmaktadır. Proje, işletmelerin müşteri memnuniyetini artırmak, ürün geliştirmek ve pazarlama stratejilerini optimize etmek için değerli bir bilgi kaynağı sağlayacaktır.

# Sentiment Analysis

The project, which will be realized with the aim of analyzing the sentiment of online comments, proposes a study aimed at understanding and interpreting texts or product reviews shared on social media platforms. This project aims to collect and process text data shared by users on the Internet and identify the sentiment of these texts.

In this way, it is planned to develop a system that can evaluate the overall sentiment of users' posts, for example, classifying them as positive, negative or neutral. The project will provide a valuable source of information for businesses to improve customer satisfaction, develop products and optimize their marketing strategies.

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

Örnek Bir Hash Tablosu:
https://github.com/enescidem/Sentiment-Analysis/blob/main/Drive(S%C3%B6zl%C3%BCk)/hash_table.csv
![HashTable](https://github.com/enescidem/Sentiment-Analysis/assets/92892867/9b62d762-949a-4191-b035-b030510c50ab)

---
### Hash tablosunu Drive'a kaydetme ve geri yükleme fonksiyonları:
```
def save_to_csv(frekans_tablosu, dosya_adı):
def load_from_csv(dosya_adı, tablo_boyutu):
```
---


## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
