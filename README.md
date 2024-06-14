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

Projenin çalışabilmesi için Drive(Sözlük) klasöründeki dosyaları kendi Drive'ınıza Sözlük adı altında bir klasör oluşturarak yüklemeniz gerekir.
Drive'ı projeye entegre etme:
```
from google.colab import drive
drive.mount('/content/drive')
```

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

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
