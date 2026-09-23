# Spaceship Titanic

Kaggle competition: [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)

**Soru:** uzay-zaman anomalisinde hangi yolcular başka bir boyuta taşındı ve bir model bunun nedenini öğrenebilir mi?

| Notebook | Dil |
|---|---|
| [`spaceship_titanic.ipynb`](spaceship_titanic.ipynb) | English |
| [`spaceship_titanic_tr.ipynb`](spaceship_titanic_tr.ipynb) | Türkçe |

## Hikâye

| Bölüm | Soru |
|---|---|
| 1 | Veride ne var, eksikler nerede? |
| 2 | Kim taşındı? |
| 3 | Veri modele nasıl hazırlanır? |
| 4 | Bulgular hangi özelliklere dönüşür? |
| 5 | Hangi model daha iyi? |
| 6 | Ayar ve birleştirme ne katıyor? |
| 7 | Model nerede yanılıyor? |

## Ana bulgular

- **Uyku ve harcama en güçlü ipucu:** kapsülünde uyuyanların %82'si taşındı, gemide harcama yapanların %30'u.
- **Harcamanın türü önemli:** oda servisi, spa ve VR güvertesinde 3000'den fazla harcayanlarda oran %5'e düşüyor. Yemek ve alışverişte ilişki U biçiminde.
- **Konum belirleyici:** B ve C güverteleri %68–73, sağ taraf %56, sol taraf %45. Alt güvertelerde sağ taraftaki 600–1199 numaralı kabinlerde oran %71.
- **Gezegen güverteyle iç içe:** Europa %66, Earth %42. Uyuyan Europa'lıların %99'u, uyuyan Earth'lülerin %66'sı taşındı.
- **Küçük çocuklar farklı:** 0–4 yaş %77, yetişkinler %46–50.
- **Grup kaderi paylaşılmıyor:** yalnızlar %45, gruptakiler %57; ama grupların yalnızca %43.6'sında tüm üyeler aynı sonucu yaşadı (rastgele karıştırmada %38.4).
- **Eksik veriler rastgele:** eksik olan ve olmayan yolcuların taşınma oranları aynı.

## Yöntem

- **Kurallı eksik doldurma:** 12 kural, doğrulukları %73–100 (uyuyanın harcaması 0, soyadından gezegen, gruptan kabin). Kalan eksikler sızıntısız `Pipeline` içinde, yalnızca eğitim fold'undan öğrenilen istatistiklerle.
- **CV:** StratifiedGroupKFold, 5 kat × 3 tekrar. Gruplar train ve test arasında bölünmediği için CV'de de bölünmüyor.
- **Karşılaştırma:** tüm deneyler aynı 15 bölmede, farklar düzeltilmiş eşleştirilmiş t-testiyle (Nadeau & Bengio) kontrol edildi.
- **Tuning:** Optuna, ayrı bir CV bölmesinde arama, bağımsız bölmelerde raporlama.

## Sonuç

| Model | CV accuracy | Kaggle public |
|---|---|---|
| Basit kural: hiç harcama yapmayan taşındı | 73.7% | – |
| Logistic Regression | 79.9% | – |
| HistGradientBoosting + harcama grupları | 81.3% | 0.80219 |
| CatBoost, Optuna ile ayarlı | 81.6% | 0.80944 |
| **Ensemble: 4 ayarlı gradient boosting** | **81.5%** | **0.81038** |
| Ensemble + `HotZone` | 81.5% | 0.80851 |

## Öğrenilenler

- **Skoru modelin ayarı değil, modele verilen bilginin yeni olup olmadığı belirledi.** Kurallı doldurma (%73–100 doğru), Optuna araması, ensemble ve keşifte bulunan kabin bölgesi: hiçbiri skoru anlamlı ölçüde artırmadı. Tek işe yarayan özellik, harcamaları lüks ve temel diye ayırmak oldu.
- **Ayar araması kendi CV'sinde +0.3 puan gösterdi, bağımsız bölmelerde bu kazanç kayboldu.** Aramayı ayrı bir bölmede yapmak bu yanılgıyı görünür kıldı.
- **Hataların %76'sı Earth'lü yolcularda, özellikle G güvertesinde.** Bu yolcuların %52–66'sı taşınmış ve elimizdeki hiçbir bilgi onları birbirinden ayırmıyor.

## Kurallar

- **Görev / metrik:** ikili sınıflandırma, accuracy.
- **Submission:** `PassengerId,Transported` (True/False), 4277 satır. Günde en fazla **10** submission.
- **Leaderboard:** public / private test ayrımı gizli; iki aydan eski submission'lar düşer. Bitiş tarihi yok.
- **Takım:** en fazla 10 kişi, tek hesap. Kod yalnızca takım içinde ya da herkese açık (forum / notebook) paylaşılabilir.
- **Veri:** yeniden dağıtılamaz (repoya eklenmez). Dış veri serbest ama herkese açık ve ücretsiz olmalı. Test verisine elle etiket vermek yasak. AutoML serbest.

## Pipeline

| # | Adım | İçerik | Durum |
|---|---|---|---|
| 1 | Veriyi tanıma | Satır/sütun sayısı, sütun anlamları, veri tipleri ve tip dönüşümü, tekrar eden kayıtlar, eksik veriler (silmek mi doldurmak mı), temel istatistikler (ortalama, medyan, dağılım) | ✅ |
| 2 | Keşifsel analiz | Her değişkenin hedefle ilişkisi: CryoSleep, harcamalar, kabin, grup/aile, gezegen, yaş | ✅ |
| 3 | Ön işleme | Sütun ayrıştırma (`PassengerId`, `Cabin`, `Name`), kurallı eksik doldurma, kodlama, ölçekleme; tümü sızıntısız `Pipeline` içinde | ✅ |
| 4 | Özellik mühendisliği | Grup büyüklüğü, toplam harcama, aile, kabin bölgesi | ✅ |
| 5 | Modeller | LR, RF, HistGB, XGBoost, LightGBM, CatBoost; aynı fold'larda, istatistiksel karşılaştırma | ✅ |
| 6 | Tuning ve ensemble | Optuna, blend / stacking | ✅ |
| 7 | Submission ve hata analizi | CV'ye göre seçim, LB ile kontrol, `HotZone` denemesi | ✅ |

Her adımın çıktıları birlikte yorumlanır, onaydan sonra bir sonraki adıma geçilir.

## Deney günlüğü

| # | Değişiklik | Accuracy | F1 | ROC AUC | Public LB |
|---|---|---|---|---|---|
| 1 | Logistic Regression, temel sütunlar, kurallı doldurma | 77.81 ± 0.84 | 78.26 | 85.30 | |
| 2 | HistGradientBoosting, temel sütunlar, kuralsız | 80.85 ± 0.63 | 80.98 | 90.03 | |
| 3 | HistGradientBoosting, temel sütunlar, kurallı doldurma | 80.75 ± 0.81 | 80.89 | 90.22 | |
| 4 | HistGB + harcama grupları → `01_hgb_spend_groups.csv` | 81.26 ± 0.81 | 81.41 | 90.49 | 80.22 |
| 5 | HistGB + tüm yeni özellikler | 81.14 ± 0.64 | 81.25 | 90.51 | |
| 6 | Logistic Regression + tüm yeni özellikler | 79.89 ± 0.91 | 80.29 | 86.87 | |
| 7 | Random Forest + tüm özellikler | 80.69 ± 0.62 | 80.63 | 89.90 | |
| 8 | XGBoost + tüm özellikler | 81.07 ± 0.52 | 81.21 | 90.50 | |
| 9 | LightGBM + tüm özellikler | 81.23 ± 0.69 | 81.20 | 90.47 | |
| 10 | CatBoost + tüm özellikler | 81.53 ± 0.90 | 81.77 | 90.84 | |
| 11 | CatBoost, Optuna ile ayarlı → `02_catboost_tuned.csv` | 81.59 ± 0.80 | 81.85 | 90.83 | 80.94 |
| 12 | LightGBM / HistGB / XGBoost, ayarlı | 81.16–81.32 | 81.55–81.68 | 90.49–90.68 | |
| 13 | Ensemble: 4 ayarlı boosting ortalaması → `03_ensemble_4_boosting.csv` | 81.51 ± 0.78 | 81.83 | 90.81 | **81.04** |
| 14 | Ensemble + `HotZone` (F/G, sağ taraf, 600–1199) → `04_ensemble_hotzone.csv` | 81.54 ± 0.73 | 81.85 | 90.86 | 80.85 |

Optuna ayarları `tuning/best_params.json` içinde; notebook aramayı yeniden çalıştırmadan bunları okuyor.

CV: StratifiedGroupKFold, 5 kat × 3 tekrar.

## Çalıştırma

Veriyi yarışma sayfasından `dataset/` altına indir (`train.csv`, `test.csv`, `sample_submission.csv`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook spaceship_titanic.ipynb
```

macOS'ta XGBoost ve LightGBM için OpenMP gerekir (`brew install libomp`). Homebrew yoksa scikit-learn'ün paketlediği kütüphane bağlanabilir:

```bash
SP=.venv/lib/python3.9/site-packages
for lib in $SP/xgboost/lib/libxgboost.dylib $SP/lightgbm/lib/lib_lightgbm.dylib; do
  ln -sf ../../sklearn/.dylibs/libomp.dylib "$(dirname $lib)/libomp.dylib"
  install_name_tool -add_rpath @loader_path "$lib"
  codesign --force -s - "$lib"
done
```
