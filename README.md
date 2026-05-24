# Görüntü Sahteciliği Tespit Sistemi (Image Forgery Detection System)

Görüntü Sahteciliği Tespit Sistemi, dijital görüntüler üzerinde yapılan kopyala-yapıştır (copy-move) manipülasyonlarını tespit etmek amacıyla geliştirilmiş, çok algoritmaya sahip ve kullanıcı dostu bir masaüstü uygulamasıdır.

Sistem, geleneksel bilgisayarlı görü algoritmalarından modern yapay zeka modellerine kadar uzanan geniş bir yelpazede analiz araçları sunar.

## Özellikler

*   **Klasik Bilgisayarlı Görü Algoritmaları:**
    *   **ORB:** Hızlı, verimli ve açık kaynaklı anahtar nokta çıkarımı.
    *   **AKAZE:** Doğrusal olmayan ölçek uzayı kullanarak detaylı analiz.
    *   **SIFT:** Ölçek ve dönüşümden bağımsız yüksek doğruluklu özellik çıkarımı.
    *   **SURF (Mahotas Destekli):** Patent kısıtlamaları aşılarak `mahotas` kütüphanesi ile hızlı özellik tespiti.
*   **Yapay Zeka (AI) Tabanlı Analiz:**
    *   **CNN (Evrişimli Sinir Ağları):** Derin öğrenme ile uzamsal anomali ve doku bozulmalarının tespiti.
    *   **LSTM:** Görüntü blokları arasındaki ardışık anormalliklerin analizi.
*   **Modern Kullanıcı Arayüzü (GUI):** Karanlık tema (dark mode) destekli, kullanımı kolay ve şık bir arayüz.
*   **Detaylı Raporlama:** Yapılan analizler sonucunda risk skorları (0-100 arası), tespit edilen anahtar nokta sayıları ve önerilerin bulunduğu çıktılar sağlayan konsol ve raporlama modülü.
*   **Toplu Analiz (Batch Test):** Seçilen görseli tek tıklama ile tüm desteklenen algoritmalarla test edebilme yeteneği.

## Kurulum ve Gereksinimler

Projenin çalışabilmesi için sisteminizde Python'un kurulu olması gerekmektedir (Python 3.7+ önerilir).

1. Proje dosyalarını bilgisayarınıza indirin veya klonlayın.
2. Terminal (veya Komut İstemcisi) üzerinden proje dizinine girin:
   ```bash
   cd GoruntuSahteciligiTespit
   ```
3. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

### Gerekli Temel Kütüphaneler:
- `opencv-python`: Görüntü işleme işlemleri için.
- `mahotas`: OpenCV'nin patent kısıtlaması nedeniyle desteklemediği SURF algoritması için alternatif.
- `numpy`, `pillow`, `matplotlib`, `scikit-image`: Veri manipülasyonu ve görselleştirme için.

## Kullanım

Uygulamayı başlatmak için proje dizininde aşağıdaki komutu çalıştırmanız yeterlidir:

```bash
python main.py
```

1. Arayüz açıldığında sol menüdeki **"📂 Görüntü Seç"** butonuna tıklayarak analiz etmek istediğiniz resmi yükleyin.
2. Sağ tarafta bulunan kontrol panelinden istediğiniz analiz yöntemini (ORB, AKAZE, SIFT, vb.) seçerek işlemi başlatın.
3. Sonuçlar, "Risk Skoru" ve teknik detaylarla birlikte alt kısımdaki **Analiz Konsolu**'na yazdırılacaktır.
4. **"🚀 Tüm Algoritmaları Test Et"** butonu ile görseli mevcut olan tüm bilgisayarlı görü ve AI yöntemleri ile sırasıyla test edebilirsiniz.

## Proje Mimarisi

*   `main.py`: Uygulamanın ana giriş noktasıdır. Arayüzü başlatır.
*   `ui/`: Arayüz tasarımlarını ve pencere yönetimini içeren modüller (Örn: `main_window.py`).
*   `core/`: Görüntü işleme ve analiz işlemlerinin yapıldığı ana beyin modülleri (Örn: `surf_detector.py`, `cnn_detector.py`).
*   `assets/`: Uygulamada test için kullanılabilecek veya arayüze ait varsayılan görsellerin bulunduğu klasör.
*   `reports/`: (Varsa) Uygulama tarafından oluşturulan çıktıların/raporların kaydedildiği dizin.

## Lisans

Bu proje, açık kaynak topluluğuna katkıda bulunmak ve akademik / araştırma amaçlı kullanım için geliştirilmiştir.
