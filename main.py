"""
Görüntü Sahteciliği Tespit Sistemi (Image Forgery Detection System)
Ana Giriş Noktası (main.py)

Bu dosya uygulamanın başlangıç noktasıdır. ui/main_window.py modülünü çağırarak 
grafiksel kullanıcı arayüzünü (GUI) başlatır.
"""

import sys
import tkinter as tk
from ui.main_window import MainWindow

def main():
    """
    Uygulamayı başlatan ve olası kritik çalışma zamanı hatalarını yakalayan ana fonksiyon.
    """
    try:
        # Ana pencere nesnesi oluşturuluyor
        app = MainWindow()
        # Tkinter olay döngüsü başlatılıyor
        app.run()
    except ImportError as e:
        print(f"[HATA] Gerekli kütüphaneler yüklenemedi. Lütfen 'requirements.txt' bağımlılıklarını kontrol edin.\nDetay: {e}")
        sys.exit(1)
    except tk.TclError as e:
        print(f"[HATA] Grafiksel kullanıcı arayüzü (GUI) başlatılamadı. Ekran bağlantısını veya sistem grafik desteğini kontrol edin.\nDetay: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[BEKLENMEYEN HATA] Uygulama çalışırken beklenmeyen bir hata oluştu:\n{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()