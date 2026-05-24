import cv2
import numpy as np

class LSTMDetector:
    # Görüntüyü bloklara bölüp, bloklar arası doku/renk geçişlerini analiz eden prototip LSTM dedektörü.

    def __init__(self):
        pass

    def analyze(self, image_path):
        try:
            # Görseli oku
            try:
                file_bytes = np.fromfile(image_path, dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            except Exception as read_err:
                raise ValueError(f"Görsel dosyası okunurken hata oluştu: {str(read_err)}")

            if img is None:
                raise ValueError("Görüntü dosyası okunamadı.")

            # Gri seviyeye dönüştür
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Görüntüyü 8x8 bloğa böl
            grid_size = 8
            block_h = h // grid_size
            block_w = w // grid_size

            # Her bloğun ortalama parlaklığını hesapla
            block_means = np.zeros((grid_size, grid_size), dtype=np.float32)
            for i in range(grid_size):
                for j in range(grid_size):
                    block = gray[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                    block_means[i, j] = np.mean(block) if block.size > 0 else 0.0

            # Komşu bloklar arasındaki farkları hesapla (Yatay ve Dikey)
            diffs = []
            for i in range(grid_size):
                for j in range(grid_size - 1):
                    diffs.append(abs(block_means[i, j] - block_means[i, j+1]))
            for i in range(grid_size - 1):
                for j in range(grid_size):
                    diffs.append(abs(block_means[i, j] - block_means[i+1, j]))

            diffs = np.array(diffs, dtype=np.float32)
            diffs = np.clip(diffs, 0.0, 255.0)

            mean_diff_val = np.mean(diffs) if len(diffs) > 0 else 0.0
            std_diff_val = np.std(diffs) if len(diffs) > 0 else 0.0

            # Yüksek fark geçişleri oranı
            high_change_threshold = 25.0
            sudden_transitions = int(np.sum(diffs > high_change_threshold)) if len(diffs) > 0 else 0
            high_change_ratio = sudden_transitions / len(diffs) if len(diffs) > 0 else 0.0

            average_diff = min(100.0, (mean_diff_val / 50.0) * 100.0)
            std_diff = min(100.0, (std_diff_val / 40.0) * 100.0)

            # Risk skoru formülü
            risk_score = int(
                average_diff * 0.25 +
                std_diff * 0.35 +
                high_change_ratio * 40.0
            )

            # Kademeli sınırlandırma kuralı
            if high_change_ratio < 0.20:
                risk_score = min(risk_score, 45)
            elif high_change_ratio < 0.40:
                risk_score = min(risk_score, 65)
            else:
                risk_score = min(risk_score, 85)

            risk_score = int(np.clip(risk_score, 0, 85))

            if risk_score <= 45:
                risk_level = "Düşük Risk"
                message = "LSTM tabanlı prototip analizde belirgin sıralı doku anomalisi tespit edilmedi."
                recommendation = "Görüntüde belirgin bir kopyala-yapıştır izi tespit edilmedi."
            elif 46 <= risk_score <= 70:
                risk_level = "Orta Risk"
                message = "Görüntüde bazı blok geçiş farklılıkları tespit edildi. Bu durum ışık, gölge, arka plan karmaşıklığı veya görüntü sıkıştırmasından kaynaklanabilir."
                recommendation = "Görüntüde bazı benzer bölgeler tespit edildi. Görselin manuel olarak incelenmesi önerilir."
            else:
                risk_level = "Yüksek Risk"
                message = "Görüntüde yoğun blok geçiş anomalisi tespit edildi. Bu sonuç kesin sahtecilik anlamına gelmez, manuel inceleme önerilir."
                recommendation = "Görüntüde yoğun benzer bölge/eşleşme tespit edildi. Kopyala-yapıştır manipülasyonu olasılığı yüksektir."

            return {
                "algorithm": "LSTM Tabanlı Prototip AI Analizi",
                "keypoint_count": len(diffs),
                "match_count": sudden_transitions,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "message": message,
                "recommendation": recommendation
            }

        except Exception as e:
            raise RuntimeError(f"LSTM AI analizi gerçekleştirilirken bir hata oluştu: {str(e)}")
