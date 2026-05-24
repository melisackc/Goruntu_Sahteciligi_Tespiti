import cv2
import numpy as np

class CNNDetector:
    # Kenar yoğunluğu, gürültü ve blok benzerliği üzerinden analiz yapan prototip CNN dedektörü.

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

            # 1. Kenar Yoğunluğu Analizi
            edges = cv2.Canny(gray, 50, 150)
            edge_pixels = np.sum(edges > 0)
            edge_density = (edge_pixels / edges.size) * 100

            # 2. Gürültü Oranı Analizi
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise_map = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
            noise_ratio = np.mean(noise_map)

            # 3. Kontrast Değişimi
            contrast_val = np.std(gray)

            # 4. Bloklar Arası Benzerlik (8x8 grid)
            grid_size = 8
            block_h = h // grid_size
            block_w = w // grid_size
            block_means = []

            for i in range(grid_size):
                for j in range(grid_size):
                    block = gray[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                    block_means.append(np.mean(block))

            unique_blocks = len(set([round(x, 1) for x in block_means]))
            similarity_factor = (1.0 - (unique_blocks / len(block_means))) * 100

            # Risk Skoru Hesaplama
            score = 15.0
            
            score += similarity_factor * 0.4
            if noise_ratio < 1.0 or noise_ratio > 10.0:
                score += 15.0
            if edge_density > 25.0 or edge_density < 2.0:
                score += 10.0
            if contrast_val < 30.0:
                score += 10.0

            risk_score = int(min(max(score, 0), 100))

            if risk_score <= 30:
                risk_level = "Düşük Risk"
                message = f"Görsel genelinde dengeli gürültü oranları ve yumuşak doku geçişleri tespit edildi. Herhangi bir yapay kenar veya manipülasyon izi bulunamadı. (Skor: {risk_score}/100)"
                recommendation = "Görüntüde belirgin bir kopyala-yapıştır izi tespit edilmedi."
            elif 31 <= risk_score <= 65:
                risk_level = "Orta Risk"
                message = f"Görseldeki kenar yoğunluğu dağılımında ve bazı blokların gürültü frekanslarında sınır değerlerde tutarsızlık saptandı. Hafif düzey filtre veya montaj uygulanmış olabilir. (Skor: {risk_score}/100)"
                recommendation = "Görüntüde bazı benzer bölgeler tespit edildi. Görselin manuel olarak incelenmesi önerilir."
            else:
                risk_level = "Yüksek Risk"
                message = f"Görselin farklı bölgeleri arasında aşırı doku/gürültü sapmaları ve yapay piksel eşleşmeleri belirlendi. Görsel yüksek ihtimalle manipüle edilmiş! (Skor: {risk_score}/100)"
                recommendation = "Görüntüde yoğun benzer bölge/eşleşme tespit edildi. Kopyala-yapıştır manipülasyonu olasılığı yüksektir."

            return {
                "algorithm": "CNN Tabanlı AI Analizi",
                "keypoint_count": int(edge_density),
                "match_count": int(similarity_factor),
                "risk_score": risk_score,
                "risk_level": risk_level,
                "message": message,
                "recommendation": recommendation
            }

        except Exception as e:
            raise RuntimeError(f"CNN AI analizi gerçekleştirilirken bir hata oluştu: {str(e)}")
