import cv2
import numpy as np

class ORBDetector:
    # ORB algoritması ile kopya-taşıma sahteciliği tespiti yapan sınıf.

    def __init__(self, max_features=2000):
        try:
            self.max_features = max_features
            self.orb = cv2.ORB_create(nfeatures=self.max_features)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        except Exception as e:
            raise RuntimeError(f"ORB Sınıfı ilklendirilirken hata oluştu: {str(e)}")

    def analyze(self, image_path):
        try:
            # Görseli oku
            try:
                file_bytes = np.fromfile(image_path, dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            except Exception as read_err:
                raise ValueError(f"Görsel dosyası okunurken hata oluştu: {str(read_err)}")

            if img is None:
                raise ValueError("Seçilen dosya geçerli bir görüntü dosyası değil veya okunamadı.")

            # Gri seviyeye dönüştür
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # ORB ile özellikleri hesapla
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)

            if descriptors is None or len(descriptors) == 0:
                raise ValueError("Görsel üzerinde analiz edilebilir ayırt edici özellik (tanımlayıcı) bulunamadı.")

            keypoint_count = len(keypoints)
            good_matches_count = 0

            # Kendi kendine eşleştirme analizi
            if len(descriptors) >= 2:
                matches = self.bf.knnMatch(descriptors, descriptors, k=2)

                for match in matches:
                    if len(match) == 2:
                        m, n = match
                        # Hamming mesafesi kontrolü
                        if n.distance < 60:
                            pt1 = keypoints[n.queryIdx].pt
                            pt2 = keypoints[n.trainIdx].pt
                            
                            # İki nokta arasındaki piksel mesafesini kontrol et
                            spatial_dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                            
                            if spatial_dist >= 30:
                                good_matches_count += 1

                # Çift yönlü eşleşmeleri tekil eşleştirmeye indirge
                good_matches_count = good_matches_count // 2

            # Risk Skoru ve Seviyesi Hesaplama
            match_ratio = good_matches_count / keypoint_count if keypoint_count > 0 else 0
            risk_score = min(100, int(match_ratio * 100))

            if risk_score <= 30:
                risk_level = "Düşük Risk"
                message = "Görsel üzerinde yapılan ORB analizinde belirgin bir kopya-taşıma (copy-move) manipülasyon izine rastlanmamıştır. Doku ve piksel dağılımları doğal görünmektedir."
                recommendation = "Görüntüde belirgin bir kopyala-yapıştır izi tespit edilmedi."
            elif 31 <= risk_score <= 65:
                risk_level = "Orta Risk"
                message = "Görsel üzerinde yapılan ORB analizinde bazı şüpheli benzer bölgeler ve tekrarlanan örüntüler tespit edilmiştir. Kısmi montaj veya piksel çoğaltma yapılmış olma ihtimali mevcuttur."
                recommendation = "Görüntüde bazı benzer bölgeler tespit edildi. Görselin manuel olarak incelenmesi önerilir."
            else:
                risk_level = "Yüksek Risk"
                message = "Görsel üzerinde yapılan ORB analizinde çok yüksek miktarda birbirine tıpatıp uyan alan eşleşmeleri saptanmıştır. Görselde bariz bir kopyala-yapıştır sahteciliği mevcuttur."
                recommendation = "Görüntüde yoğun benzer bölge/eşleşme tespit edildi. Kopyala-yapıştır manipülasyonu olasılığı yüksektir."

            return {
                "algorithm": "ORB Algoritması",
                "keypoint_count": keypoint_count,
                "match_count": good_matches_count,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "message": message,
                "recommendation": recommendation
            }

        except Exception as e:
            raise RuntimeError(f"ORB analizi gerçekleştirilirken bir hata oluştu: {str(e)}")
