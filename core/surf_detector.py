import cv2
import numpy as np

class SURFDetector:
    # SURF algoritması ile kopya-taşıma sahteciliği tespiti yapan sınıf.

    def __init__(self):
        pass

    def analyze(self, image_path):
        # 1. SURF Desteği Kontrolü
        surf_supported = False
        use_mahotas = False
        try:
            if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SURF_create"):
                surf_supported = True
        except Exception:
            pass

        if not surf_supported:
            try:
                import mahotas.features.surf
                surf_supported = True
                use_mahotas = True
            except ImportError:
                pass

        if not surf_supported:
            return {
                "algorithm": "SURF Analizi",
                "keypoint_count": 0,
                "match_count": 0,
                "risk_score": 0,
                "risk_level": "Desteklenmiyor",
                "message": "SURF algoritması bu OpenCV kurulumunda desteklenmiyor. Alternatif SURF motoru için 'mahotas' kütüphanesi bulunamadı.",
                "recommendation": "SURF analizini aktifleştirmek için terminale 'pip install mahotas' yazarak kütüphaneyi yükleyebilirsiniz."
            }

        try:
            # 2. Görseli Oku
            try:
                file_bytes = np.fromfile(image_path, dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            except Exception:
                return {
                    "algorithm": "SURF Analizi",
                    "keypoint_count": 0,
                    "match_count": 0,
                    "risk_score": 0,
                    "risk_level": "Hata",
                    "message": "Görüntü dosyası okunamadı.",
                    "recommendation": "Lütfen geçerli bir görüntü dosyası seçin."
                }

            if img is None:
                return {
                    "algorithm": "SURF Analizi",
                    "keypoint_count": 0,
                    "match_count": 0,
                    "risk_score": 0,
                    "risk_level": "Hata",
                    "message": "Görüntü dosyası okunamadı.",
                    "recommendation": "Lütfen geçerli bir görüntü dosyası seçin."
                }

            # 3. Gri seviyeye dönüştür
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if use_mahotas:
                import mahotas.features.surf
                
                class KeyPointDummy:
                    def __init__(self, pt):
                        self.pt = pt
                
                spoints = mahotas.features.surf.surf(gray)
                if spoints is not None and len(spoints) > 0:
                    # spoints format: [y, x, scale, score, sign, descriptor...]
                    keypoints = [KeyPointDummy((float(p[1]), float(p[0]))) for p in spoints]
                    descriptors = np.array(spoints[:, 5:], dtype=np.float32)
                else:
                    keypoints = []
                    descriptors = None
            else:
                surf = cv2.xfeatures2d.SURF_create(400)
                keypoints, descriptors = surf.detectAndCompute(gray, None)

            if descriptors is None or len(descriptors) == 0:
                return {
                    "algorithm": "SURF Analizi",
                    "keypoint_count": len(keypoints),
                    "match_count": 0,
                    "risk_score": 0,
                    "risk_level": "Düşük Risk",
                    "message": "Görsel üzerinde analiz edilebilir ayırt edici SURF özelliği (tanımlayıcı) bulunamadı.",
                    "recommendation": "Görüntüde belirgin bir kopyala-yapıştır izi tespit edilmedi."
                }

            keypoint_count = len(keypoints)
            good_matches_count = 0
            bf = cv2.BFMatcher(cv2.NORM_L2)

            # Kendi kendine eşleştirme analizi
            if len(descriptors) >= 2:
                matches = bf.knnMatch(descriptors, descriptors, k=2)

                for match in matches:
                    if len(match) == 2:
                        m, n = match
                        # Öklid mesafesi kontrolü
                        if n.distance < 250.0:
                            pt1 = keypoints[n.queryIdx].pt
                            pt2 = keypoints[n.trainIdx].pt
                            
                            # İki nokta arasındaki piksel mesafesini kontrol et
                            spatial_dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                            
                            if spatial_dist >= 30:
                                good_matches_count += 1

                # Çift yönlü eşleşmeleri tekil eşleştirmeye indirge
                good_matches_count = good_matches_count // 2

            # Risk Skoru Hesaplama
            risk_score = min(100, int((good_matches_count / keypoint_count) * 100)) if keypoint_count > 0 else 0

            if risk_score <= 30:
                risk_level = "Düşük Risk"
                message = "Görsel üzerinde SURF analizi ile belirgin bir kopyala-yapıştır izine rastlanmamıştır. Doku sürekliliği korunmuştur."
                recommendation = "Görüntüde belirgin bir kopyala-yapıştır izi tespit edilmedi."
            elif 31 <= risk_score <= 65:
                risk_level = "Orta Risk"
                message = "Görsel içerisinde SURF ile şüpheli bazı benzer alanlar ve tekrarlanan doku blokları tespit edilmiştir. Kısmi manipülasyon şüphesi mevcuttur."
                recommendation = "Görüntüde bazı benzer bölgeler tespit edildi. Görselin manuel olarak incelenmesi önerilir."
            else:
                risk_level = "Yüksek Risk"
                message = "Görselde SURF ile çok sayıda birbirine uyan alan kopyalaması belirlenmiştir. Kopyala-yapıştır manipülasyonu yapılmış olma ihtimali çok yüksektir."
                recommendation = "Görüntüde yoğun benzer bölge/eşleşme tespit edildi. Kopyala-yapıştır manipülasyonu olasılığı yüksektir."

            return {
                "algorithm": "SURF Analizi",
                "keypoint_count": keypoint_count,
                "match_count": good_matches_count,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "message": message,
                "recommendation": recommendation
            }

        except Exception as e:
            raise RuntimeError(f"SURF analizi gerçekleştirilirken bir hata oluştu: {str(e)}")
