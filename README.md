# Moodle AI Chatbot Plugin 

Bu proje, Moodle tabanlı bir öğrenim yönetim sistemine (LMS) **AI destekli bir Chatbot** entegre etmeyi amaçlar. Chatbot, belirli PDF kitaplardan oluşturulan **vektör veritabanı** (ChromaDB) üzerinden kullanıcı sorularına yanıt üretir. Arka planda Google Gemini API ve Sentence Transformers modeli kullanılır.

## Sistem Mimarisi

graph TD
    Kullanıcı -->|Soru| Moodle Chatbot Arayüzü
    Moodle Chatbot Arayüzü -->|Soru| chatbot.py
    chatbot.py -->|Embedding| SentenceTransformer
    chatbot.py -->|Sorgu| ChromaDB
    chatbot.py -->|Yanıt| Google Gemini API
    Google Gemini API -->|Cevap| chatbot.py
    chatbot.py -->|Yanıt| Moodle Arayüzü``` 

## Proje Yapısı
moodle-chatbot-plugin/
├── chatbot.py
├── chromadb/ 
│   ├── chroma.sqlite3
│   ├── <segment dosyaları>
├── lang/
│   └── en/
│       └── local_chatbot.php
├── db/
│   └── access.php
├── version.php
├── index.php
├── lib.php
└── README.md

## Dosya Açıklamaları
'chatbot.py': 

    * Amaç: Komut satırından (CLI) veya Moodle web arayüzünden gelen soruları alır.

    * İşlevi: ChromaDB vektör veritabanından benzer belgeleri alır, Gemini LLM ile anlamlı bir cevap üretir.

    * Bağımlılıklar: google.generativeai, chromadb, sentence-transformers, transformers

'chromadb/': (Github reposunda bu klsör mevcut değil ancak siz kendi klosör yapınız içerisinde kendi verilerinizi gömebilirsiniz)
Amaç: ChromaDB formatında saklanan vektörleştirilmiş PDF içerikleri içerir.

İçerik: 

    * chroma.sqlite3: Chroma'nın metadata deposudur.

    * 87c5.../: Her koleksiyon için oluşturulan HNSW segment klasörü.

    * *.bin, *.pickle: Vektör indeksleme ve sorgu için gerekli segment dosyalarıdır.

 'lang/en/local_chatbot.php':
    * Amaç: Moodle eklentisinin kullanıcı izinlerini tanımlar.

    * İşlevi: Kimin bu eklentiyi kullanabileceğini Moodle seviyesinde belirler.

 'version.php':
    * Amaç: Moodle sistemine bu eklentinin sürüm, uyumluluk ve bağımlılık bilgilerini bildirir.

    * Zorunludur.

'index.php':
    * Amaç: Chatbot arayüzünün çalıştığı Moodle sayfasıdır.

    * İçerik: HTML + PHP ile Gradio veya basit form tabanlı frontend arayüz sağlar.

    * Yönlendirme: chatbot.py dosyasına shell exec komutuyla mesaj gönderir.
## API Anahtarı ve Ortam Değişkenleri
chatbot.py aşağıdaki ortam değişkenlerini kullanır:

    * GOOGLE_API_KEY → Gemini LLM için gerekli.

    * .env yerine, sunucunun ortam değişkenlerinde tanımlanmalıdır (/etc/environment veya export).

## Kullanılan Teknolojiler
| Kütüphane               | Amaç                                                                   |
| ----------------------- | ---------------------------------------------------------------------- |
| `chromadb`              | Vektör veritabanı yönetimi                                             |
| `sentence-transformers` | Metin embedding işlemleri                                              |
| `transformers`          | HuggingFace tabanlı model yükleme                                      |
| `pypdf`                 | PDF sayfalarından metin çıkartma                                       |
| `google-generativeai`   | Google Gemini API istemcisi                                            |
| `langchain`             | Text splitter ve embed yapıları için (opsiyonel kullanımda)            |
| `tqdm`                  | İlerleme çubuğu (opsiyonel, debug için)                                |
| `python-dotenv`         | Ortam değişkenlerini `.env` dosyasından okuma (opsiyonel ama önerilir) |


## Kurulum ve Çalıştırma
1. chromadb/ klasörünü ve chatbot.py'yi Moodle sunucusuna yerleştir.

2. Gerekli Python ortamını oluştur:
      """python3 -m venv chatbotenv
      source chatbotenv/bin/activate
      pip install -r requirements.txt"""
3. Ortam değişkenlerini tanımla:
      """export GOOGLE_API_KEY="your-api-key" """

## Web Arayüzü Kullanımı

Moodle içinden /local/chatbot/index.php adresine git. Sorunuzu yazın ve gönderin. Arayüz, chatbot.py'yi terminalden çalıştırarak cevap üretir ve ekrana basar.


