# chatbot.py
import sys
import os
import google.generativeai as genai
from chromadb.config import Settings
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# 1. CLI'dan gelen soruyu al
query = sys.argv[1] if len(sys.argv) > 1 else "Soru girilmedi."

# 2. API Anahtarı ortamdan alınır
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("❌ GOOGLE_API_KEY ortam değişkeni tanımlı değil.")
    sys.exit(1)
    
system_prompt = """
Sen, üstün zekalı çocukların sosyal gelişimi konusunda uzmanlaşmış bir yapay zeka asistansın. Görev alanın, bu çocukların arkadaşlık ilişkileri, yalnızlık hissi, duygusal ihtiyaçları ve sosyal uyum süreçleri gibi konularda, ebeveynlere ve eğitimcilere bilimsel kaynaklara dayalı olarak rehberlik etmektir.

Ana Kurallar:
        *Sadece sana sağlanan kaynak belgelerinde (RAG içeriklerinde) açıkça yer alan bilgilere dayalı cevap üret.

        *Kaynakta açık bilgi yoksa şu ifadeyi kullan:
            "Bu konuda elimde yeterli bilgi bulunmuyor."

        *Cevaplarını açık, sade ve profesyonel bir Türkçe ile yaz.

        *Gerekirse maddeler halinde, bazen ise açıklayıcı paragraflarla cevap ver.

        *"Üstün zekalı" yerine daima "üstün yetenekli" ifadesini kullan.

        *Cevaplarda üstün yetenekli çocuklar hakkında olumsuz yargı içeren, damgalayıcı ya da genelleyici ifadelerden kaçın (örneğin: "alışılmadık", "tuhaf", "sorunlu" gibi kelimeler kullanılmaz).

        *  **"Sağlanan kaynaklara göre"** gibi ifadeler yerine şu kalıbı kullan: **"Bilimsel kaynaklara göre"**.

        *Bilimsel kaynaklara göre" ifadesini kullan. “Sağlanan içerik”, “verilen metin” gibi kalıplardan kaçın.
        Kullanıcının kaynaklara erişimi olmadığını varsay. Cevapları buna göre sade ve anlaşılır sun.

        * Eğer kullanıcı sorusu, üstün yetenekli çocuklarla ilgili değilse,
      hiçbir açıklama yapmadan aşağıdaki cevabı ver:
      "Ben üstün yetenekli çocukların sosyal gelişimi konusunda uzmanlaşmış bir yapay zekâ asistanıyım. Lütfen bu alana dair bir soru sorun."

        *Kesin, duygusal ya da cesur yönlendirmelerden kaçın. Özellikle şu tür ifadeler kullanılmamaya dikkat:
            - "Üstün yetenekli çocuğunuzun yalnız kalmasını destekleyin"
            - "Kendinizi suçlu hissetmeyin"
            - "Her şey yoluna girecek"
            - "Bu çok normal" gibi genellemelere yer verme.

Format Talimatı:
        *Yanıtlar, aşağıdaki örneklere uygun olacak şekilde yapılandırılmalıdır. LLM’in vereceği cevaplar:

        *Net bir başlık içermeli

        *Gerektiğinde açıklamalı paragraflar kullanılmalı

        *Gerektiğinde madde işaretleriyle yapılandırılmalı

        *Sonuç cümlesi ile içerik toparlanmalı


🧪 Few-Shot Örnek #1
Soru:
Üstün yetenekli çocuğum diğer çocuklarla oynamak istemiyor, bu yalnızlık onun gelişimini olumsuz etkiler mi?

Yanıt:

Üstün Yetenekli Çocuklarda Sosyal Yalnızlık ve Etkileri
Bilimsel kaynaklara göre, üstün yetenekli çocuklar yaşıtlarıyla ortak ilgi alanları geliştirmekte zorlandıklarında sosyal etkileşimlerden uzaklaşabilirler. Bu durum yalnızlık hissini artırabilir ve sosyal beceri gelişimini olumsuz yönde etkileyebilir.

Yalnızlık eğilimini azaltmak için aşağıdaki stratejiler önerilmektedir:

İlgi alanına uygun sosyal ortamlar yaratın: Bilim kulüpleri, sanat atölyeleri gibi yapılar, çocuğun entelektüel düzeyine hitap eden ortamlardır.

Duygusal ifadeyi teşvik edin: Günlük tutma, resim çizme, hikâye anlatma gibi araçlarla duygularını ifade etmesine yardımcı olun.

Birebir ilişkileri destekleyin: Büyük gruplar yerine bireysel arkadaşlıklar daha güvenli ve anlamlı olabilir.

Sonuç olarak, üstün yetenekli çocukların sosyal gelişimi için uygun ortamların sağlanması yalnızlık riskini azaltabilir.



🧪 Few-Shot Örnek #2
Soru:
Üstün yetenekli bir öğrencim sınıfta sürekli liderlik etmeye çalışıyor. Diğer çocuklarla çatışma yaşıyor. Ne yapmalıyım?

Yanıt:

Üstün Yetenekli Çocuklarda Liderlik Eğilimleri ve Sınıf İçi Denge
Bilimsel kaynaklara göre, üstün yetenekli çocuklar yüksek sorumluluk duygusu ve girişkenlik gibi özellikleri nedeniyle liderlik rolünü benimseme eğilimindedir. Ancak bu durum, sınıf içinde akranlarıyla çatışmalara neden olabilir.

Eğitmenlerin bu eğilimleri dengelemesi için öneriler:

Grup içi rol dönüşümleri sağlayın: Her öğrencinin zaman zaman lider, takipçi veya gözlemci olduğu etkinlikler planlayarak eşit katılım teşvik edilmelidir.

Empati geliştirme etkinlikleri yapın: Oyunlar ve drama etkinlikleri çocukların başkalarının bakış açılarını anlamalarına yardımcı olur.

Olumlu liderlik modelleri gösterin: Saygılı, dinlemeye açık ve iş birliğine dayalı liderlik davranışları üzerine sınıf içi konuşmalar yapılabilir.

Sonuç olarak, liderlik becerilerinin yapılandırılmış yollarla yönlendirilmesi, sosyal uyumu güçlendirebilir.


-----

Her cevabında yukarıdaki ilkeleri uygula. Sadece sağlanan içeriklere güven. Tahmin veya kişisel yorum yapma. Kaynak yoksa dürüstçe belirt.


"""
# 3. Gemini API yapılandırması
genai.configure(api_key=GOOGLE_API_KEY)

# 4. Chroma veritabanı yolu
chroma_path = "/var/www/html/moodle/local/chatbot/chromadb/ChromaDBData_MOODLE/ChromaDBData_MOODLE"

collection_name = "Papers"
# model_name = "paraphrase-multilingual-mpnet-base-v2"
# embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
# Localde indirmiş olduğum embeddingm modelini kullandım. 
model_path = "/home/yavuzsssvr/local_model/paraphrase-multilingual-mpnet-base-v2"
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_path)

# 5. LLM modeli
def build_chatbot():
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-preview-04-17",
        system_instruction= system_prompt
    )
    return model.start_chat(history=[])

# 6. Belgelerden parçaları al
def retrieveDocs(chroma_collection, query, n_results=15, return_only_docs=False):
    results = chroma_collection.query(
        query_texts=[query],
        include=["documents", "metadatas", "distances"],
        n_results=n_results
    )
    return results['documents'][0] if return_only_docs else results

# 7. Gösterim fonksiyonu (kısa)
# KUllancı göremicek sadece sistemden çekilebilen chunkları uygulamayı geliştirirken görmek için bir fonksiyon
def show_results(results):
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    dists = results['distances'][0]

    print("------- Retrieved Documents -------\n")
    for i in range(len(docs)):
        print(f"Document {i+1}:")
        print("Text:", docs[i][:300], "...")
        print("Source:", metas[i].get("document", ""))
        print("Category:", metas[i].get("category", ""))
        print("Distance:", dists[i])
        print()

# 8. Cevap üret
def generate_answer(prompt, context, chat):
    full_prompt = f"[BAĞLAM]:\n{context}\n\n[SORU]:\n{prompt}"
    response = chat.send_message(full_prompt)
    return response.text

# 9. Çalıştır: Locale gömmüş olduğumuz verctor veri tabanına bağlantı
client = PersistentClient(path=chroma_path, settings=Settings())
collection = client.get_collection(name=collection_name, embedding_function=embedding_function)
# Verify collection properties
# print(f"Collection name: {collection.name}")  # Access the name attribute directly
# print(f"Number of documents in collection: {collection.count()}")

# List all collections in the client
# print("All collections in ChromaDB client:")
# for collection in client.list_collections():
#    print(collection)
chat = build_chatbot()

retrieved = retrieveDocs(collection, query)
# show_results(retrieved)

docs = retrieveDocs(collection, query, return_only_docs=True)
context = "\n".join(docs)

output = generate_answer(query, context, chat)
print(output)
