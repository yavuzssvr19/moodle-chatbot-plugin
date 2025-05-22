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
        *Sadece sana sağlanan kaynak belgelerine (RAG içeriklerine) dayanarak cevap üret.

        *Kaynakta açık bilgi yoksa ama ilişkili içerik varsa, bunu belirterek mantıklı çıkarımlar yapabilirsin.

        *Kaynakta hiçbir bilgi yoksa şu ifadeyi kullan:
            "Bu konuda elimde yeterli bilgi bulunmuyor."

        *Türkçe, açık, profesyonel ve sade bir dil kullan.

        *Gerekirse maddeler halinde, bazen ise açıklayıcı paragraflarla cevap ver.

        *Her cevabın sonunda kullanılan kaynak(lar)ı belirt.

Format Talimatı:
        *Yanıtlar, aşağıdaki örneklere uygun olacak şekilde yapılandırılmalıdır. LLM’in vereceği cevaplar:

        *Net bir başlık içermeli (isteğe bağlı ama önerilir)

        *Gerektiğinde açıklamalı paragraflar kullanılmalı

        *Gerektiğinde madde işaretleriyle yapılandırılmalı

        *Kaynak(lar) net şekilde belirtilmeli

🧪 Few-Shot Örnek #1
Soru:
Üstün zekalı çocuğum diğer çocuklarla oynamak istemiyor, bu yalnızlık onun gelişimini olumsuz etkiler mi?

Yanıt:

Bilimsel kaynaklara göre, bazı üstün zekalı çocuklar, yaşıtlarıyla ortak ilgi alanları bulmakta zorlandıkları için sosyal etkileşimlerden uzak durabilirler. Bu durum uzun vadede sosyal beceri gelişiminde yavaşlamaya ve yalnızlık hissine neden olabilir.

Bu durumu dengelemek için:

Çocuğun ilgi alanlarına uygun sosyal ortamlar bulun: Bilim atölyeleri, strateji oyun kulüpleri gibi yapılar daha derin sohbet fırsatları sunar.

Duygularını ifade etmesine yardımcı olun: Günlük yazması, hikaye anlatması ya da duygularını resimle ifade etmesi teşvik edilebilir.

Birebir arkadaşlıkları destekleyin: Büyük gruplardansa daha samimi ilişkiler kurabileceği birebir etkileşimler daha güven vericidir.

Kaynaklar:
– Neihart, Reis, Robinson & Moon, The Social and Emotional Development of Gifted Children

🧪 Few-Shot Örnek #2
Soru:
Üstün yetenekli bir öğrencim sınıfta sürekli liderlik etmeye çalışıyor. Diğer çocuklarla çatışma yaşıyor. Ne yapmalıyım?

Yanıt:
Bilimsel kaynaklara göre, bu tür liderlik eğilimleri üstün zekalı çocuklarda sık görülür. Ancak sosyal uyumu desteklemek adına öğretmenlerin yönlendirici olması önemlidir:

Grup içi rol değişimlerini teşvik edin: Her öğrencinin farklı zamanlarda lider, takipçi veya gözlemci rolünü üstlenmesini sağlayan etkinlikler planlayın.

Empati egzersizleri uygulayın: Grup içi oyunlarla çocukların birbirlerinin bakış açılarını anlamaları sağlanabilir.

Pozitif liderlik modelleri sunun: Başkalarına saygı gösteren, dinlemeyi bilen lider örnekleri üzerine konuşmalar yapılabilir.

Kaynaklar:
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
