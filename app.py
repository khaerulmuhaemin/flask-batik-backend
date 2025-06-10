from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os, time, traceback, dotenv
from groq import Groq
import gdown

app = Flask(__name__)
CORS(app)
dotenv.load_dotenv()

# === [ DOWNLOAD MODEL DARI GOOGLE DRIVE JIKA BELUM ADA ] ===
model_path = './comproteam3/BatikModel2.h5'
file_id = '1ZhnK8-LjQ0hUx9tra1lodzRu_QMoq_6l'  # ID file Google Drive
#https://drive.google.com/file/d/1ZhnK8-LjQ0hUx9tra1lodzRu_QMoq_6l/view?usp=sharing

if not os.path.exists(model_path):
    print("📥 Downloading model from Google Drive...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)
else:
    print("✅ Model sudah tersedia secara lokal.")

# === [ LOAD MODEL ] ===
model = load_model(model_path)
model.make_predict_function()  # optional, depending on TensorFlow version

# === [ DAFTAR KELAS OUTPUT ] ===
output_class = [
    'Bali Barong',
    'Bali Merak',
    'Jakarta Ondel-Ondel',
    'Jakarta Tumpal',
    'Jawa Barat Megamendung',
    'Jawa Tengah Masjid Agung Demak',
    'Jawa Tengah Parang',
    'Jawa Tengah Sidoluhur',
    'Jawa Tengah Truntum',
    'Jawa Timur Gentongan',
    'Jawa Timur Pring',
    'Kalimantan Barat Insang',
    'Kalimantan Dayak',
    'Lampung Bledheg',
    'Lampung Gajah',
    'Lampung Kacang Hijau',
    'Maluku Pala',
    'NTB Lumbung',
    'Papua Asmat',
    'Papua Cendrawasih',
    'Papua Tifa',
    'Sulawesi Selatan Lontara',
    'Sumatera Barat Rumah Minang',
    'Sumatera Utara Boraspati',
    'Sumatera Utara Pintu Aceh',
    'Yogyakarta Kawung'
]

UPLOAD_FOLDER = './img_raw'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return "<h1>Batik + AI Describer</h1>"

def ai_description(category):
    try:
        token = os.getenv("API_GROQ")
        if not token:
            print("❌ API_GROQ token not found in environment.")
            return "Deskripsi tidak tersedia (token tidak ditemukan)."

        client = Groq(api_key=token)
        messages = [
            {
                "role": "system",
                "content": (
                    "Anda adalah seorang ahli seni budaya batik Indonesia."
                    " Berikan deskripsi cukup panjang 1 paragraf saja tentang motif batik yang disebutkan,"
                    " jawaban dalam bahasa Indonesia."
                )
            },
            {
                "role": "user",
                "content": category
            }
        ]
        chat = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages
        )
        reply = chat.choices[0].message.content

        #sentences = reply.split('. ')
        #description = sentences[0] + '.' if len(sentences) > 0 else "Deskripsi tidak tersedia."
        #description = reply if reply else "Deskripsi tidak tersedia."

        cleaned_reply = reply.replace('\n', ' ').strip()
        cleaned_reply = ' '.join(cleaned_reply.split())

        description = cleaned_reply if cleaned_reply else "Deskripsi tidak tersedia."


        return description
    except Exception as e:
        print("❌ Error generating description:", e)
        traceback.print_exc()
        return "Deskripsi tidak tersedia (error API)."

def predict_image(img_path):
    try:
        # === PREPROCESS IMAGE (SAMA SEPERTI KODE PERTAMA) ===
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # === PREDICT WITH MODEL ===
        pred = model.predict(img)
        predicted_class_idx = np.argmax(pred, axis=1)[0]
        predicted_probability = pred[0][predicted_class_idx]

        # === GET CLASS NAME FROM output_class ===
        if predicted_class_idx < len(output_class):
            predicted_class = output_class[predicted_class_idx]
        else:
            predicted_class = "Unknown"

        # === GET DESCRIPTION FROM AI ===
        description = ai_description(predicted_class)

        return {
            "accuracy": f"{predicted_probability:.2%}",
            "class_category": predicted_class.title(),  # Ensure from predicted_class
            "description": description
        }

    except Exception as e:
        print("❌ Prediction error:", e)
        traceback.print_exc()
        return {
            "accuracy": "-%",
            "class_category": "Not Found",
            "description": "Not Found"
        }

@app.route('/upload', methods=['POST'])
def upload():
    start_time = time.time()

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        result = predict_image(file_path)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    duration = time.time() - start_time
    print(f"🕒 Execution time: {duration:.2f} seconds")

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=1000)
