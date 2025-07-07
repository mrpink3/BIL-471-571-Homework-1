import json 
import os
import re
import string
from collections import defaultdict
from typing import Dict, List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error

# 🔹 1. LC Dosyalarını Oku ve Birleştir
def load_lc_file(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        return data.get("answers", []), data.get("questions", [])

def combine_questions_and_answers(answers: List[Dict], questions: List[str]) -> Dict[str, List[str]]:
    combined = defaultdict(list)
    for answer in answers:
        student_id = answer.get("id")
        for i in range(1, len(questions)):
            question = questions[i]
            cevap_key = f"cevap{i}"
            cevap = answer.get(cevap_key, "")
            if isinstance(cevap, str) and cevap.strip():
                combined[student_id].append(f"Soru: {question} Cevap: {cevap}")
    return combined

def aggregate_lc_with_questions(file_names: List[str], base_path: str) -> Dict[str, str]:
    all_combined = defaultdict(list)
    for file_name in file_names:
        full_path = os.path.join(base_path, file_name)
        answers, questions = load_lc_file(full_path)
        student_combined = combine_questions_and_answers(answers, questions)
        for student_id, texts in student_combined.items():
            all_combined[student_id].extend(texts)
    return {student_id: " ".join(texts) for student_id, texts in all_combined.items()}

# 🔹 2. Metinleri Temizle
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    turkish_stopwords = {
        "ve", "bu", "bir", "şu", "için", "ile", "de", "da", "ne", "mi", "mı", "mu", "mü",
        "ama", "fakat", "çünkü", "ancak", "gibi", "ki", "daha", "çok", "az", "en", "ise",
        "her", "şey", "hiç", "yani", "olan", "olanlar", "oldu", "olabilir", "vardır", "yoktur",
        "soru", "cevap"
    }
    return " ".join([token for token in tokens if token not in turkish_stopwords])

def preprocess_all_students(student_texts: Dict[str, str]) -> Dict[str, str]:
    return {student_id: preprocess_text(text) for student_id, text in student_texts.items()}

# 🔹 3. Model Kur, Eğit ve Değerlendir
def train_and_evaluate(X, y):
    model = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
    y_pred = cross_val_predict(model, X, y, cv=5)
    mse = mean_squared_error(y, y_pred)

    print("\n✅ Logistic Regression modeli eğitildi.")
    print(f"🎯 Mean Squared Error (MSE): {mse:.2f}\n")

    results = pd.DataFrame({
        "Gerçek Seviye": y,
        "Tahmin": y_pred
    })
    print(results.head(10))

# 🔹 4. Ana Fonksiyon
if __name__ == "__main__":
    lc_files = [
        "LC1.section1.json", "LC1.section2.json",
        "LC2.section1.json", "LC2.section2.json",
        "LC3.section1.json", "LC3.section2.json",
        "LC4.section1.json", "LC4.section2.json",
        "LC5.section1.json",
        "LC6.section1.json", "LC6.section2.json"
    ]

    base_path = "data/LCs/"
    train_csv_path = "data/processed_train_student.csv"

    print("🔄 Eğitim verisi hazırlanıyor...")

    # LC dosyalarından metinleri al
    student_combined_text = aggregate_lc_with_questions(lc_files, base_path)

    # Metinleri temizle
    student_cleaned_text = preprocess_all_students(student_combined_text)

    # Etiket dosyasını oku ve eşleşenleri al
    train_df = pd.read_csv(train_csv_path)
    train_df.rename(columns={"UID": "student_id", "MidtermClass": "level"}, inplace=True)
    train_df["student_id"] = train_df["student_id"].astype(str)
    matched_df = train_df[train_df["student_id"].isin(student_cleaned_text.keys())].copy()
    matched_df["cleaned_text"] = matched_df["student_id"].map(student_cleaned_text)

    # TF-IDF
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(matched_df["cleaned_text"])
    y_train = matched_df["level"].values

    print(f"✅ TF-IDF vektörleme tamamlandı.")
    print(f"➡️  Öğrenci sayısı (n_samples): {X_train.shape[0]}")
    print(f"➡️  Özellik sayısı (n_features): {X_train.shape[1]}")

    # Model eğitimi
    print("🚀 Model eğitiliyor...")
    train_and_evaluate(X_train, y_train)
    print("✅ Model eğitimi tamamlandı.")