import re
import matplotlib.pyplot as plt
import ollama

try:
    with open('epstein.txt', 'r', encoding='utf-8') as file:
        raw_text = file.read()
except FileNotFoundError:
    print("Помилка: файл не знайдено")
    exit()

sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw_text) if len(s.strip()) > 10]

categories = {
    "Юридичні процеси та суд": 0,
    "Зв'язки з відомими особами": 0,
    "Теорії змови та конспірологія": 0,
    "Інше": 0
}


def classify_text_with_llm(text_fragment):
    prompt = f"""Ти — експерт-аналітик. Прочитай наступне речення і віднеси його суворо до однієї з трьох категорій:
1. Юридичні процеси та суд
2. Зв'язки з відомими особами
3. Теорії змови та конспірологія
Якщо текст взагалі не стосується цих тем, напиши "Інше".

У відповіді напиши ТІЛЬКИ назву категорії (один рядок), без цифр на початку та без крапок у кінці.

Текст для аналізу:
{text_fragment}
"""
    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        result = response['message']['content'].strip()

        result_lower = result.lower()
        if "юрид" in result_lower or "суд" in result_lower:
            return "Юридичні процеси та суд"
        elif "зв'яз" in result_lower or "відомим" in result_lower:
            return "Зв'язки з відомими особами"
        elif "змов" in result_lower or "конспіролог" in result_lower:
            return "Теорії змови та конспірологія"
        else:
            return "Інше"

    except Exception as e:
        print(f"Помилка Ollama: {e}")
        return "Інше"


for i, s in enumerate(sentences):
    clean_s = re.sub(r'[^\w\s\.,!?\'"«»—-]', '', s)

    cluster_name = classify_text_with_llm(clean_s)

    categories[cluster_name] += 1

    print(f"Речення {i + 1} -> Кластер: [{cluster_name}]")
    print(f"Текст: {clean_s[:60]}...\n")


labels_filtered = []
values_filtered = []
for label, value in categories.items():
    if value > 0:
        labels_filtered.append(label)
        values_filtered.append(value)

if not values_filtered:
    print("Немає даних для побудови графіка")
else:
    clr = ['#4bb2c5', '#EAA228', '#579575', '#958c12']

    plt.figure(figsize=(9, 6))
    plt.pie(values_filtered, labels=labels_filtered, autopct='%1.1f%%',
            startangle=140, colors=clr[:len(values_filtered)], wedgeprops={'edgecolor': 'black'})
    plt.title("Розподіл тексту за тематичними", fontsize=14, pad=20)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()