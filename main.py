---

## 🧠 Código completo (`main.py`)

```python
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from matplotlib import pyplot as plt

# === Etapa 1: Carregar e filtrar o dataset ===
base_path = "dataset/lfw-deepfunneled/lfw-deepfunneled"
pessoas_selecionadas = sorted(os.listdir(base_path))[:5]

images = []
labels = []

for person in pessoas_selecionadas:
    person_path = os.path.join(base_path, person)
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (100, 100))
            images.append(img)
            labels.append(person)

# === Etapa 2: Pré-processamento ===
X = np.array(images).reshape(-1, 100, 100, 1) / 255.0
le = LabelEncoder()
y = to_categorical(le.fit_transform(labels))
classes = le.classes_

# === Etapa 3: Criar e treinar o modelo ===
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(100,100,1)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, batch_size=16)

# === Etapa 4: Salvar o modelo ===
os.makedirs("model", exist_ok=True)
model.save("model/reconhecimento_facial.h5")

# === Etapa 5: Reconhecimento facial ===
model = load_model("model/reconhecimento_facial.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

img_path = os.path.join(base_path, pessoas_selecionadas[0], os.listdir(os.path.join(base_path, pessoas_selecionadas[0]))[0])
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (100, 100))
    face = face / 255.0
    face = face.reshape(1, 100, 100, 1)

    prediction = model.predict(face)
    predicted_label = np.argmax(prediction)
    nome = classes[predicted_label]

    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, nome, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title(f"Rosto reconhecido: {nome}")
plt.show()
