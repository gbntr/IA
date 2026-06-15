import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

# 1. Carregando o dataset cats_vs_dogs do tensorflow_datasets
# Dividindo em dados de treino (80%) e validação (20%)
(train_ds, val_ds), info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    with_info=True,
    as_supervised=True,
)

# 2. Pré-processamento das imagens
IMG_SIZE = 150 # Redimensionar todas as imagens para 150x150
BATCH_SIZE = 32

def format_image(image, label):
    # Converte os pixels para float32
    image = tf.cast(image, tf.float32)
    # Normaliza os pixels para o intervalo [0, 1]
    image = image / 255.0
    # Redimensiona a imagem
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

# Aplicando o mapeamento, agrupando em batches e otimizando com prefetch
train_batches = train_ds.map(format_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_batches = val_ds.map(format_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 3. Construindo a rede neural CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    # Como é classificação binária (gato ou cachorro), usamos 1 neurônio com sigmoid
    layers.Dense(1, activation='sigmoid') 
])

model.summary()

# 4. Compilando o modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5. Treinando o modelo
EPOCHS = 10
history = model.fit(
    train_batches,
    epochs=EPOCHS,
    validation_data=val_batches
)

# 6. Avaliando o modelo final
loss, accuracy = model.evaluate(val_batches)
print(f"Validação final -> Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
