# Vision Transformer (ViT) - Implementación en C++

[![Language](https://img.shields.io/badge/language-C++-blue.svg)](https://isocpp.org/)
[![Standard](https://img.shields.io/badge/C++-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Una implementación completa desde cero del **Vision Transformer (ViT)** en C++ puro, capaz de clasificar imágenes en múltiples datasets médicos y tradicionales. Este proyecto implementa el mecanismo de self-attention y arquitectura transformer para visión por computadora.

## Tabla de Contenidos

- [Características](#-características)
- [Arquitectura](#-arquitectura)
- [Datasets Soportados](#-datasets-soportados)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Resultados](#-resultados)
- [Documentación Técnica](#-documentación-técnica)
- [Contribuir](#-contribuir)
- [Referencias](#-referencias)

## Características

- **Implementación completa del Vision Transformer** desde cero en C++
- **Multi-Head Self-Attention** con escalado y normalización
- **Soporte para múltiples datasets**:
  - MNIST clásico - Accuracy ~90
  - Fashion-MNIST - Accuracy ~89
  - Afro-MNIST - Accuracy ~85
- **Entrenamiento y evaluación** con métrica de accuracy
- **Guardado y carga de modelos** entrenados (formato .bin)
- **Procesamiento de patches** de imagen configurable
- **Sin dependencias externas** (implementación pura en C++)

## Arquitectura

### Vision Transformer (ViT)

```
Input Image (28x28) → Patches (7x7) → Linear Projection → 
→ Position Embedding → VIT Block → Classification Head
```

**Componentes principales:**

1. **Patch Embedding**: División de la imagen en patches y proyección lineal
2. **(VIT Block) Multi-Head Attention**: Mecanismo de atención con múltiples cabezas
3. **(VIT Block) Layer Normalization**: Normalización de capas para estabilidad
4. **(VIT Block) Feed Forward Network**: Redes completamente conectadas
5. **Classification Head**: Capa final para clasificación

### Fórmula de Atención

```
Attention(Q,K,V) = softmax(QK^T / √d_k)V
```

Donde:
- Q = Query matrix
- K = Key matrix  
- V = Value matrix
- d_k = Dimensión de las claves

## Datasets Soportados

| Dataset | Clases | Tamaño Entrenamiento | Tamaño Test | Descripción |
|---------|--------|---------------------|-------------|-------------|
| **MNIST** | 10 | 60,000 | 10,000 | Dígitos manuscritos |
| **Fashion-MNIST** | 10 | 60,000 | 10,000 | Artículos de moda |
| **Afro-MNIST** | 10 | 60,000 | 10,000 | Imágenes de dígitos manuscritos africanos |

## Instalación

### Prerrequisitos

- **Compilador C++17** (GCC, Clang, o MSVC)
- **CMake 3.18+**
- **Git** (para clonar el repositorio)

### Compilación

```bash
# Clonar el repositorio
git clone https://github.com/ChristianPE1/Proyecto-Transformer-TIA.git
cd Proyecto-Transformer-TIA

# Crear directorio de construcción
mkdir build && cd build

# Configurar con CMake
cmake ..

# Compilar
make

# O usar CMake para compilar
cmake --build .
```

### Ejecutable generado

```bash
# El ejecutable se genera en build/bin/
./build/bin/main
```

## Uso

### Entrenamiento Básico

Cambiar la linea de entrenamiento en `main.cpp` para seleccionar el dataset:

```cpp
// Entrenar con MNIST
ViTMNIST model = train_mnist();

// Entrenar con Fashion-MNIST  
ViTMNIST model = train_fashion();

// Entrenar con OrganCMNIST
ViTMNIST model = train_organc();

// Continuar entrenamiento desde pesos guardados
ViTMNIST model = continue_train_organc();
```

### Configuración de Parámetros

```cpp
int patch_size = 7;              // Tamaño de patches (7x7)
int embed_dim = 64;              // Dimensión de embedding
int num_heads = 2;               // Número de cabezas de atención
int num_layers = 3;              // Número de capas transformer
int mlp_hidden_layers_size = 96; // Tamaño de capas ocultas MLP
int num_classes = 11;            // Número de clases de salida

ViTMNIST vit_model(patch_size, embed_dim, num_heads, 
                   num_layers, mlp_hidden_layers_size, num_classes);
```

### Parámetros de Entrenamiento

```cpp
int num_epochs = 15;           // Número de épocas
int batch_size = 64;           // Tamaño de lote
float learning_rate = 0.0001f; // Tasa de aprendizaje
int save_each_epoch = 3;       // Guardar cada N épocas

Trainer trainer(num_epochs, batch_size, learning_rate);
trainer.train(vit_model, train_data, test_data, save_each_epoch);
```

## Estructura del Proyecto

```
Proyecto-Transformer-TIA/
├── src/
│   ├── main.cpp                    # Punto de entrada principal
│   ├── data/
│   │   └── mnist_loader.hpp        # Cargador de datasets (MNIST, NPY)
│   ├── layers/
│   │   ├── feed_forward.cpp        # Red feed-forward
│   │   ├── layer_norm.cpp          # Normalización de capas
│   │   └── linear.cpp              # Capas lineales
│   ├── training/
│   │   └── classification_loss.cpp # Función de pérdida
│   ├── transformer/
│   │   ├── attention.cpp           # Multi-head attention
│   │   └── vit_mnist.cpp          # Vision Transformer principal
│   └── utils/
│       ├── classifier.cpp          # Clasificador
│       ├── matrix.cpp              # Operaciones de matrices
│       └── trainer.cpp             # Entrenador
├── data/                           # Datasets
│   ├── afro/                       # Afro-MNIST (.npy)
│   ├── mnist/                      # MNIST clásico (.ubyte)
│   ├── fashion/                    # Fashion-MNIST (.ubyte)
│   ├── organc/                     # OrganCMNIST (.npy)
│   └── blood/                      # BloodCMNIST (.npy)
├── weights/                        # Modelos entrenados (.bin)
├── CMakeLists.txt                  # Configuración de construcción
└── README.md                       # Este archivo
```

## Resultados


| Dataset | Test Accuracy | Épocas | Batch Size | Learning Rate |
|---------|---------------|--------|------------|---------------|
| MNIST | ~90%* | 50 | 64 | 0.001 |
| Fashion-MNIST | ~80%* | 20 | 64 | 0.001 |
| Afro-MNIST | 98% | 15 | 32 | 0.0005 |

*Resultados aproximados, pueden variar según la configuración y el dataset.

## Documentación Técnica

### Mecanismo de Atención

La implementación del **Multi-Head Attention** sigue estos pasos:

1. **Proyección a Q, K, V:**
   ```cpp
   Matrix Q = query.multiply(W_Q);
   Matrix K = key.multiply(W_K);
   Matrix V = value.multiply(W_V);
   ```

2. **Cálculo de scores (similitud):**
   ```cpp
   float scale = 1.0f / std::sqrt((float)d_model);
   float score = dot_product(Q[i], K[j]) * scale;
   ```

3. **Aplicación de Softmax:**
   ```cpp
   attention_weights[i][j] = exp(score - max_score) / sum_exp;
   ```

4. **Aplicación de atención a valores:**
   ```cpp
   output[i] = sum(attention_weights[i][j] * V[j]);
   ```

### Carga de Datos

El proyecto soporta dos formatos:

- **MNIST tradicional (.ubyte)**: Formato binario original
- **NPY (.npy)**: Formato NumPy para datasets médicos

```cpp
// Cargar MNIST tradicional
MNISTData data = loader.load(images_path, labels_path);

// Cargar formato NPY (OrganCMNIST, BloodCMNIST, Afro-MNIST)
auto [train_data, test_data] = loader.load_organc_mnist(data_dir);
```

### Guardado de Modelos

```cpp
// Guardar pesos del modelo
vit_model.save_weights("vit-{epoch}.bin");
Los pesos se guardan automáticamente durante el entrenamiento:

```cpp
// Se guarda como: "vit-{epoch}.bin"
vit_model.save_weights("vit-{epoch}.bin");

// Cargar pesos guardados
vit_model.load_weights("vit-{epoch}.bin");
```



### Áreas de mejora

- Implementación de data augmentation
- Soporte para GPU/CUDA
- Optimización de memoria
- Más datasets médicos
- Interfaz gráfica para visualización

## Referencias

1. **Dosovitskiy, A., et al.** (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR 2021*. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

2. **Vaswani, A., et al.** (2017). "Attention Is All You Need." *NIPS 2017*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

## Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## Autor

**Christian Pardavé Espinoza** - [ChristianPE1](https://github.com/ChristianPE1)

**Berly Diaz Castro** - [Berly01](https://github.com/Berly01)

**Leonardo Montoya Choque** - [Legonnarth](https://github.com/Legonnarth)

**Saul Condori Machaca** - [SaulCondoriM](https://github.com/SaulCondoriM)

---

*Proyecto desarrollado como parte del curso de Tópicos en Inteligencia Artificial (TIA) - UNSA*
