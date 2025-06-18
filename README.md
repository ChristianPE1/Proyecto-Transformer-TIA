# Transformer C++ - Fase 1

Este proyecto es la **primera fase** de una réplica en C++ del modelo Transformer presentado en el paper ["Attention Is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762).

## Descripción

- Implementación desde cero en C++ de los componentes principales del Transformer:
  - Embeddings
  - Codificación posicional
  - Multi-Head Attention
  - Normalización por capas (LayerNorm)
  - Feed Forward
  - Encapado en Encoder y Decoder
  - Utilidades para máscaras y vocabulario simple

Por ahora, **no hay entrenamiento** ni carga de datos reales: el objetivo es replicar la arquitectura y el forward pass.

## Estructura del proyecto

- [`main.cpp`](main.cpp): Ejemplo de uso y definición de la clase `Transformer`.
- [`layers.cpp`](layers.cpp): Embedding, LayerNorm y FeedForward.
- [`main_layers.cpp`](main_layers.cpp): EncoderLayer y DecoderLayer.
- [`operations.cpp`](operations.cpp): MultiHeadAttention y PositionalEncoding.
- [`matrix.cpp`](matrix.cpp): Clase de matrices y operaciones básicas.
- [`utils.cpp`](utils.cpp): Funciones auxiliares (softmax, relu, máscaras).
- [`test.cpp`](test.cpp): Vocabulario simple para pruebas.

## Ejecución

Compila todos los archivos juntos, por ejemplo:

```sh
g++ -std=c++17 main.cpp -o main.exe
./main.exe
```

## Estado actual

- Forward pass funcional con datos de ejemplo y vocabularios pequeños.
- No incluye entrenamiento ni evaluación.
- Todo el código es didáctico y autocontenible.

## Próximos pasos

- Cargar un dataset para entrenamiento de traducción.
- Añadir soporte para CUDA y optimizar operaciones.

---

**Referencia principal:**  
Vaswani, A., et al. (2017).