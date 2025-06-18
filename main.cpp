#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <memory>
#include <iostream>
#include <string>
#include <unordered_map>
#include "matrix.cpp"
#include "utils.cpp"
#include "layers.cpp"
#include "main_layers.cpp"
#include "test.cpp"

class Transformer
{
private:
   std::vector<EncoderLayer> encoder_layers;
   std::vector<DecoderLayer> decoder_layers;
   PositionalEncoding pos_encoding;
   Embedding input_embedding;
   Embedding target_embedding;
   Matrix output_projection;
   size_t d_model;
   size_t n_layers;
   size_t input_vocab_size;
   size_t target_vocab_size;

public:
   Transformer(size_t input_vocab_size, size_t target_vocab_size,
               size_t d_model = 512, size_t n_heads = 8,
               size_t n_layers = 6, size_t d_ff = 2048)
       : d_model(d_model), n_layers(n_layers),
         input_vocab_size(input_vocab_size), target_vocab_size(target_vocab_size),
         pos_encoding(d_model),
         input_embedding(input_vocab_size, d_model),
         target_embedding(target_vocab_size, d_model),
         output_projection(d_model, target_vocab_size)
   {

      // Crear layers del encoder
      for (size_t i = 0; i < n_layers; ++i)
      {
         encoder_layers.emplace_back(d_model, n_heads, d_ff);
      }

      // Crear layers del decoder
      for (size_t i = 0; i < n_layers; ++i)
      {
         decoder_layers.emplace_back(d_model, n_heads, d_ff);
      }

      // Inicializar projection layer
      output_projection.initializeXavier();
   }

   Matrix encode(const std::vector<int> &input_tokens)
   {
      // Embeddings
      Matrix embeddings = input_embedding.forward(input_tokens);

      // Escalar embeddings como en el paper original
      embeddings = embeddings.scale(std::sqrt(d_model));

      // Agregar codificacion posicional
      size_t seq_len = input_tokens.size();
      Matrix pos_enc = pos_encoding.getEncoding(seq_len);
      Matrix encoder_input = embeddings.add(pos_enc);

      // Crear mascara de padding
      Matrix src_mask = MaskUtils::createPaddingMask(input_tokens);

      // Pasar por todas las capas del encoder
      Matrix output = encoder_input;
      for (auto &layer : encoder_layers)
      {
         output = layer.forward(output, &src_mask);
      }

      return output;
   }

   Matrix decode(const std::vector<int> &target_tokens, const Matrix &encoder_output,
                 const std::vector<int> &input_tokens)
   {
      // Embeddings
      Matrix embeddings = target_embedding.forward(target_tokens);

      // Escalar embeddings
      embeddings = embeddings.scale(std::sqrt(d_model));

      // Agregar codificacion posicional
      size_t seq_len = target_tokens.size();
      Matrix pos_enc = pos_encoding.getEncoding(seq_len);
      Matrix decoder_input = embeddings.add(pos_enc);

      // Crear mascaras
      Matrix target_mask = MaskUtils::combineDecoderMasks(target_tokens);
      Matrix src_mask = MaskUtils::createPaddingMask(input_tokens);

      // Pasar por todas las capas del decoder
      Matrix output = decoder_input;
      for (auto &layer : decoder_layers)
      {
         output = layer.forward(output, encoder_output, target_mask, &src_mask);
      }

      return output;
   }

   Matrix forward(const std::vector<int> &source_tokens, const std::vector<int> &target_tokens)
   {
      Matrix encoder_output = encode(source_tokens);
      Matrix decoder_output = decode(target_tokens, encoder_output, source_tokens);

      // Proyeccion final a vocabulario de salida
      return decoder_output.multiply(output_projection);
   }

   // **NUEVO: Funcion de inferencia para generar traducciones**
   std::vector<int> generate(const std::vector<int> &source_tokens,
                             int sos_token = 1, int eos_token = 2,
                             size_t max_length = 50)
   {
      Matrix encoder_output = encode(source_tokens);

      std::vector<int> generated = {sos_token};

      for (size_t i = 0; i < max_length; ++i)
      {
         Matrix decoder_output = decode(generated, encoder_output, source_tokens);

         // Obtener logits del ultimo token
         Matrix logits = decoder_output.multiply(output_projection);
         size_t last_pos = generated.size() - 1;

         // Encontrar token con mayor probabilidad (greedy decoding)
         int next_token = 0;
         double max_score = logits[last_pos][0];
         for (size_t j = 1; j < target_vocab_size; ++j)
         {
            if (logits[last_pos][j] > max_score)
            {
               max_score = logits[last_pos][j];
               next_token = j;
            }
         }

         generated.push_back(next_token);

         if (next_token == eos_token)
         {
            break;
         }
      }

      return generated;
   }
};

// Funcion de ejemplo para usar el Transformer completo
int main()
{
   try
   {
      std::cout << "=== Transformer Completo en C++ ===" << std::endl;

      // Crear vocabularios de ejemplo
      SimpleVocab eng_vocab, spa_vocab;

      // Palabras de ejemplo en ingles
      std::vector<std::string> eng_words = {"hello", "world", "how", "are", "you", "good", "morning"};
      for (const auto &word : eng_words)
      {
         eng_vocab.addWord(word);
      }

      // Palabras de ejemplo en espanol
      std::vector<std::string> spa_words = {"hola", "mundo", "como", "estas", "tu", "buenos", "dias"};
      for (const auto &word : spa_words)
      {
         spa_vocab.addWord(word);
      }

      std::cout << "Vocabulario ingles: " << eng_vocab.size() << " palabras" << std::endl;
      std::cout << "Vocabulario espanol: " << spa_vocab.size() << " palabras" << std::endl;

      // Crear Transformer
      Transformer transformer(eng_vocab.size(), spa_vocab.size(), 128, 4, 2, 256);
      std::cout << "Transformer creado exitosamente!" << std::endl;

      // Crear secuencias de ejemplo
      std::vector<int> source = {eng_vocab.getWordId("hello"), eng_vocab.getWordId("world")};
      std::vector<int> target = {spa_vocab.getWordId("<sos>"), spa_vocab.getWordId("hola")};

      std::cout << "Secuencia fuente: ";
      for (int id : source)
      {
         std::cout << eng_vocab.getWord(id) << "(" << id << ") ";
      }
      std::cout << std::endl;

      std::cout << "Secuencia objetivo: ";
      for (int id : target)
      {
         std::cout << spa_vocab.getWord(id) << "(" << id << ") ";
      }
      std::cout << std::endl;

      // Forward pass
      Matrix output = transformer.forward(source, target);
      std::cout << "Forward pass completado!" << std::endl;
      std::cout << "Forma de salida: " << output.getRows() << "x" << output.getCols() << std::endl;

      // Prueba de generacion
      std::cout << "\n=== Prueba de Generacion ===" << std::endl;
      std::vector<int> generated = transformer.generate(source, 1, 2, 10);

      std::cout << "Secuencia generada: ";
      for (int id : generated)
      {
         std::cout << spa_vocab.getWord(id) << "(" << id << ") ";
      }
      std::cout << std::endl;

      std::cout << "\n¡Transformer completado exitosamente!" << std::endl;
   }
   catch (const std::exception &e)
   {
      std::cerr << "Error: " << e.what() << std::endl;
      return 1;
   }

   return 0;
}