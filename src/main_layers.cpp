#pragma once
#include "matrix.cpp"
#include "utils.cpp"
#include "layers.cpp"
#include <vector>
#include <cmath>

// Encoder Layer
class EncoderLayer
{
private:
   MultiHeadAttention self_attention;
   FeedForward feed_forward;
   LayerNorm norm1, norm2;

public:
   EncoderLayer(size_t d_model, size_t n_heads, size_t d_ff = 2048)
       : self_attention(d_model, n_heads), feed_forward(d_model, d_ff),
         norm1(d_model), norm2(d_model) {}

   Matrix forward(const Matrix &input, const Matrix *src_mask = nullptr)
   {
      // Self-attention con conexion residual y normalizacion
      Matrix attention_output = self_attention.forward(input, input, input, src_mask);
      Matrix norm1_output = norm1.forward(input.add(attention_output));

      // Feed-forward con conexion residual y normalizacion
      Matrix ff_output = feed_forward.forward(norm1_output);
      Matrix norm2_output = norm2.forward(norm1_output.add(ff_output));

      return norm2_output;
   }
};

// Decoder Layer
class DecoderLayer
{
private:
   MultiHeadAttention masked_self_attention;
   MultiHeadAttention encoder_decoder_attention;
   FeedForward feed_forward;
   LayerNorm norm1, norm2, norm3;

public:
   DecoderLayer(size_t d_model, size_t n_heads, size_t d_ff = 2048)
       : masked_self_attention(d_model, n_heads),
         encoder_decoder_attention(d_model, n_heads),
         feed_forward(d_model, d_ff),
         norm1(d_model), norm2(d_model), norm3(d_model) {}

   Matrix forward(const Matrix &input, const Matrix &encoder_output,
                  const Matrix &target_mask, const Matrix *src_mask = nullptr)
   {
      // Masked self-attention
      Matrix self_att_output = masked_self_attention.forward(input, input, input, &target_mask);
      Matrix norm1_output = norm1.forward(input.add(self_att_output));

      // Encoder-decoder attention
      Matrix enc_dec_att_output = encoder_decoder_attention.forward(
          norm1_output, encoder_output, encoder_output, src_mask);
      Matrix norm2_output = norm2.forward(norm1_output.add(enc_dec_att_output));

      // Feed-forward
      Matrix ff_output = feed_forward.forward(norm2_output);
      Matrix norm3_output = norm3.forward(norm2_output.add(ff_output));

      return norm3_output;
   }
};
