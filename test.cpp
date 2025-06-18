#pragma once
#include <unordered_map>
#include <string>
#include <vector>

class SimpleVocab
{
public:
   std::unordered_map<std::string, int> word_to_id;
   std::unordered_map<int, std::string> id_to_word;

   SimpleVocab()
   {
      // Tokens especiales
      word_to_id["<pad>"] = 0;
      word_to_id["<sos>"] = 1;
      word_to_id["<eos>"] = 2;
      word_to_id["<unk>"] = 3;

      id_to_word[0] = "<pad>";
      id_to_word[1] = "<sos>";
      id_to_word[2] = "<eos>";
      id_to_word[3] = "<unk>";
   }

   void addWord(const std::string &word)
   {
      if (word_to_id.find(word) == word_to_id.end())
      {
         int id = word_to_id.size();
         word_to_id[word] = id;
         id_to_word[id] = word;
      }
   }

   int getWordId(const std::string &word)
   {
      auto it = word_to_id.find(word);
      return (it != word_to_id.end()) ? it->second : word_to_id["<unk>"];
   }

   std::string getWord(int id)
   {
      auto it = id_to_word.find(id);
      return (it != id_to_word.end()) ? it->second : "<unk>";
   }

   size_t size() const
   {
      return word_to_id.size();
   }
};