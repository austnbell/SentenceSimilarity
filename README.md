# SentenceSimilarity
Semantic Similarity Between Sentences

Implementation of Mueller and Thyagarajan (2016) paper titled "Siamese Reccurent architectures for Learning Sentence Similarity". 
[Link to download](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwiNr7CpyuzjAhWH1lkKHaRrBrQQFjAAegQIARAC&url=https%3A%2F%2Fwww.aaai.org%2Focs%2Findex.php%2FAAAI%2FAAAI16%2Fpaper%2Fdownload%2F12195%2F12023&usg=AOvVaw1LpgzBISqbbU-VfpKyx9-M)


Overview:
Various data pre-processing to augment training data and model initialization followed by Siamese LSTM to estimate Sentence Semantic Similarity. 

Data: 
* STS SemEval 2013 for pretraining
* SICK for training

Data Augmentation and Transfer Learning: 
* Employ Thesaurus-based augmentation.  
  * Identify synsets for each Noun, Adjective, Verb, and Adverb in each sentence
  * Replace and augument data
    * Geometric distribution to select number of training sentences to add
    * Randomly select which words to replace with synonyms
    * Geometric distribution to select which synonyms to replace with
* Pretrain Sentence Similarity Model on STS SemEval 2013 data
  * Initialize core weights with pretrained weights
  
Model:
* Manhattan Siamese LSTM (MaLSTM)
  * input: two sentences into shared LSTM
  * Utilize L1 norm in similarity function 
