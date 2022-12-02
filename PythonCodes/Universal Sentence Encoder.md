# Universal Sentence Encoder
#UniversalSentenceEncoder

## Concept

https://arxiv.org/pdf/1803.11175.pdf

### Transfer learning via sentence embeddings
- encoding *sentences* into embedding (high-dimensional) vectors
- target transfer learning to other NLP tasks like text classification, semantic similarity, clustering
- greater-than-word length text: sentences, phrases, or short paragraphs
- 2 variants of the encoding models allow for trade-offs between accuracy and compute resources
- sentence embeddings tends to outperform word level transfer
- good performance with minimal amounts of supervised training data
- transformer architecture and a deep averaging network (DAN) encoder

## Process
- input: English strings
- output: a fixed dimensional embedding representation of the string
- can be used directly => compute sentence level semantic similarity scores
- can be incorporated into larger model graphs => can be fine tuned for specific tasks using gradient based updates


## 2 encoding models
### Both
- Input a lowercased PTB tokenized string, and output a 512 dimensional vectors as the sentence embedding
- Multi-task learning whereby a single encoding model / DAN encoder is used to feed multiple down-stream tasks

### Transformer architecture, high accuracy
- $O(n^2)$
- As general purpose as possible
- High accuracy at the cost of greater model complexity and resource consumption
- encoding sub-graph of the transformer architecture
- use attention to compute context aware representations of words in sentence
- ordering and identity of all other words
- convert to a fixed length sentence encoding vector
- No sentence length effect: computing the element-wise sum of the representations at each word position, and divide by the square root of the length of the sentence

- the cost of compute time and memory usage scaling dramatically with sentence length

## Deep Averaging Network (DAN)
- $O(n)$
- Efficient inference with slightly reduced accuracy
- input embeddings for words and bi=grams are first averaged together, then passed through a feedforward deep neural network to produce sentence embeddings
- Input a lowercased PTB tokenized string, and output a 512 dimensional vectors as the sentence embedding
- compute time is linear in the length of the input sequence


## Semantic similarity task
- compute the cosine similarity of the two sentence embeddings
- use arccos to convert the cosine similarity into an angular distance


## Transfer learning is critically important when training data for a target task is limited



