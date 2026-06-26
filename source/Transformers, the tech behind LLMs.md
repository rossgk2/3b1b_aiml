# Transformers, the tech behind LLMs

The release of ChatGPT in November 2022 will always be memorable as the time when AI first captured the public conscience, never to leave. Since then, "ChatGPT" has become synonymous with AI itself. It's therefore in some sense fitting that but the esoteric "GPT" in "ChatGPT" actually hints at the inner workings of modern AI technology.

GPT, you see, stands for **Generative Pretrained *Transformer***. 

The meaning of "generative" here is fairly obvious, as ChatGPT *generates* content in response to an input prompt. So is the meaning of "pretrained", if you've followed the 3Blue1Brown videos and my articles up to now. ChatGPT is essentially a machine with lots of knobs and dials parameters that have been gradually tweaked and tuned into an optimal configuration during a *training* process.

Neither "generative" nor "pretrained" describes what's groundbreaking about ChatGPT, though. The last word here, "transformer", does. It is the name of core invention behind the boom in AI, invented by Google researchers in 2017.

There are many kinds of models that can be built using transformers: voice-to-text, text-to-voice, and text-to-image models. In this article, we'll consider the kind of text-to-text transformer used by ChatGPT.

## Predict, sample, repeat

A transformer is a function with the following input and output:

- **Input:** a sequence of words
- **Output:** a ranking of all possible words that could come after the input, in terms of how likely they are to be the correct next word[^1]

[^1]: In technical jargon, the output is a *probability distribution over all possible next words*.

To make a chatbot like ChatGPT out of a transformer, one uses the following simple process:

1. **(Predict).** Apply the transformer to the input prompt and obtain the ranking of all possible next words.
2. **(Sample).** Take a random sample of ranked words, and append the most likely of these words to the input prompt.
3. **(Repeat).** Repeat the process on the modified input prompt.

If you feel like this process is much too basic to produce complicated enough behavior enabling the semantic understanding that chatbots like ChatGPT seem to have, you're not alone. Grant says:

>  I don't know about you, but it really doesn't feel like this should actually work.

But, as we know, it somehow does work!

The only additional step is to ensure that a so-called *system prompt*, which tells the transformer to imagine what a conversation between an AI assistant and a user *would* look like, is appended to every prompt the user sends ChatGPT behind the scenes. A system prompt might look like this:

> What follows is a conversation between a user and a helpful, very knowledgeable AI assistant.

Thus, when a user tells ChatGPT "Give me some ideas for what to do when visiting Santiago.", ChatGPT is really receiving something like:

> What follows is a conversation between a user and a helpful, very knowledgeable AI assistant.
>
> User: Give me some ideas for what to do when visiting Santiago.

## Inside a transformer

To begin understanding how a transformer produces the probability distribution that it does, Grant kicks things off with a high-level preview of how data flows through one: 

1. **(Tokenization).** First, the input is broken up into a bunch of little pieces called *tokens*. In the case of text, tokens might be sequences of characters separated by delimiters- it's accurate enough to simply imagine them being words. (In the case of images or audio, tokens might be little pieces of an overall image, or little pieces of the overall audio clip.)
2. **(Embedding).** Each token is associated with a vector (a list of numbers) in such a way that tokens with similar meaning correspond to vectors that, when interpreted as locations in a higher-dimensional space, are close to each other. (Later on in the video, we learn that we like to think of the act of associating tokens with vectors as "embedding" the tokens into a "space" of vectors.)
3. **(Attention).** All of these vectors are acted on by an *attention* operation. The output of the attention operation is a list of transformed vectors whose meanings have been refined by the meanings of the other input vectors they appear near. It can be helpful to interpret the input vectors as being the columns of a matrix, and to imagine the attention operation as a sort of physical machine that the input matrix moves through before proceeding to the next machine.
4. **(Feedforward).** That next machine is a neural network referred to as a *feedforward layer*. Output vectors from the attention operation are acted on in parallel, and independently of each other, by this feedforward layer, which is sometimes also called a *multilayer perceptron*, or *MLP*.
5. **(Repeat).** The process repeats many times. The output vectors from the previous step are passed through more and more attention-block-feedforward-layer pairs.

## Chapter layout

In this article, we'll begin understanding how a transformer produces the probability distribution that it does. There are two parts to this beginning. 

First, we'll expand on what happens at the very beginning and very end of a transformer function. The key invention at the heart of the transformer, the *attention* mechanism- aptly mentioned in the seminal research paper on transformers, *Attention is All You Need*- will be covered in the next article.

Second, we'll review background knowledge that would have been second nature to any machine learning engineer at the time *Attention is All You Need* was published, like word embeddings, dot products, and softmax.

## The very beginning: word embedding

The *embedding matrix* $\mathbf{W}_E$ is a matrix whose columns are all of the words in the dictionary

"Turning words into vectors was common practice in machine learning long before transformers, but it's a little weird if you've never seen it before, and it sets the foundation for everything that follows, so let's take a moment to get familiar with it"

$E(\text{woman}) - E(\text{man}) \approx E(\text{queen}) - E(\text{king})$

$\textbf{plur} = E(\text{cats}) - E(\text{cat})$

$\textbf{plur} \cdot E(\text{one}) < \textbf{plur} \cdot E(\text{two}) < \textbf{plur} \cdot E(\text{three}) < \textbf{plur} \cdot E(\text{four})$

### Embeddings beyond words

Consider the following input:

> "The King doth wake tonight and takes his rouse"

We can associate each input word with an embedding vector so that, for instance, the embedding vector for "King" lives in that portion of the embedding space associated with "male ruler of a nation".

In this state of things, the embedding vectors just represent mere, simple words. But, precisely because they live in high-dimensional spaces, they have the *capacity* to soak in so much more context than this.

This is precisely what a transformer achieves. As embedding vectors progress through more and more transformer operations, they begin to point in more and more specific and nuanced directions than they did originally, so that, by the end, the embedding for "King" not only corresponds to "male ruler of a nation", but also somehow points in a specific and nuanced direction that encodes "this is a king who lived in Scotland, who achieved his post after murdering the previous king, and who's being described in Shakespearian language".

## The very end: unembedding