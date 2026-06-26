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

To build a chatbot like ChatGPT from a transformer, one uses the following simple process:

1. **(Predict).** Apply the transformer to the input prompt and obtain the ranking of all possible next words.
2. **(Sample).** Take a random sample of ranked words, and append the most likely of these words to the input prompt.
3. **(Repeat).** Repeat the process on the modified input prompt.

If you feel like this process is too basic to produce complicated enough behavior enabling the semantic understanding that chatbots like ChatGPT seem to have, you're not alone. In the video, Grant says:

>  I don't know about you, but it really doesn't feel like this should actually work.

But, as we know, it somehow does work!

The only additional step is to ensure that a so-called *system prompt*, which tells the transformer to imagine what a conversation between an AI assistant and a user *would* look like, is appended to every prompt the user sends ChatGPT behind the scenes. A system prompt might look like this:

> What follows is a conversation between a user and a helpful, very knowledgeable AI assistant.

Thus, when a user tells ChatGPT "Give me some ideas for what to do when visiting Santiago.", ChatGPT is really receiving something like this:

> What follows is a conversation between a user and a helpful, very knowledgeable AI assistant.
>
> User: Give me some ideas for what to do when visiting Santiago.

## Inside a transformer

To begin understanding how a transformer produces the probability distribution that it does, Grant kicks things off with a high-level preview of how data flows through one: 

1. **(Tokenization).** First, the input is broken up into a bunch of little pieces called *tokens*. In the case of text, tokens might be sequences of characters separated by delimiters- it's accurate enough to simply imagine them being words. (In the case of images or audio, tokens might be little pieces of an overall image, or little pieces of the overall audio clip.)
2. **(Embedding).** Each token is associated with a vector (a list of numbers) in such a way that tokens with similar meaning correspond to vectors that, when interpreted as locations in a higher-dimensional space, are close to each other. (Later on in the video, we learn that we like to think of the act of associating tokens with vectors as "embedding" the tokens into a "space" of vectors.)
3. **(Attention).** All of these vectors are acted on by an *attention* operation. The output of the attention operation is a list of transformed vectors whose meanings have been refined by the meanings of the other input vectors they appear near. It can be helpful to interpret the input vectors as being the columns of a matrix, and to imagine the attention operation as a sort of physical machine that the input matrix moves through before proceeding to the next machine.
4. **(Feedforward).** That next "machine" is a neural network referred to as a *feedforward layer*. Output vectors from the attention operation are acted on in parallel, and independently of each other, by this feedforward layer, which is sometimes also called a *multilayer perceptron*, or *MLP*.
5. **(Repeat).** The process repeats many times. The output vectors from the previous step are passed through more and more attention-block-feedforward-layer pairs.

## Chapter layout

Attention is the core idea at the heart of transformers. It was aptly mentioned in the title of the seminal 2017 Google Brain research paper that introduced transformers, *Attention is All You Need*. All of the other pieces of the transformer function were well known to the machine learning community when transformers were introduced.

Since attention is a relatively complicated mechanism, Grant saves the discussion of exactly how it works for the next video. The current video, and article, are dedicated to explaining everything else- what happens at the very beginning of the transformer, and what happens at the end, while covering the background knowledge that would have been second nature to any machine learning engineer when *Attention is All You Need* was published.

## The very beginning: word embedding

If there was ever to be any hope of computationally representing the semantics of human language, it was always going to be necessary to figure out how to represent human words, and the relationships between them, mathematically. It shouldn't be too surprising, considering this, that the machine learning community had come with ways to represent words, long before attention and transformers were introduced in 2017.

Specifically, it was already possible to have a word representation function that maps every word to a vector (a list of numbers)[^2].

[^2]: Google's Word2Vec, presented in the *Efficient Estimation of Word Representations in Vector Space*, was the first such library providing this functionality.

Since a vector of $n$ numbers can interpreted to the coordinates of a point in $n$-dimensional space, we can think of a word representation function as mapping words to locations. And, if we conflate the words with their representations, pretending that a word *is* its representation, we might suggestively say that that a word representation function *embeds* words into that space; it takes the hazy, abstract notions of human words, and grounds them in the more tangible reality of lists of numbers, in such a way that relationships between the representations somehow mirror the relationships between the originals.

For these metaphorical reasons, we have the following terminology:

* The word representation function (the function sending words to vectors) is known as an *embedding*.
* A vector representing a word is also sometimes called an *embedding*. (So, you have to use context to figure out whether "word representation function" or "representation of a word" is meant!)
* The set of all possible word representation is called the *embedding space*.

### Mirrored relationships

It really is true that embeddings preserve the relationships between words. It's pretty awe-inspiring. For example, we might have the prior impression that the semantic difference between the words "woman" and "man" is about the same as the semantic difference between "queen" and "king":
$$
\text{woman} - \text{man} \approx \text{queen} - \text{king}
$$
In regular human language, there's no real way to quantify precisely in what sense in which this is true. But, if we have an embedding $E$ that maps words to vectors, then we can apply the embedding to both sides of the above to obtain
$$
E(\text{woman} - \text{man}) \approx E(\text{queen} - \text{king})
$$
So far, we've only used the embedding to rephrase the previous approximation we suspected. The key insight is that, if relationships are to mirrored, then representation of the difference must be the same as the difference of the representations[^3]. Thus the above is equivalent to
$$
E(\text{woman}) - E(\text{man}) \approx E(\text{queen}) - E(\text{king})
$$


[^3]: It is easiest to see this when, for a representation function $E$ mapping words to vectors, we write $w \mapsto \mathbf{v}$ to indicate $E(w) =  \mathbf{v}$. With this notation, it is easy to see that representations mirror their originals only if $c_1 w_1 + c_2 w_2 \mapsto c_1 E(w_1) + c_2 E(w_2)$. (And *this* is just the idea behind a mathematical isomorphism.)

Since this approximation only involves words, and not differences of words like $\text{woman} - \text{man}$, it can be put to the test with the embedding function $E$. 

And, when we test it by using an embedding library, we do indeed find it holds.

### Dot product semantics



$\textbf{plur} = E(\text{cats}) - E(\text{cat})$

$\textbf{plur} \cdot E(\text{one}) < \textbf{plur} \cdot E(\text{two}) < \textbf{plur} \cdot E(\text{three}) < \textbf{plur} \cdot E(\text{four})$

### Embeddings in a transformer

The *embedding matrix* $\mathbf{W}_E$ is a matrix whose columns are all of the words in the dictionary

"Turning words into vectors was common practice in machine learning long before transformers, but it's a little weird if you've never seen it before, and it sets the foundation for everything that follows, so let's take a moment to get familiar with it"

$$

Consider the following input:

> "The King doth wake tonight and takes his rouse"

We can associate each input word with an embedding vector so that, for instance, the embedding vector for "King" lives in that portion of the embedding space associated with "male ruler of a nation".

In this state of things, the embedding vectors just represent mere, simple words. But, precisely because they live in high-dimensional spaces, they have the *capacity* to soak in so much more context than this.

This is precisely what a transformer achieves. As embedding vectors progress through more and more transformer operations, they begin to point in more and more specific and nuanced directions than they did originally, so that, by the end, the embedding for "King" not only corresponds to "male ruler of a nation", but also somehow points in a specific and nuanced direction that encodes "this is a king who lived in Scotland, who achieved his post after murdering the previous king, and who's being described in Shakespearian language".

## The very end: unembedding