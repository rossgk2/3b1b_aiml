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

The third step, *attention*, is the core idea at the heart of transformers. It was aptly featured in the title of the seminal 2017 Google research paper that introduced transformers, *Attention is All You Need*. 

Since attention is a relatively complicated mechanism, Grant saves the discussion of exactly how it works for the next video. The current video and article are dedicated to explaining everything else- what happens at the very beginning of the transformer, and what happens at the end. This "everything else" would have been second nature to any machine learning engineer when *Attention is All You Need* was published, so it's worth taking time to familiarize ourselves with it.

## The very beginning: word embedding

If there was ever to be any hope of computationally representing the semantics of human language, it was always going to be necessary to figure out how to represent human words, and the relationships between them, mathematically. It shouldn't be too surprising, considering this, that the machine learning community had come with ways to represent words, long before attention and transformers were introduced in 2017.

Specifically, it was already possible to have a word representation function that maps every word to a vector (a list of numbers)[^2].

[^2]: Google's Word2Vec, presented in the *Efficient Estimation of Word Representations in Vector Space*, was the first such library providing this functionality.

Since a vector of $n$ numbers can interpreted to the coordinates of a point in $n$-dimensional space, we can think of a word representation function as mapping words to locations. And, if we conflate the words with their representations, pretending that a word *is* its representation, we might suggestively say that that a word representation function *embeds* words into that space; it takes the hazy, abstract notions of human words, and grounds them in the more tangible reality of lists of numbers.

For these metaphorical reasons, we have the following terminology:

* The word representation function (the function sending words to vectors) is known as an *embedding*.
* A vector representing a word is also sometimes called an *embedding*. (So, you have to use context to figure out whether "word representation function" or "representation of a word" is meant!)
* The set of all possible word representation is called the *embedding space*.

### Mirrored relationships

The exciting idea to represent words with vectors is only actually useful if the relationships between vectors somehow *mirror* the relationships[^3] between the corresponding words. For example, we might have the prior impression that the semantic difference between the words "woman" and "man" is about the same as the semantic difference between "queen" and "king".
$$
\text{woman} - \text{man} \approx \text{queen} - \text{king}
$$
[^3]: We want the embedding to be a vector space isomorphism between the space of words and the space of embedded vectors.

Of course, it's not really clear how one is supposed to subtract "man" from "woman". However, if we could take the embedding vector of "man", we *could* subtract that from the embedding vector of "woman". And we could do the same for "queen" and "king". So, if $\mathbf{E}$ is the embedding function, we might expect that we would have something like
$$
\mathbf{E}(\text{woman}) - \mathbf{E}(\text{man}) \approx \mathbf{E}(\text{queen}) - \mathbf{E}(\text{king}).
$$

Marvelously, when we use an actual embedding library to check this, we find that it is indeed true.

At this point, the obvious question is- how in the world was the machine learning community able to come up with embedding functions that have this mirroring property? The answer is *training*. The embedding function is secretly a neural network that takes words as input, produces vectors as output, and that has been trained to do this mapping in a way that ensures mirroring of relationships.

### Dot product semantics

There's even more semantics to be explored in the embedding space. This peculiarity, though, requires a bit of prior knowledge from linear algebra. So, here is a quick crash course:

> There is an operation between two vectors $\mathbf{v}$ and $\mathbf{w}$ that is denoted by $\vv \cdot \ww$, that is read out loud as "v dot w", and is defined to be the result of multiplying the length of the projection (the "shadow") of $\mathbf{v}$ onto $\mathbf{w}$ by the length of $\mathbf{w}$. Notice that since the dot product of two vectors is a product of lengths, it is a number. When we have vectors $\mathbf{v} = \begin{pmatrix} v_1 \\\\ vdots \\\\ v_n \end{pmatrix}$ and $\mathbf{w} = \begin{pmatrix} w_1 \\\\ vdots \\\\ w_n \end{pmatrix}$, it is possible to prove that the dot product of $\mathbf{v}$ and $\mathbf{w}$ is equal to the result of multiplying corresponding components and summing them: $\vv \cdot \ww = v_1 w_1 + ... + v_n w_n$.

For the full details, you can see [my book on linear algebra, tensors, and manifolds](https://github.com/rossgk2/Linear-algebra-tensors-manifolds). For our purposes, though, all that need be understood is that the dot product of two vectors is a tool we are fortunate to have, since it is both geometrically useful (as it measures a vector projection) and easy to compute (as all you have to do is multiply and sum vector components).

Now we return to word embeddings. 

You might imagine that if you take any plural version of a noun (like "cats") and then "subtract" the singular version ("cat") you will end up with the letter "s". Or, more fundamentally, you end up with the notion of plurality- after all, in English, we pluralize nouns by adding "s" to them.

We can formalize this idea with embeddings by defining a *plurality embedding* to be a difference of any plural embedding (like "cats") and its corresponding singular embedding ("cat"):
$$
\textbf{plur} := \mathbf{E}(\text{cats}) - \mathbf{E}(\text{cat}).
$$
Now, here is the magic. If we use the dot product to measure the projection of the embeddings of the English words "one", "two", "three", and "four" onto this plurality vector, we see that the measurements are increasing:
$$
\mathbf{E}(\text{one}) \cdot \textbf{plur} < \mathbf{E}(\text{two}) \cdot \textbf{plur} < \mathbf{E}(\text{three}) \cdot \textbf{plur} < \mathbf{E}(\text{four}) \cdot \textbf{plur}
$$
So, a properly trained embedding space will "know" that "one" is less plural than "two", which is less plural than "three", which is less plural than "four".

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