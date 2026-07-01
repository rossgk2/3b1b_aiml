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

Since attention is a relatively complicated mechanism, Grant saves the discussion of exactly how it works for the next video. The current video and article are dedicated to explaining everything else- some background on word embeddings, what happens before attention, what the goal of attention is, and what happens after attention. This "everything else" would have been familiar to any machine learning engineer when *Attention is All You Need* was published, so it's worth taking time to familiarize ourselves with it.

First, let's go over some background.

## Background: word embeddings

If there was ever to be any hope of computationally representing the semantics of human language, it was always going to be necessary to figure out how to represent human words, and the relationships between them, mathematically. It shouldn't be too surprising, considering this, that the machine learning community had come with ways to represent words, long before attention and transformers were introduced in 2017.

Specifically, it was already possible to have a word representation function that maps every word to a vector (a list of numbers)[^2].

[^2]: Google's Word2Vec, presented in the *Efficient Estimation of Word Representations in Vector Space*, was the first such library providing this functionality.

Since a vector of $n$ numbers can interpreted to the coordinates of a point in $n$-dimensional space, we can think of a word representation function as mapping words to locations, or equivalently, directions (since every location corresponds to the direction between it and the origin). And, if we conflate the words with their representations, pretending that a word *is* its representation, we might suggestively say that that a word representation function *embeds* words into that space; it takes the hazy, abstract notions of human words, and grounds them in the more tangible reality of lists of numbers.

For these metaphorical reasons, we have the following terminology:

* The word representation function (the function sending words to vectors) is known as an *embedding*.
* A vector representing a word is also sometimes called an *embedding*. (So, you have to use context to figure out whether "word representation function" or "representation of a word" is meant!)
* The set of all possible word representation is called the *embedding space*.

### Mirrored relationships

The exciting idea to represent words with vectors is only actually useful if the relationships between vectors somehow *mirror* the relationships between the corresponding words. For example, we might have the prior impression that the semantic difference between the words "woman" and "man" is about the same as the semantic difference between "queen" and "king".
$$
\text{woman} - \text{man} \approx \text{queen} - \text{king}
$$
Of course, it's not really clear how one is supposed to subtract "man" from "woman". However, if we could take the embedding vector of "man", we *could* subtract that from the embedding vector of "woman". And we could do the same for "queen" and "king". So, if $\mathbf{E}$ is the embedding function, we might expect that we would have something like
$$
\mathbf{E}(\text{woman}) - \mathbf{E}(\text{man}) \approx \mathbf{E}(\text{queen}) - \mathbf{E}(\text{king}).
$$

Marvelously, when we use an actual embedding library to check this, we find that it is indeed true.

At this point, the obvious question is- how in the world was the machine learning community able to come up with embedding functions that have this mirroring property? The answer is *training*. The embedding function is secretly a neural network that takes words as input, produces vectors as output, and that has been trained to do this mapping in a way that ensures mirroring of relationships.

### Dot product semantics

There's even more semantics to be explored in the embedding space. This peculiarity, though, requires a bit of prior knowledge from linear algebra. So, here is a quick crash-course:

> There is an operation between two vectors $\mathbf{v}$ and $\mathbf{w}$ that is denoted by $\vv \cdot \ww$, that is read out loud as "v dot w", and is defined to be the result of multiplying the length of the projection (the "shadow") of $\mathbf{v}$ onto $\mathbf{w}$ by the length of $\mathbf{w}$. Since the dot product of two vectors is a product of lengths, it is a number. When we have vectors $\mathbf{v} = \begin{pmatrix} v_1 \\\\ vdots \\\\ v_n \end{pmatrix}$ and $\mathbf{w} = \begin{pmatrix} w_1 \\\\ vdots \\\\ w_n \end{pmatrix}$, it is possible to prove that the dot product of $\mathbf{v}$ and $\mathbf{w}$ is equal to the result of multiplying corresponding components and summing them: $\vv \cdot \ww = v_1 w_1 + ... + v_n w_n$.

For the full details, you can see [my book on linear algebra, tensors, and manifolds](https://github.com/rossgk2/Linear-algebra-tensors-manifolds). For our purposes, though, all that need be understood is that the dot product of two vectors is an easily computable measure of how much they align.

Now we return to word embeddings. 

You might imagine that if you take any plural version of a noun (like "cats") and then "subtract" the singular version ("cat") you will end up with the letter "s". More fundamentally, you end up with the notion of plurality- after all, in English, we pluralize nouns by adding "s" to them.

We can formalize this idea with embeddings by defining a *plurality embedding* to be a difference of any plural embedding (like "cats") and its corresponding singular embedding ("cat"):
$$
\textbf{plur} := \mathbf{E}(\text{cats}) - \mathbf{E}(\text{cat}).
$$
Now, here is the magic. If we use the dot product to measure how much each of the embeddings of the English words "one", "two", "three", and "four" align with the plurality vector, we see that the measurements are increasing:
$$
\mathbf{E}(\text{one}) \cdot \textbf{plur} < \mathbf{E}(\text{two}) \cdot \textbf{plur} < \mathbf{E}(\text{three}) \cdot \textbf{plur} < \mathbf{E}(\text{four}) \cdot \textbf{plur}
$$
So, a properly trained embedding space will "know" that "one" is less plural than "two", which is less plural than "three", which is less plural than "four".

## The steps in a transformer

Now that we're ready to revisit the steps in a transformer in more depth, it'll help to have an example. Let's suppose a user types the following into a chatbot:

> Harry Potter, the boy who

We're expecting that the chatbot correctly predict the word that should follow this phrase, "lived". Let's see how a transformer-based chatbot would handle this!

### Before attention: tokenization and embedding

**(Tokenization).** First, the input is first broken up into a bunch of little pieces called *tokens*. In our example, the tokens might be "Harry", "Potter,", "the", "boy", "who", "lived".

**(Embedding).** We map the tokens to embedded vectors by using the *embedding matrix*, which has one column vector for every word in the dictionary. The embedding matrix is denoted by $\mathbf{W}_E$, and was learned during training in such a way that it ensures the semantic "mirroring" we discussed earlier.

As we pass to the next step, it's convenient to conceptualize our list of embedded vectors as being the column vectors of a matrix $\mathbf{X}$,
$$
\mathbf{X} := \begin{pmatrix} | & | & | & | & | \\ \mathbf{E}(\text{Harry}) & \mathbf{E}(\text{Potter,}) & \mathbf{E}(\text{the}) & \mathbf{E}(\text{boy}) & \mathbf{E}(\text{who}) \\ | & | & | & | & | \end{pmatrix}
$$


### The goal of attention: embeddings beyond words

**(Attention, feedforward, repeat).** When our matrix $\mathbf{X}$ of embedded vectors arrives at the attention step, its first column vector, "Harry", points in whatever direction in the embedding space is associated with traditional British male names. In fact, *all* of the embedded vectors in the matrix represent mere, simple words. But, precisely because they reside in a high-dimensional space, they have the *capacity* to soak in so much more context than this.

Thus, as $\mathbf{X}$ passes through more and more attention blocks (and associated feedforward layers), the hope, and goal, is that its column vectors begin to point in more and more specific and nuanced directions than they did originally. The hope is that by the end, the embedding for "Harry" not only points in the direction associated with traditional British male names, but also somehow in a more specific and nuanced direction that encodes "this Harry is the globally famous fictional character, who, in his story, is an orphan, a wizard, and famous, even though all he ever wanted was a normal life with a loving family".

### After attention: unembedding

**(Unembedding). **After our matrix $\mathbf{X}$ comes out of the attention block, its last column vector ($\mathbf{E}(\text{who})$, in our case), being a highly-contextualized version of the last word in the input phrase, points in a direction that indicates it is strongly semantically related to whatever the next word should be. The only thing that needs to be done at this point, then, is to extract that information somehow[^3].

[^3]: It might seem strange that we *only* use information from the last vector to make our next-word prediction. All of the other vectors are just sitting there, holding lots of context-rich meaning! Grant tells us that one reason the other vectors are present is because they happen to be useful for training. Another, more fundamental reason, is that the last vector is only able to attain the rich contextual meaning it does if, during the attention mechanism, the contextual meanings of the other vectors are present to influence it

What we do is take this last vector, $\mathbf{E}(\text{who})$, and compute how much it aligns with a embedding of every possible word. Of course, to compute how much two vectors align, we take their dot product. So, we will need to take the dot product of the last vector, $\mathbf{E}(\text{who})$, with the embedding for each possible word. Doing so gives us a column vector of dot products, which we can label by the embedding being compared against:
$$
\begin{pmatrix}
	\mathbf{E}(\text{who}) \cdot (\text{an embedding of ``aah''}) & | & \text{aah} \\
	\mathbf{E}(\text{who}) \cdot (\text{an embedding of ``aardvark''}) & | & \text{aardvark} \\
	\mathbf{E}(\text{who}) \cdot (\text{an embedding of ``aardwolf''}) & | & \text{aardwolf} \\
	\mathbf{E}(\text{who}) \cdot (\text{an embedding of ``aargh''}) & | & \text{aargh} \\
	\vdots \\
	\mathbf{E}(\text{who}) \cdot (\text{an embedding of ``lived''}) & | & \text{lived} \\
	\mathbf{E}(\text{who}) \cdot (\text{an embedding of ``liver''}) & | & \text{liver} \\
	\mathbf{E}(\text{who}) \cdot (\text{an embedding of ``living''}) & | & \text{living} \\
	\vdots \\
	\mathbf{E}(\text{who}) \cdot (\text{an embedding of ``zyme''}) & | & \text{zyme} \\
	\mathbf{E}(\text{who}) \cdot (\text{an embedding of ``zymogen''}) & | & \text{zymogen} \\
	\mathbf{E}(\text{who}) \cdot (\text{an embedding of ``zymosis''}) & | & \text{zymosis} \\
	\mathbf{E}(\text{who}) \cdot (\text{an embedding of ``zzz''}) & | & \text{zzz}
\end{pmatrix}
=
\begin{pmatrix}
    -1.500 & | & \text{aah} \\
    -1.500 & | & \text{aardvark} \\
    -1.500 & | & \text{aardwolf} \\
    -1.500 & | & \text{aargh} \\
    \vdots &   & \vdots \\
     3.344 & | & \text{lived} \\
    -1.500 & | & \text{liver} \\
     1.760 & | & \text{lives} \\
    \vdots &   & \vdots \\
    -1.500 & | & \text{zyme} \\
    -1.500 & | & \text{zymogen} \\
    -1.500 & | & \text{zymosis} \\
    -1.500 & | & \text{zzz}
\end{pmatrix}
$$

### The unembedding matrix

**(Unembedding, continued).** One convenient way of computing the above vector of dot products is by taking the matrix whose rows are the possible words, and multiplying it by the column vector $\mathbf{E}(\text{who})$:
$$
\begin{pmatrix}
	\text{an embedding of ``aah''} \\
	\text{an embedding of ``aardvark''} \\
	\text{an embedding of ``aardwolf''} \\
	\text{an embedding of ``aargh''} \\
	\vdots \\
	\text{an embedding of ``lived''} \\
	\text{an embedding of ``liver''} \\
	\text{an embedding of ``living''} \\
	\vdots \\
	\text{an embedding of ``zyme''} \\
	\text{an embedding of ``zymogen''} \\
	\text{an embedding of ``zymosis''} \\
	\text{an embedding of ``zzz''}
\end{pmatrix}
\mathbf{E}(\text{who})
$$
This matrix is called the *unembedding matrix*, and it is denoted $\mathbf{W}_U$. Like the embedding matrix, it is learned from training.

You might be wondering- why can't we just use the columns of the embedding matrix as the rows of the unembedding matrix? Don't both sets of vectors represent the set of all possible words? Well, yes. And some models indeed do this. But it's a common enough choice to maintain entirely separate matrices for embedding and unembedding. The difference between *representing* a word and *predicting* the next word is enough that doing so can lead to substantial model improvement.

### Softmax

**(Unembedding, continued).** Back to our vector of dot products:
$$
\begin{pmatrix}
    -1.500 & | & \text{aah} \\
    -1.500 & | & \text{aardvark} \\
    -1.500 & | & \text{aardwolf} \\
    -1.500 & | & \text{aargh} \\
    \vdots &   & \vdots \\
     3.344 & | & \text{lived} \\
    -1.500 & | & \text{liver} \\
     1.760 & | & \text{lives} \\
    \vdots &   & \vdots \\
    -1.500 & | & \text{zyme} \\
    -1.500 & | & \text{zymogen} \\
    -1.500 & | & \text{zymosis} \\
    -1.500 & | & \text{zzz}
\end{pmatrix}
$$
We almost have the probability distribution we want! Already, more positive values seem to be correlating with words that should have a higher chance of being predicted, and more negative values seem to to be correlating with words that should have a lower chance of being predicted.

Except, we can't really rightfully refer to the numbers in our distribution as "probabilities", because some of them are negative, which probabilities can't be (there is no notion of a "negative chance" of an event happening), and they don't add up to 1.0 (which probabilities must, since *some* event is guaranteed to happen). Instead, people in the machine learning community call these psuedo-probability sort of numbers by the name "logits".

Fortunately, there is a standard solution for computing logits to probabilities, called the softmax function. The softmax of a vector is pretty simple. Apply the function $x \mapsto e^x$ to each entry, and then divide each result by the sum of the exponentiated entries:
$$
\text{softmax} \left( \begin{pmatrix} x_1 \\ \vdots \\ x_n \end{pmatrix} \right)
= 
\begin{pmatrix} e^{x_1}/\sum_{i = 1}^n e^{x_i} \\ \vdots \\ e^{x_n}/\sum_{i = 1}^n e^{x_i} \end{pmatrix}
$$
We can deduce from the properties of exponents that this softmax function sends the largest logits close to 1 and the smallest logits close to 0, as desired.

Applying softmax to our logit distribution from before, we obtain
$$
\text{softmax}\left(\begin{pmatrix}
    -1.500 & | & \text{aah} \\
    -1.500 & | & \text{aardvark} \\
    -1.500 & | & \text{aardwolf} \\
    -1.500 & | & \text{aargh} \\
    \vdots &   & \vdots \\
     3.344 & | & \text{lived} \\
    -1.500 & | & \text{liver} \\
     1.760 & | & \text{lives} \\
    \vdots &   & \vdots \\
    -1.500 & | & \text{zyme} \\
    -1.500 & | & \text{zymogen} \\
    -1.500 & | & \text{zymosis} \\
    -1.500 & | & \text{zzz}
\end{pmatrix}\right)
=
\begin{pmatrix}
    0.00 & | & \text{aah} \\
    0.00 & | & \text{aardvark} \\
    0.00 & | & \text{aardwolf} \\
    0.00 & | & \text{aargh} \\
    \vdots \\
    0.78 & | & \text{lived} \\
    .00 & | & \text{liver} \\
    0.16 & | & \text{lives} \\
    \vdots \\
    0.00 & | & \text{zyme} \\
    0.00 & | & \text{zymogen} \\
    0.00 & | & \text{zymosis} \\
    0.00 & | & \text{zzz}
\end{pmatrix}
$$

At last, we have our probability distribution.

### Extra: softmax with temperature

Today's chatbot models often expose a parameter called *temperature* that you can play around with. Interestingly, this parameter has to do with softmax. 