---
layout: post
title:  One Letter At a Time
excerpt: "How I used Recurrent Neural Networks to generate text in my style."
comments: true
---

## A little backstory

I used to write a lot. Like a lot. At my high school ([The Bronx High School of Science](https://bxscience.edu/)), we had daily homework in all our subjects. While the ones in mathematics and the sciences did not require any real amount of writing (but rather a ginormous amount of extra-dimensional thinking), the ones in the English and History classes did.

Whether it be to analyze the latest chapter of the book we were reading or create notes of historical accounts of Ben Franklin on the spirit of America, I had my Google Docs opened for writing.

Over the course of three and a half years, my Google Drive was filled with documents containing homework, research papers, essays, poems, Document Based Questions, and more pieces of text. However, after returning back to India, the high school I went to to finish my diploma did not require us to do any such writing. It was mostly mundane and focused on PCM (Physics, Chemistry, Mathematics). The only English writing there was just to write one page three paragraph essays on short articles. Ugh!

My college wasn't much better. It did not even have the boring essay writing freedom I had at the previous school. In the freshman year of college, we were "taught" to write letters (formal and complaint), essays, flowcharts, summaries, idioms and phrases, degrees of comparison, and dialogues. It was as if they assumed we never knew the basic structures of writing before attending those classes.


Naturally, because of the lack of my daily habit of writing, my cognitive abilities related to analytical text generation slowly deteriorated. Since I no longer had deadlines to submit my literary analysis, I switched my focus over to my newfound passion of **Machine Learning** and **Deep Learning**.


## Motivation

But I missed those days of writing. So my next logical step was to turn to Neural Networks of course. At this point, I had recently finished my Deep Learning specialization from [deeplearning.ai](http://deeplearning.ai/) on Coursera.

The last course, [Sequence Models](https://www.coursera.org/learn/nlp-sequence-models), taught us in good detail the basics and intermediate of **Recurrent Neural Networks** for various kinds of sequence models such as *name entity recognition*, *text generation*, *music generation*,  *speech recognition*, *attention models*, and *trigger word detection*.

The text generation exercises we did in the course were related to name generation (dinosaur names in fact). But I decided to expand this in order to be compatible with my project.

I was going to make a **RNN** made of LSTM layers to generate new text in my style, sort of like my brain child.


## Methodology

Of course, this was not an easy task at all. Luckily, there was an amazing blog post written on text generation (on which actually some of the videos in the Sequence Models course took inspiration from) by *Andrej Karpathy* which you can read [here](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

Dr. Karpathy took deliberate effort to clearly explain the effectiveness of Recurrent Neural Networks, why he believes their architecture allows them to learn about the data in a different but phenomenal way, and even shows us some examples. In the post, he generated Shakespeare plays that were completely indiscernible from the original works, but more importantly, he generated *C code* that was syntactically accurate. Having a model learn how to open and close brackets, follow proper indentation style, remember the supposed syntax (arbitrary rules) of an arbitrary set of text is remarkable.

---

I was going to feed my Neural Network a merged text files containing texts that I have written. You can see the entire text file [here](https://github.com/ramanshsharma2806/Alternate-Reality/blob/master/text_data/merged.txt) on the [GitHub](https://github.com/ramanshsharma2806/Alternate-Reality) repository page.

An example from the text is as follows:

> One of the biggest change brought by the American Revolution was the general grudge against the British empire inside of the colonists. It started with the Proclamation Line of 1763 which turned the Ohio River valley into Royal Indian Reserve and forbade colonists from settling west of the Appalachian mountains. The Ohio River valley was the place that the colonists fought the French and Indian war for in the first place. And now, Britain forbids them to live there. This made colonists resent the British empire. When the colonists still settled in the river valley (it was nearly impossible to strictly enforce the proclamation), Ottawa chief Pontiac led a rebellion in western Pennsylvania in May 1763 due to conflicts between colonists and indians in the river valley. Disgusted with Pontiac’s rebellion and the failure of colonial government to keep the people’s interest secured, Paxton Boys of Lancaster killed 6 Indian men, women, and children. The British promised to enforce their agreement with the Indians. After this, Britain passed the Stamp Act in 1765 which imposed a tax on commonly used items like cards, dice, legal documents, newspapers, pamphlets, and advertisements to pay for the debt incurred by Britain during the French and Indian War. Anger and discontent led the colonists to hold the Stamp Act Congress, the Virginia Resolves, boycott British textiles, and violent protests by the Sons of Liberty. The first three were nonviolent protests in hope that the British government would understand their problems because of the Stamp Act. The Stamp Act Congress (first continental congress) even sent a petition for this to Britain. But the Sons of Liberty had different ideas. They tarred and feathered the British officials who tried to enforce the unfair taxes and planted liberty poles in celebration of colonial self government. However, regardless of the way of protest, all colonists strongly resented the colonial government for this direct taxation. However, colonists did not want or were prepared to separate from the British empire yet. But then people read and observed in the 1700s that the British government was not virtuous at all and violated all enlightenment ideas. It didn’t put the public welfare in front of the king’s selfishness, or limited its own power. This was essentially the critical point where the colonists could have been stopped. However, the Boston Massacre of 1770 (British soldiers attacked colonists and killed 5 colonists) and the Boston Tea Party of 1773 (colonists threw tea offboard a ship in protest to tax on it), made the unsure colonists chose their sides. Enraged by such actions, Britain passed the Coercive Acts (Intolerable Acts for the colonists). The Acts shut down all trade ports in Boston which affected trade and commerce in all the colonies, all elected positions such as local judges in Massachusetts were to be appointed by British king, all royal officials received trials in different colony or even in Britain, and increased powers were given to French speaking Catholics in Canadian province of Quebec. As if that was not enough, Britain passed the Quartering Act of 1765, that allowed military commanders to live in houses among the colonists. The colonists also had to pay the rent and give food to the military. These laws were literally openly challenging the colonists to protest and gave a clear message that the British government was not virtuous. In 1774-1775, colonists start talking about breaking apart from the British empire. In 1776, Jefferson wrote the Declaration of Independence which was the definite decision of all the colonies. People could no longer take the oppression of Britain over them. They wanted to drive away from them every British who wanted to see them in “chains of tyranny” to their island of Britain where they would enjoy their slavery and to never let them return to the happy and free land of America (Document A). The colonists have come as far from quietly bearing heavy taxes from Britain to declaring their independence from the British empire. It was without any doubt, a major change brought by the American Revolution.


Note that the text is filled with all kinds of writing semantics: punctuation, brackets, parentheses, numbers, and special characters. This indicates that the model would need to be considerably strong and convoluted if it is to understand the rules of the said semantics.


## Architecture

I trained a two-layer unidirectional LSTM (Long Short Term Memory) Recurrent Neural Network on the data. In each of the two LSTM layers, there were 256 nodes

In the [```merged.txt```](https://github.com/ramanshsharma2806/Alternate-Reality/blob/master/text_data/merged.txt) file, there are a little less than 230,000 characters.


```python
print(f"Length of corpus: {LENGTH}")
print(f"X.shape = {X.shape}")
print(f"Y.shape = {Y.shape}")
print(f"Number of examples: {m}")

# Length of corpus: 229570
# X.shape = (76520, 10, 96)
# Y.shape = (76520, 96)
# Number of examples: 76520

# m can be changed by changing the step variable in the notebook
```

Some of the hyperparameters I used were

```python
learning_rate = 0.01
batch_size = 8192
n_a = 256 # number of LSTM cells
n_L = 2 # two LSTM layers
```

One change I want to bring around in these blog posts is **demystifying the construction of the Neural Networks**. For this reason, I will be including code of how the model was constructed so that if required, someone may refer to this post instead of finding the necessary file on GitHub.

```python
def Ram_Says(Tx, vocab, output_length):
  # network architecture LSTM -> Dropout -> Reshape -> LSTM -> Dropout -> Dense

  # define the initial hidden state a0 and initial cell state c0
  a0 = Input(shape=(output_length,), name='a0')
  c0 = Input(shape=(output_length,), name='c0')
  a = a0
  c = c0

  X = Input(shape=(Tx, len(vocab)), name='X')

  a, _, c = LSTM(units=output_length, activation='tanh', return_state=True, dtype='float32', name=f'lstm_1')(X, [a, c])
  a = Dropout(rate=0.2, name=f'dropout_1')(a)
  a = Reshape((1, output_length), name='reshape_1')(a) # needed after a dropout layer for another LSTM layer
  a = LSTM(units=output_length, activation='tanh', dtype='float32', name=f'lstm_2')(a)
  a = Dropout(rate=0.2, name=f'dropout_2')(a)
  out = Dense(units=len(vocab), activation='softmax', name=f'dense')(a)

  model = Model(inputs=[X, a0, c0], outputs=out, name='Ram')

  return model

"""
creating the model and the summary of it
"""
#====================Creating important variables===============================
n_a = 256 # number of hidden state dimensions for each LSTM cell

a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))
#===============================================================================

model = Ram_Says(Tx=Tx - 1, vocab=vocab, output_length=n_a)
```

**This is what the model architecture looks like:**

![model architecture]({{site.baseurl}}/assets/img/nn_graph.png){: style="max-width: 99%; max-height: 99%;"}



## Results

After tuning the hyperparameters, I was able to achieve a **94% accuracy on the dataset**.

However

> Accuracy is not the most important metric in text generation models

This is because a model can fit with a 99% accuracy on a dataset and still perform poorly. This is the reason why text generation models are hard to evaluate, the most logical way they are assessed is by manually checking how close the model's generated text is resembling the human written text it was trained on.

An example of my model's generated text is

> My name is Ramansh Sharma My name is the public control considered to fut tell overt of a decide the world to life people real and massive really came to congres to be streated it could have ingarding the world many that estabe by nugues and could have been guvern for the Soligition, purched that remamplogr prices repective independence in 1982. Bech of 1980ided to social and enjoyable capting the sent a see that her seperones in Am

Another one is

> Thor is the king Thor is the rare to be some known as the enemonity of Congress, the meint in the Cold War when Chine with them. I was also high domination which dickins of Curness, he will be a beation….ther would reduce the decision of appressed nor the increased communist against controlles. Perretimmert. Itay of nuclas they gave he was a tork the side and very impossing this him as a facutive promest it be a stay a and fack as the anities and the EPand and being people white Sinally sochiets. He did not convered congresity.

And another one

> The The Date-10foce deneed to ave the police of the sent to be some known as “To loon duberd many as it is to New York, to his domestic oil for a long of the weap in Congress she had ease and ensore considerable spending neges and cered to confinced the not on that is a senter aftarity, and a communist confidend ming the enemonity of Congress, complelated to make me to ausheres decart did not prife to meacunion of the Indians, if the stopled. And order to be the same that I am good family defense does money starting ut from America. a recatted to the 1972 Nair ance me to eventle, to stay the cares and even Dieanation, even to geter from the British other trages in Robard on New Yer proved a coffitted them in 1975, up they wanted to raised her free that they will be jasunang but was is the mind end hand of the part endere to the money starting under President during the reader to the side because Opea which states who arm diresten to whith with whith security plannerd


## Discussion

There are several factors which I will be evaluating my model upon. It will be better suited if I do this in a list below.

The model performed well on the following aspects of writing semantics:

- **Punctuation**: The model managed to learn where commas and full stops are to be placed. At most places, it also learned how to open and close brackets. Moreover, entity names are to be capitalized appears to be functional as well.

- **Sentence structure**: This is only weakly accomplished by the model. The generated text seems to have regularly structured sentences, that is to say that verbs come after nouns, adjectives and adverbs are places accordingly, and prepositions are adequately used.

- **Dates**: Wherever the model has used years, it has done so in the proper grammatical way. While this may be considered to be an inherent property, it is important to note that the model learned to place arbitrary numbers in the middle of alphabets maintaining the 4 digits of a year.

Some semantics the model failed to properly learn are:

- **Spelling**: The model appears to be terrible at spelling. Spelling errors exist in every sentence. However, it is noteworthy that the spellings are not so completely destroyed that the word in the context is completely unrecognizable.

- **Overall meaning**: The model fails to make a story. Unlike Dr. Karpathy's model which were able to write Shakespeare plays, which needless to say made proper sense, my model regards each sentence as an individual entity, completely unrelated from each other.

#### NOTE

- It is important to mention that the model was trained on just 230,000 characters. In order to create a proper language modeling program, at least 1 Million characters are recommended.

- The model is a character based model. It looks at 10 characters, and then predicts the 11th one. Then subsequently, shifts one character to the right and repeats. Character models when trained on inadequate data are prone to spelling errors.

## Future

I really enjoyed making this project. Even with some of the shortcomings of the model, I am proud that I was able to make this project in a little over a month. The LSTM layers were especially difficult to construct because their parameters were hard to set.

In order to improve the model, I am going to be trying a stronger model, perhaps 3 bidirectional LSTM layers. I will also search for more of my written text so that the corpus size increases.


## Code


In the GitHub repository, I have presented by code as a Jupyter notebook. But I originally coded in Google colab. **I am providing the live link to the colab for viewing and making it public :)**

Link - [http://bit.ly/alternatereality](http://bit.ly/alternatereality)

Drop your comments below if you'd like to share something, ask any questions, or simply reach out to me at [sharmar@bxscience.edu](mailto:sharmar@bxscience.edu)!

Thank you for reading all the way to the end.
