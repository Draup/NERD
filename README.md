# NERD

In the realm of text data, two problems are more common than others.
 - Text Classification
 - Named Entity Recognition
 
 The number of use cases that these technique have is quite astonishing. 
 Text classification itself has many different variants. News Classification, Span classification, etc are the common usecases.
 However we come across uncommon usecases every now and then. For example, whether a given piece of text is an address or not.
 
 Similarly Named Entity Recognition is another very common use case. There are great open-source packages for standard entities like Companies, Location, Person Name.
 However, we come across cases where we deal with uncommon entity types. For Example, whether a text contains skill/technology mentioned.
 
 The problem escalates vastly because more often than not we don't have the training data present already. Instead we ourselves have to prepare the data from scratch which is the most of our troubles.
 
 ### NERD is a solution.
 
 NERD provide modules for TextClassification and NER Tagging.
 Using NERD is simple. Collect all the raw text into a list of strings along with unique class types and pass it to the modules and start server.
 
 ```
from NERD.NER import NerTagger
```