.. -*- coding: utf-8 -*-

.. ===========
.. Quick-Start
.. ===========
.. A very short introduction into topic models and how to solve them using topicmodel-lib. This document also introduces some basic concepts and conventions.


---------------------------
Topic models
---------------------------
Topic models are probabilistic models of document collections that use latent variables to encode recurring patterns of word use (Blei, 2012). Topic modeling algorithms are inference algorithms; they uncover a set of patterns that pervade a collection and represent each document according to how it exhibits them. These patterns tend to be thematically coherent, which is why the models are called "topic models." Topic models are used for both descriptive tasks, such as to build thematic navigators of large collections of documents, and for predictive tasks, such as to aid document classification. Topic models have been extended and applied in many domains

`Latent Dirichlet Allocation (LDA)`_ is the simplest topic model, LDA is a generative probabilistic model for collections of discrete data such as text corpora.

.. _Latent Dirichlet Allocation (LDA): ./LatentDirichletAllocation.rst

Large-scale learning
====================
Modern data analysis requires computation with massive data. These problems illustrate some of the challenges to modern data analysis. Our data are complex and high-dimensional; we have assumptions to make from science, intuition, or other data analyses that involve structures we believe exist in the data but that we cannot directly observe; and finally, our data sets are large, possibly even arriving in a never-ending stream. We deploy this library to computing with graphical models that are appropriate for massive datasets, data that might not fit in memory or even be stored locally. This is an efficient tool for learning LDA at large scales


Learning methods for LDA
========================
To learn LDA at large-scale, a good and efficient approach is stochastic learning (online/streaming methods). The learning process includes 2 main steps:

- Inference for individual document: infer to find out the **hidden local variables**: topic proportion :math:`\theta` or topic indices **z**. To deal with this, we can estimate directly or estimate their distribution P(:math:`\theta` | :math:`\gamma`), P(**z** | :math:`\phi`) (:math:`\gamma`, :math:`phi` called "variational parameters"). 
- Update **global variable** in a stochastic way to find out directly topics :math:`\beta` or we can estimate topics by finding out its distribution P(:math:`\beta` | :math:`\lambda`) (estimating variational parameter :math:`\lambda`). Global variable here maybe :math:`\beta` or :math:`\lambda` depend on each stochastic methods.

Indeed, this phase is as same as training step in machine learning. 

---------------------------------------------------------
Training Data
---------------------------------------------------------

Corpus
======
A corpus is a collection of digital documents. This collection is the input to topicmodel-lib from which it will infer the structure of the documents, their topics, topic proportions, etc. The latent structure inferred from the corpus can later be used to assign topics to new documents which were not present in the training corpus. For this reason, we also refer to this collection as the training corpus. No human intervention (such as tagging the documents by hand) is required - the topic classification is unsupervised.

Data Format
===========

Our framework is support for 3 input format:

- Corpus with **raw text** format:
  
  ::

    raw_corpus = ["Human machine interface for lab abc computer applications",
                  "A survey of user opinion of computer system response time",
                  "The EPS user interface management system",
                  "System and human system engineering testing of EPS",              
                  "Relation of user perceived response time to error measurement",
                  "The generation of random binary unordered trees",
                  "The intersection graph of paths in trees",
                  "Graph minors IV Widths of trees and well quasi ordering",
                  "Graph minors A survey"]

  The raw corpus must be stored in a file. Each document is placed in 2 pair tag <DOC></DOC> and <TEXT></TEXT> as follow:

  .. image:: ../images/format.PNG

  You can see `ap corpus`_ for example

  .. _ap corpus: https://github.com/TruongKhang/documentation/blob/master/examples/ap/data/ap_infer_raw.txt

- Term-frequency format (**tf**):

  The implementations only support reading data type in LDA. Please refer to the following site for instructions: http://www.cs.columbia.edu/~blei/lda-c/
  Under LDA, the words of each document are assumed exchangeable.  Thus, each document is succinctly represented as a sparse vector of word counts. The data is a file where each line is of the form:

  `[N] [term_1]:[count] [term_2]:[count] ...  [term_N]:[count]`

  where [N] is the number of unique terms in the document, and the [count] associated with each term is how many times that term appeared in the document.  Note that [term_i] is an integer which indexes the term (index of that term in file vocabulary); it is not a string.

  For example, with corpus as raw_corpus above and file vocabulary is:

  ::

       0. "human"
       1. "machine"
       2. "interface"
       3. "lab"
       4. "abc"
       5. "computer"
       6. "applications"
       7. "survey"
       8. "user"
       9. "opinion"
       10. "system"
       11. "response"
       12. "time"
       13. "eps"
       14. "management"
       15. "engineering"
       16. "testing"
       17. "relation"
       18. "perceived"
       19. "error"
       20. "measurement"
       21. "generation"
       22. "random"
       23. "binary"
       24. "unordered"
       25. "trees"
       26. "intersection"
       27. "graph"
       28. "paths"
       29. "minors"
       30. "widths"
       31. "quasi"
       32. "ordering"

  The **tf** format of corpus will be:
     
  ::

       7 0:1 1:1 2:1 3:1 4:1 5:1 6:1 
       7 7:1 8:1 9:1 5:1 10:1 11:1 12:1 
       5 13:1 8:1 2:1 14:1 10:1 
       5 10:2 0:1 15:1 16:1 13:1 
       7 17:1 8:1 18:1 11:1 12:1 19:1 20:1 
       5 21:1 22:1 23:1 24:1 25:1 
       4 26:1 27:1 28:1 25:1 
       6 27:1 29:1 30:1 25:1 31:1 32:1 
       3 27:1 29:1 7:1 

- Term-sequence format (**sq**):

  Each document is represented by a sequence of token as follow
    
  `[token_1] [token_2] [token_3]....`

  [token_i] also is index of that token in vocabulary file, not a string. (maybe exist that [token_i] = [token_j]) 
  The **sq** format of the corpus above will be:

  ::

       0 1 2 3 4 5 6 
       7 8 9 5 10 11 12 
       13 8 2 14 10 
       10 0 10 15 16 13 
       17 8 18 11 12 19 20 
       21 22 23 24 25 
       26 27 28 25 
       27 29 30 25 31 32 
       27 29 7 

--------------------------
Guide to the learning step
--------------------------

In this phase, the main task is find out the global variable (topics) - in this project, we call it named `model` for simple. We designed the state-of-the-art methods (online/streaming learning): `Online VB`_, `Online CVB0`_, `Online CGS`_, `Online OPE`_, `Online FW`_, `Streaming VB`_, `Streaming OPE`_, `Streaming FW`_, `ML-OPE`_, `ML-CGS`_, `ML-FW`_

.. _Online VB: ./methods/online_vb.rst
.. _Online CVB0: ./methods/online_cvb0.rst
.. _Online CGS: ./methods/online_cgs.rst
.. _Online OPE: ./methods/online_ope.rst
.. _Online FW: ./methods/online_fw.rst
.. _Streaming VB: ./methods/streaming_vb.rst
.. _Streaming OPE: ./methods/streaming_ope.rst
.. _Streaming FW: ./methods/streaming_fw.rst
.. _ML-OPE: ./methods/ml_ope.rst
.. _ML-CGS: ./methods/ml_cgs.rst
.. _ML-FW: ./methods/ml_fw.rst

All of this methods are used in the same way. So, in this guide, we'll demo with a specific method such as Online VB. This method is proposed by Hoffman-2010, using stochastic variational inference

Data Preparation
================
Make sure that your training data must be stored in a text file and abide by the `format`_: **tf**, **sq** or **raw text**

.. _format: ./quick_start.rst#data-format

We also support the `preprocessing`_ module to work with the raw text format, you can convert to the tf or sq format. But if you don't want to use it, it's OK because we integrated that work in class ``DataSet``. Therefore, the first thing you need to do is create an object ``DataSet``

::

  from tmlib.datasets import DataSet
  # data_path is the path of file contains your training data
  data = DataSet(data_path, batch_size=5000, passes=5, shuffle_every=2)

The statement above is used when `data_path` is the raw text format. If your training file is the tf or sq format. You need to add an argument is the vocabulary file of the corpus as follow:

::

  # vocab_file is the path of file vocabulary of corpus
  data = DataSet(data_path, batch_size=5000, passes=5, shuffle_every=2, vocab_file=vocab_file)

The parameters **batch_size**, **passes**, **shuffle_every** you can see in `documentation here`_

.. _documentation here: ./methods/online_vb.rst
.. _preprocessing: ./preprocessing.rst

Learning
========

First, we need to create an object ``OnlineVB``:

::

  from tmlib.lda import OnlineVB
  onl_vb = OnlineVB(data, num_topics=100, alpha=0.01, eta=0.01, tau0=1.0, kappa=0.9)

``data`` is the object which created above. Parameter **num_topics** number of requested latent topics to be extracted from the training corpus. **alpha**, **eta** are hyperparameters of LDA model that affect sparsity of the topic proportions (:math:`\theta`) and topic-word (:math:`\beta`) distributions. **tau0**, **kappa** are learning parameters which are used in the update global variable step (same meaning as learning rate in the gradient descent optimization)

Start learning by call function **learn_model**:

::

  model = onl_vb.learn_model()

The returned result is an object `LdaModel`_

.. _LdaModel: ./ldamodel.rst

You can also save the model (:math:`\beta` or :math:`\lambda`) or some statistics such as: learning time, sparsity of document in the learning process

::

  model = onl_vb.learn_model(save_statistic=True, save_model_every=2, compute_sparsity_every=2, save_top_words_every=2, num_top_words=10, model_folder='models')

The result is saved in folder `models`. More detail about this parameters, read `here`_

.. _here: ./methods/online_vb.rst

One more thing, the topic proportions (:math:`\theta`) of each document in the corpus can be saved in a file ``.h5``. This work is necessary for `visualization`_ module but it'll make the learning time slower. So, be careful when using it!

::

  # for example: path_of_h5_file = 'models/database.h5'
  model = onl_vb.learn_model(save_topic_proportion=path_of_h5_file)

.. _visualization: ./visualization.rst



---------------------------
Inference for new documents
---------------------------
After learning phase, you have the `model` - topics (:math:`\beta` or :math:`\lambda`). You want to infer for some documents to find out what topics these documents are related to. We need to estimate topic-proportions :math:`\theta`
