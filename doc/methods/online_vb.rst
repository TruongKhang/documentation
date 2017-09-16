=========
Online VB
=========

Online VB stand for Online Variational Bayes

----------------------------------
class OnlineVB
----------------------------------

::

  tmlib.lda.OnlineVB (data=None, num_topics=100, alpha=0.01, eta=0.01, tau0=1.0, kappa=0.9, conv_infer=0.0001, iter_infer=50, lda_model=None)

This class inherits super-class LdaLearning. This used for learning LDA by Online-VB method.

Parameters
==========

- **num_terms**: int,

  number words of vocabulary file

- **num_topics**: int, default: 100

  number of topics of model.

- **alpha**: float, default: 0.01

  parameter :math:`\alpha` of model LDA

- **eta** (:math:`\eta`): float, default: 0.01 

- **tau0** (:math:`\tau_{0}`): float, default: 1.0

- **kappa** (:math:`\kappa`): float, default: 0.9

- **conv_infer**: float, default: 0.0001

  The relative improvement of the lower bound on likelihood of VB inference. If If bound hasn't changed much, the inference will be stopped

- **iter_infer**: int, default: 50.

  number of iterations to do inference

- **lda_model**: object of class LdaModel, default: None.

  If this is None value, it will be initialized and become a new object. If not, it will be the model learned previously

Attributes
==========

- **num_terms**: int,

- **num_topics**: int, 

- **alpha**: float, 

- **eta** (:math:`\eta`): float, 

- **tau0** (:math:`\tau_{0}`): float, 

- **kappa** (:math:`\kappa`): float, 

- **conv_infer**: float, 

- **iter_infer**: int,

- **lda_model**: object of class LdaModel

- **_Elogbeta**: float,

  This is expectation of random variable :math:`\beta` (topics of model).

- **_expElogbeta**: float, this is equal exp(**_Elogbeta**)

Methods
=======

- __init__ (*num_terms, num_topics=100, alpha=0.01, eta=0.01, tau0=1.0, kappa=0.9, conv_infer=0.0001, iter_infer=50, lda_model=None*)

- **static_online** (*wordids, wordcts*)

  Excute the learning algorithm, includes: inference for individual document and update :math:`\lambda`. 2 parameters *wordids*, *wordcts* represent for term-frequency format of mini-batch

  **Return**: tuple (time of E-step, time of M-step, gamma). gamma (:math:`\gamma`) is variational parameter of :math:`\theta`

- **e_step** (*wordids, wordcts*)

  Do inference for indivial document (E-step)

  **Return**: tuple (gamma, sstats), where, sstats is the sufficient statistics for the M-step

- **update_lambda** (*batch_size, sstats*)

  Update :math:`\lambda` by stochastic way. 

- **learn_model** (*data, save_model_every=0, compute_sparsity_every=0, save_statistic=False, save_top_words_every=0, num_top_words=20, model_folder='model'*)

  Inheritted method
  
  see class LdaLearning

- **infer_new_docs** (*new_corpus*)

  Inheritted method
  
  see class LdaLearning
