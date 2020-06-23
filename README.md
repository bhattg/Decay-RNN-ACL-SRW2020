# Decay-RNN-ACL-SRW2020
This is an official pytorch implementation for the experiments described in the paper - "How much complexity does an RNN architecture need to learn syntax-sensitive dependencies?", Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop

To download the generalization set templates, please see [here](https://drive.google.com/file/d/13Q_zUz5fZxYwGuo_-ZbS20HHUkZEHPzl/view?usp=sharing)

For tests mentioned in Section 6.1, 6.2 and 6.3, please look at the individual model folder. For LM, please choose the Language Modeling folder. 

Dependencies:

* numpy
* pytorch : >= 1.1 
* inflect
* pandas
* statsmodels

Suggested : Install Anaconda (python library manager). Then install inflect, pytorch
and any other libraries as needed.

If you find our work useful, then please consider citing us using:
```
@inproceedings{bhatt-etal-2020-much,
    title = "How much complexity does an {RNN} architecture need to learn syntax-sensitive dependencies?",
    author = "Bhatt, Gantavya  and
      Bansal, Hritik  and
      Singh, Rishubh  and
      Agarwal, Sumeet",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-srw.33",
    pages = "244--254",
    abstract = "Long short-term memory (LSTM) networks and their variants are capable of encapsulating long-range dependencies, which is evident from their performance on a variety of linguistic tasks. On the other hand, simple recurrent networks (SRNs), which appear more biologically grounded in terms of synaptic connections, have generally been less successful at capturing long-range dependencies as well as the loci of grammatical errors in an unsupervised setting. In this paper, we seek to develop models that bridge the gap between biological plausibility and linguistic competence. We propose a new architecture, the Decay RNN, which incorporates the decaying nature of neuronal activations and models the excitatory and inhibitory connections in a population of neurons. Besides its biological inspiration, our model also shows competitive performance relative to LSTMs on subject-verb agreement, sentence grammaticality, and language modeling tasks. These results provide some pointers towards probing the nature of the inductive biases required for RNN architectures to model linguistic phenomena successfully.",
}
```
