# Slacked Decay RNN 

# Setups

* Note that, it is desirable to have an environment variable named 'RNN_AGREEMENT_ROOT' which saves the location of the folder (or your master working directory where you will save the dataset!) 

* The training data (overall data) is available [here](http://tallinzen.net/media/rnn_agreement/agr_50_mostcommon_10K.tsv.gz). Save this to a folder named data. 

* Make a folder name logs in the running directory. 

* To run the tests, you will need to run run_plus.py file. Please see the edit details below:

# Full Grammaticality test experiment:

It is often desirable to save the dataset before training so that to have uniformity across the testing examples for different runs. Therefore, when running any model for the first time, please send the following argument in pvn.pipeline method:

```
save_data=True
```

prop_train is proportion of dataset used for training:

```
from agreement_acceptor_decay_rnn import FullGramSentence
import filenames

pvn = FullGramSentence(filenames.deps, prop_train=0.1, output_filename='output_log.txt', len_after_verb=10, embedding_size=50, hidden_dim = 50)

pvn.pipeline(train=True,model="sdrnn_fullGram.pkl", load_data=True,load=False,epochs=10, model_prefix='sdrnn_fullGram', 
			data_name='fullGram', test_size=7000)

```

Note that, if you had saved the dataset, then you can keep the load_data = True, otherwise set it False. While training, for verbose it is recommended to have the test_size set at 7000. The trained model will be saved as 'model_prefix.pt'. 

All verbose will be stored in logs folder. There will be output.txt which will have the saved logs and the testing results. 


To test the model, set the testing size to 0. This will take the remaining dataset!

```

from agreement_acceptor_decay_rnn import FullGramSentence
import filenames

pvn = FullGramSentence(filenames.deps, prop_train=0.1, output_filename='output_log.txt', len_after_verb=10, embedding_size=50, hidden_dim = 50)

pvn.pipeline(train=False,model="sdrnn_fullGram.pkl", load_data=True,load=True,epochs=10, model_prefix='sdrnn_fullGram', 
			data_name='fullGram', test_size=0)
```

Results will be demarcated in terms of number of intervening nouns. 

# Number Prediction Tests


It is often desirable to save the dataset before training so that to have uniformity across the testing examples for different runs. Therefore, when running any model for the first time, please send the following argument in pvn.pipeline method:

```
save_data=True, load_data=False
```

prop_train is proportion of dataset used for training:

```

from agreement_acceptor_decay_rnn import PredictVerbNumber
import filenames

pvn = PredictVerbNumber(filenames.deps, prop_train=0.1, output_filename='output_log.txt', len_after_verb=10, embedding_size=50, hidden_dim = 50)

pvn.pipeline(train=True,model="sdrnn_pvn.pkl", load_data=True,load=False,epochs=10, model_prefix='sdrnn_pvn', 
			data_name='number_pred', test_size=7000)
```

Note that, if you had saved the dataset, then you can keep the load_data = True, otherwise set it False. While training, for verbose it is recommended to have the test_size set at 7000. The trained model will be saved as 'model_prefix.pt'. 

All verbose will be stored in logs folder. There will be output.txt which will have the saved logs and the testing results. 


To test the model, set the testing size to 0. This will take the remaining dataset!

```

from agreement_acceptor_decay_rnn import PredictVerbNumber
import filenames

pvn = PredictVerbNumber(filenames.deps, prop_train=0.1, output_filename='output_log.txt', len_after_verb=10, embedding_size=50, hidden_dim = 50)

pvn.pipeline(train=False,model="sdrnn_pvn.pkl", load_data=True,load=True,epochs=10, model_prefix='sdrnn_pvn', 
			data_name='number_pred', test_size=0)

```

Results will be demarcated in terms of number of intervening nouns and num_attractors . 
