from agreement_acceptor_decay_rnn import FullGramSentence
import filenames

pvn = FullGramSentence(filenames.deps, prop_train=0.1, output_filename='output_log.txt', len_after_verb=10)

pvn.pipeline(train=False,model="abdrnn_fullGram.pkl", load_data=True,load=True,epochs=10, model_prefix='abdrnn_fullGram', data_name='fullGram', test_size=0)
