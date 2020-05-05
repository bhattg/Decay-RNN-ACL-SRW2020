from agreement_acceptor import FullGramSentence
import filenames

pvn = FullGramSentence(filenames.deps, prop_train=0.3, output_filename='output_log.txt', len_after_verb=10, embedding_size=50, hidden_dim = 50)
pvn.pipeline(train=False, load_data=True,load=True,model='lstm_fullGram.pkl',epochs=10, model_prefix='lstm_fullGram',test_size=0, data_name='fullGram')
#testing should be done with train=False and test_size=0
