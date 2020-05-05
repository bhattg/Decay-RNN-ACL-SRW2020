from agreement_acceptor_eirnn_seq import FullGramSentence
import filenames

pvn = FullGramSentence(filenames.deps, prop_train=0.1, output_filename='output_log.txt', len_after_verb=10, hidden_dim=50, embedding_size=50)

pvn.pipeline(train=False,model="eirnn_seq.pkl", load_data=True,load=True,epochs=10, model_prefix='eirnn_seq', data_name='fullGram', test_size=0, test_external=False, load_external=False, external_file=None)
