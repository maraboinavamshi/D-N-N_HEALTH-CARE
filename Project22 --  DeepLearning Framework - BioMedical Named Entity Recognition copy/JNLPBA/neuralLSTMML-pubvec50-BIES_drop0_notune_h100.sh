cmake .
make LSTMCRFMLLabeler
./LSTMCRFMLLabeler -l -train ../jnlpbaBIESdev/jnlpba.train.feats -dev ../jnlpbaBIESdev/jnlpba.dev.feats -test ../jnlpbaBIESdev/jnlpba.test.feats -option example/option.tune.pub.C.drop0.notune.h100 -word /home/lvchen/Documents/embeddings/bio/pub.50.vec -model example/demoLSTMMLCpubvec50BIES.drop0.notune.h100.model
./LSTMCRFMLLabeler -test ../jnlpbaBIESdev/jnlpba.test.feats -model example/demoLSTMMLCpubvec50BIES.drop0.notune.h100.model -output data/testLSTMMLCpubvec50BIES.drop0.notune.h100.output
