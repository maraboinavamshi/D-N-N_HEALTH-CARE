cmake .
make LSTMCRFMLLabeler
./LSTMCRFMLLabeler -l -train ../jnlpbaBIESdev/jnlpba.train.feats -dev ../jnlpbaBIESdev/jnlpba.dev.feats -test ../jnlpbaBIESdev/jnlpba.test.feats -option example/option.tune.pub.C.drop0.notune.h100 -word /home/lvchen/Documents/embeddings/senna/google.txt -model example/demoLSTMMLCgoogle300BIES.drop0.notune.h100.model
./LSTMCRFMLLabeler -test ../jnlpbaBIESdev/jnlpba.test.feats -model example/demoLSTMMLCgoogle300BIES.drop0.notune.h100.model -output data/testLSTMMLCgoogle300BIES.drop0.notune.h100.output
