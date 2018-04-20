cmake .
make LSTMCRFMLLabeler
./LSTMCRFMLLabeler -l -train ../jnlpbaBIESdev/jnlpba.train.feats -dev ../jnlpbaBIESdev/jnlpba.dev.feats -test ../jnlpbaBIESdev/jnlpba.test.feats -option example/option.tune.pub.C.drop0.notune.h100 -model example/demoLSTMMLCrandom300BIES.drop0.notune.h100.model
./LSTMCRFMLLabeler -test ../jnlpbaBIESdev/jnlpba.test.feats -model example/demoLSTMMLCrandom300BIES.drop0.notune.h100.model -output data/testLSTMMLCrandom300BIES.drop0.notune.h100.output
