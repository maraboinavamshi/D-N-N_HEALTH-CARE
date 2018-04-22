cmake .
make LSTMCRFMLLabeler
./LSTMCRFMLLabeler -l -train ../gene/gene.train.feats -dev ../gene/gene.dev.feats -test ../gene/gene.test.feats -c2_devout data/gene.dev.LSTMML.drop0.pubvec50.h100.notune -c2_testout data/gene.test.LSTMML.drop0.pubvec50.h100.notune -option example/option.tune.pub.C.drop0.notune.h100 -word /home/lvchen/Documents/embeddings/bio/pub.50.vec -model example/demoLSTMMLCpubvec50BIES.drop0.notune.h100.model
./LSTMCRFMLLabeler -test ../gene/gene.test.feats -model example/demoLSTMMLCpubvec50BIES.drop0.notune.h100.model -output data/testLSTMMLCpubvec50BIES.drop0.notune.h100.output
for i in `seq 0 29`;
do
echo "Iter" $i
echo "Dev run!"
java -jar getGMresult.jar data/evalResutl.gene.dev geneGold/train/dev.in data/gene.dev.LSTMML.drop0.pubvec50.h100.notune.$i
perl geneGold/alt_eval.perl -gene geneGold/train/GENE.eval -altgene geneGold/train/ALTGENE.eval -ids geneGold/train/dev.in data/evalResutl.gene.dev
echo "Test run!"
java -jar getGMresult.jar data/evalResutl.gene.test geneGold/test/test.in data/gene.test.LSTMML.drop0.pubvec50.h100.notune.$i
perl geneGold/alt_eval.perl -gene geneGold/test/GENE.eval -altgene geneGold/test/ALTGENE.eval data/evalResutl.gene.test
done
