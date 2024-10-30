data: maestro-v3.0.0 jsb-chorales-16th.json lmd_clean

maestro-v3.0.0: 
	curl https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip -O
	unzip maestro-v3.0.0-midi.zip

jsb-chorales-16th.json:
	curl https://raw.githubusercontent.com/czhuang/JSB-Chorales-dataset/refs/heads/master/jsb-chorales-16th.json -O 

lmd_clean:
	curl http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz -O
	tar -xvzf clean_midi.tar.gz
	mv clean_midi lmd_clean
