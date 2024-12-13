# Melody generator neural network

Experimenting with neural networks for generating music melodies as MIDI files.

Install requirements:

```bash
pipenv install
```

Download data and prepare for training: 

```bash
cd data
make jsb-chorales-16th.json
cd ..
python prepare_data.py
```

Train and generate: 

```bash
python train_model.py # ~3 hours
python generate.py
```
