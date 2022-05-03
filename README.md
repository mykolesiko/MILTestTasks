```
│   README.md
│   requirements.txt
│   text
│   
├───data
├───eda
│       layout.ipynb
│       
├───models
│       best.pth
│       
├───scripts
│       download_dataset.sh
│       
└───src
    │   constants.py
    │   predict.py
    │   train.py
    │   
    ├───datasets
    │       datasets.py
    │       __init__.py
    │       
    ├───model
    │       model.py
    │       __init__.py
    │       
    ├───transforms
    │       transforms.py
    │       __init__.py
    │       
    └───utils
            losses.py
            __init__.py
```

   
Assuming that current directory is root of git repository you should make next steps

1) Create and activate new environment:
```
    Installation (for Windows):
    python -m venv .venv
    .venv\Scripts\activate.bat
    pip install -r requirements.txt

    Installation (for Linux):
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
```
2) download dataset and best model
```               
    ./scripts/download_dataset.sh
    ./scripts/download_best_model.sh

```
3) try to predict test data (test.json and folder data should be) 
```
    python predict.py -dd  ./data -md ./models -mn best.pth

```
3) try to train model (parameters of training coulf be changed, use help)
```
    python train.py  -dd ./data -md ./models -mn best_trained.pth --epochs 40
```

Link to notebook
https://colab.research.google.com/drive/1igQ1D3VqowJpzjANe8hjG5_4OlHwSltQ?usp=sharing



