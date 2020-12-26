# Quantum-inspired Neural Network

Focus on quantum-inspired representation learning and text classification. 

## Credits

This repository is supported by [Benyou Wang](https://wabyking.github.io/old), [Qiuchi Li](https://qiuchili.github.io/), Donghao Zhao, [Chen Zhang](https://genezc.github.io/) and [Amit](https://amitkumarj441.github.io/).

## Requirements

- PyTorch > 1.0.0
- Tensorflow
- NLTK

## Implemented Models

- QDNN (Global Mixture)
- LocalMixtureNN (Local Mixture)
- MLLM 
- SentiQDNN
- FastText
- CNN
- LSTM
## Data
- Download the data at https://www.dropbox.com/s/zpu2wx5bq54agk8/data.zip?dl=0.
- Put the downloaded data folder in the root directory.

## Usage

- Set up a configuration file and put it into directory **/config**. Details about how to configurate could be referred to in **/config/config_qdnn.ini**.
    - First, you should make it clear which network you will use, i.e. *network_type*.
    - Second, if you are using a network utilizing sentiment lexicon as external information such as **SentiQDNN**, we should set *strategy* in configuration file to *multi-task*.
    - Plus, you need specify other details like *dataset_name*, *sentiment_dic_file*, *batch_size*, *lr*, etc.
- Modify the configuration file you will use in **run.py** as
```python
config_file = 'config/config_*.ini'
```
- Type in command line
```python
python run.py 
```

## Citations

If you find our work is useful, please kindly cite our papers:

```bibtex
@misc{li2018quantuminspired,
    title={Quantum-inspired Complex Word Embedding},
    author={Qiuchi Li and Sagar Uprety and Benyou Wang and Dawei Song},
    year={2018},
    eprint={1805.11351},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
@misc{wang2019semantic,
    title={Semantic Hilbert Space for Text Representation Learning},
    author={Benyou Wang and Qiuchi Li and Massimo Melucci and Dawei Song},
    year={2019},
    eprint={1902.09802},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
