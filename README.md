# NLP_Yanan

* `DistilBert` The instruction of how to realize distilbert

  * Working Mechanism of Distillation and Distiller
  * Distillation Models 
    * DistilBert (Main)
    * BERT-PKD
    * Distilled BiLSTM
    * MobileBert
    * TinyBert
  * Loss Structure
    * Loss_ce
    * Loss_mlm
    * Loss_cos
    * Loss_kl
    * Loss_mse
  * Data Augmentations

* `BertStructure`

  * **Transformers_Bert.md**  
  	The working mechanism for hugging face transformers and bert.
  	
  * **Simple_Transformer.md**  
  	How does the simple transformer work? And the downstream task like NER, Translation, Generation, Representation

  * **DNN2GP.md**  
  	Change Deep Neural Network to Gaussian Process to realize the linear simulation.

  * **Data_Stream.md**  
    How the data changes in the complete process for NLP

* `TranslateEnCn_torch`
  * How to realize English and Chinese Translation by PyTorch
  * Without using other built-in libraries, realize the Encoder and the Decoder for Transformers using PyTorch.

* `TimeSeriesForecast`

  * Realize a time series forecast using some Time Series and NLP methods.

  - one for advanced methods: **LSTM, Neural Networks(based on PyTorch), Prophet, ARIMA**
  - one for sklearn libraries for **["knn", "random forest", "adaboost", "gbrt", "support vector regression", "lasso", "decision tree", "linear", "ridge", "elastic-net"]**

* `HornorOfKings`
  * Perform statistics and analysis for the mobile game Honor of Kings,
  * Train a neural network model to extract game events from raw data. 
  * Use time-series algorithms to predict future player intent and game events.
  * Build an intelligent commentary algorithm by NLG, and match automatic commentary for game anchors.

* `Word2Vec`
  * The source code of Migolov Word2Vec model based on C language
  * Add Chinese Annotation.
  * There will be future addition for more detailed annotatio.