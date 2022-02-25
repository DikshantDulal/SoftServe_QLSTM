# SoftServe_QLSTM
## Introduction

Stock price prediction is one of the most rewarding problems in modern finance, where the accurate forecasting of future stock prices can yield significant profit and reduce the risks. LSTM (Long Short-Term Memory) is a recurrent Neural Network (RNN) applicable to a broad range of problems aiming to analyze or classify sequential data. Therefore, many people have used LSTM to predict the future stock price based on the historical data sequences with great success.

On the other hand, recent studies have shown that the LSTM's efficiency and trainability can be improved by replacing some of the layers in the LSTM with variational quantum layers, thus making it a quantum-classical hybrid model of LSTM which we will call QLSTM for Quantum LSTM. In the study done by Samuel Yen-Chi Chen, Shinjae Yoo, and Yao-Lung L. Fang, they show that QLSTM offers better trainability compared to its classical counterpart as it proved to learn significantly more information after the first training epoch than its classical counterpart, learnt the local features better, all while having a comparable number of parameters. Inspired by these recent results, we proceed to test this variational quantum-classical hybrid neural network technique on stock price predictions.

## Submission

For the implementation of QLSTM and its comparison to the classical LSTM, please look at the notebook <b> Stock Prediction Draft 2 </b>. It provides a proof of concept that QLSTM can be used to perform stock price predictions by training the model to predict the stock prices of Merck and Co. Inc (MRK), and that it has comparable results for its prediction to its classical counterpart while needing much fewer parameters. In addition, we show that its trainability is arguably better.

For more in depth view of how we collected the data, please look at the notebook <b> Data Collection </b>. It lists the data collection decisions that we made that provided us a relevant csv file to train the QLSTM and LSTM for the above submission.

## Outline of GitHub

- <b> Stock Prediction Draft 2 </b>: Main notebook for this GitHub. Provides a proof of concept that QLSTM can be used to great effect for stock prices forecasting for MRK, and compares it to the classical counterpart while also performing the complexity analysis.
- <b> Data Collection </b>: Describes the data collection process, and the relevant decisions made during that process.
- <b> Factory.py </b>: Has the main classes used for <b> Stock Prediction Draft 2 </b>, including the LSTM and QLSTM classes.
- <b> helper.py </b>: Has useful functions for data collection
- <b> dataset_MRK_prediction.csv </b>: Main csv used for the <b> Stock Prediction Draft 2 </b>

