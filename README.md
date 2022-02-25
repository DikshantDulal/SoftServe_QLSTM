# SoftServe_QLSTM
## Introduction

Stock price prediction is one of the most rewarding problems in modern finance, where the accurate forecasting of future stock prices can yield significant profit and reduce the risks. LSTM (Long Short-Term Memory) is a recurrent Neural Network (RNN) applicable to a broad range of problems aiming to analyze or classify sequential data. Therefore, many people have used LSTM to predict the future stock price based on the historical data sequences with great success.

On the other hand, recent studies have shown that the LSTM's efficiency and trainability can be improved by replacing some of the layers in the LSTM with variational quantum layers, thus making the classical LSTM a quantum-classical hybrid model, which we will call QLSTM for Quantum LSTM. A recent study done by Samuel Yen-Chi Chen, Shinjae Yoo, and Yao-Lung L. Fang shows that QLSTM offers better trainability compared to its classical counterpart as it proved to learn significantly more information after the first training epoch than its classical counterpart and learned the local features better, all while having a comparable number of parameters. Inspired by these recent results, we proceed to test this variational quantum-classical hybrid neural network technique on stock price predictions.

In this submission, we provide a proof of concept that QLSTM can be used to predict stock prices on a particular stock (Merck and Co. Inc (MRK)), and that the results of its prediction is comparable, and perhaps even arguably better in terms of loss, to its classical counter part. We demonstrate that it has a higher trainability as the loss decreases faster with the QLSTM per epoch, and that the results were achieved using much less parameters in QLSTM as compared to the classical LSTM.

## Submission

For the implementation of QLSTM and its comparison to the classical LSTM, please refer to the notebook <b> Stock Prediction Draft 3 </b>. It provides a proof of concept that QLSTM can be used to perform stock price predictions by training the model to predict the stock prices of Merck and Co. Inc (MRK), and that it has comparable results for its prediction to its classical counterpart while requiring much fewer parameters. Furthermore, we show that trainability of QLSTM is arguably better than LSTM.

For more in depth view of our data collection process, please refer to the notebook <b> Data Collection </b>. It lists our data collection decisions that provided us a relevant csv file to train the QLSTM and LSTM for the above submission.

Disclaimer: As of right now, we are testing QLSTM on the Pennylane simulator, but the technique is technology agnostic and can work on any gate based device, be it IBM's Qiskit or AWS Braket's gate based devices. We are planning to test its viability on NISQ era devices soon. 

## Outline of GitHub

- <b> Stock Prediction Draft 3 </b>: Main notebook for this GitHub. Provides a proof of concept that QLSTM can be used to great effect for stock prices forecasting for MRK, and compares it to the classical counterpart while also performing the complexity analysis.
- <b> Data Collection </b>: Describes the data collection process, and the relevant decisions made during that process.
- <b> Factory.py </b>: Has the main classes used for <b> Stock Prediction Draft 3 </b>, including the LSTM and QLSTM classes.
- <b> helper.py </b>: Has useful functions for data collection
- <b> dataset_MRK_prediction.csv </b>: Main csv used for the <b> Stock Prediction Draft 3 </b>
- <b> MRK.csv </b>: MRK data, used in <b> Data Collection </b>
- Other files are just images used in descriptions in <b> Stock Prediction Draft 3 </b>

