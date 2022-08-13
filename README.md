# Transformer Encoder(Vision Transformer) on Bitcoin Price Prediction
## Author
**Name:** Jeng-Chung, Lien<br/>
**Email:** masa67890@gmail.com
## Summary
This project demonstrates how a Transformer Encoder is implemented to perform on Bitcoin Price Prediction. The Transformer Encoder here is from the model Vision Transformer[[1]](#reference_anchor1), this model is introduce from the google research and brain team. The model here is used from the code that I recently implemented the Vision Transformer model in tensorflow[[2]](#reference_anchor2) with a bit of modification so it works on continuous time series data. The data is from a website investing where the unit of the data is in a day[[3]](#reference_anchor3), which I used the data from 29 Aug 2017 to 17 Jun 2022.
## Modification on Vision Transformer
Please look into my self implementation of Vision Transformer repository[[2]](#reference_anchor2) for the details of how the model works.
1. The "Linear Projections of Flatten Patches" mentioned in the Vision Transformer paper[[1]](#reference_anchor1) here now will simply just represent as a learnable embedding, taking the features of a day and turing it into a vector representation(transforming different days with same type of features into the same embedding space). Which is a dense layer with linear activation.
2. The "Class Token" mentioned in the Vision Transformer paper[[1]](#reference_anchor1) here now will be the token that represents the next day token(next day of the input of the last date), so now the MLP head would produce the result of the next day.
3. The "MLP head" mentioned in the Vision Transformer paper[[1]](#reference_anchor1) now doesn't ouput with the activation of sigmoid or softmax, it is output using linear activation for continuous values.
## Results
The data is from 29 Aug 2017 to 17 Jun 2022 in the unit of a day, which includes the price, open, high, low and volume. The input takes 7 days of data and output the next day as prediction. There are two models one taking the first 90% of the timeline as training the following 10% as testing, and the other on is taking 80% of the timeline as training, 10% of the following timeline as validation and 10% of the following timeline as testing.<br/>
1. Below shows the first model that uses 90% of the timeline as training. The orange line is the training data and the green line is the testing data.
  <br/>**90% Model ViT_7_32_V1 Figure:**<br/>
  ![ViT_7_32_V1.jpeg](Figures/ViT_7_32_V1.jpeg)
2. Below shows the second model that uses 80% of the timeline as training. The orange line is the training data, the green line is the validation data and the red line is the testing data.
  <br/>**80% Model ViT_7_32_Test Figure:**<br/>
  ![ViT_7_32_Test.jpeg](Figures/ViT_7_32_Test.jpeg)
## Short Conclusion
We can see from the results that this model quite acuratly predicts the next day's price using 7 days as input. However, this model has a weakpoint. When it deals with multiple outputs, outputing the next 3 days as example it wouldn't work well. By adding additional n tokens representing the next n days wouldn't work well, it is possible that the learable tokens all pick up the same information so it confuses the model. Here the solution would be using the whole transformer model[[4]](#reference_anchor4), or just the transformer decoder like GPT-2[[5]](#reference_anchor5). Since the transformer decoder has a mask component in the multi-head self-attention, so it takes the input and output one by one and unmask each iteration to let it consider the sequence of the coming data more so it doesn't confuses the model.
## Reference
<a name="reference_anchor1"></a>[1] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale”, Jun. 2021. [Online]. Available: [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929) <br/>
<a name="reference_anchor2"></a>[2] Cifar-10 Dataset Classification with Vision Transformer, Github: [https://github.com/cloudstrife1117/VisionTransformer](https://github.com/cloudstrife1117/VisionTransformer) <br/>
<a name="reference_anchor3"></a>[3] Bitcoin Dataset, Available: [https://www.investing.com/crypto/bitcoin/btc-usd-historical-data?cid=1035793](https://www.investing.com/crypto/bitcoin/btc-usd-historical-data?cid=1035793) <br/>
<a name="reference_anchor4"></a>[4] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhim, "Attention Is All You Need", Dec. 2017. [Online]. Available: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) <br/>
<a name="reference_anchor5"></a>[5] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever, “Language Models are Unsupervised Multitask Learners”, [Online]. Available: [https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
