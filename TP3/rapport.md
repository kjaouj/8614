- **Name** : KJAOUJ Aymane

## **Question 1:** 
Yes, there is a clear difference. In the "Model Structure After LoRA", the standard `nn.Linear` layers in the transformer blocks have been replaced by `LinearWithLoRA` modules. These modules wrap the original frozen Linear layer and add a trainable `LoRALayer` branch.

## **Question 2:** 
- Trainable parameters: 1,327,104
- All parameters: 164,364,288
- Fraction: 0.81%

## **Question 3:** 
- Trainable parameters: 1,328,642
- All parameters: 125,768,450
- Fraction: 1.06%

=> **Comparison:** The number of trainable parameters increased slightly (by 1,538) because the new 2-class classification head is fully trainable. However, the total number of parameters decreased significantly (from ~164M to ~125M) because the original language modeling head (which projected to 50,257 tokens) was replaced by a much smaller 2-class head.

## **Question 4:** 
During training, the loss decreased from around 0.69 down to an average of 0.2055 over the first epoch. The final training accuracy of 95.1% is very good, indicating the model has effectively learned the spam classification task with only a small fraction of trainable parameters.

## **Question 5:** 
The test accuracy reached 96.32%, which is slightly higher than the training accuracy. This is a very strong result, confirming that the LoRA fine-tuned model generalizes well to unseen data and performs robustly in distinguishing spam from legitimate messages.
