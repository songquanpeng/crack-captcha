# Crack Captcha
The captcha we about to crack is very simple, for example: ![example](archive/example.gif)

## Model Accuracy History
|Hash|Accuracy|Remark|
|---|---|---|
|`f15750`|45.3125%|Initial version, 10k.|


## TODOs
+ [x] Construct the model.
+ [x] Crawl the captcha dataset.
+ [x] Label the dataset.
+ [x] Make it runnable.
+ [x] Finish the model training.
+ [ ] Construct API for the trained model.
+ [ ] Data augmentation: add salt & pepper noise.

## Others
I referred [xmcp/elective-dataset-2021spring](https://github.com/xmcp/elective-dataset-2021spring) when crafting this project.