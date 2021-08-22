# Crack Captcha
The captcha we about to crack is very simple, for example: ![example](archive/example.gif)

## Model Accuracy History
|Commit Hash|Accuracy|Remark|
|---|---|---|
|`f15750`|50.5208%|Batch size 64, 20k.|
|`f15750`|55.4688%|Batch size 128, 20k.|
|`not save`|48.4375%|Disable random translate, 20k.|
|``|45.3125%|Use VGG16, 10k.|

The loss don't drop, always ~2.7. :(

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