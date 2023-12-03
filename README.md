# MRI_condensation
In this repo, I perform dataset condensation via distrbution matching (https://github.com/VICO-UoE/DatasetCondensation), without differentiable siamese augmentation, on MRI classification dataset (https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)



## Conensation of Whole Unbalanced Dataset

![image](https://github.com/dianujizer/MRI_condensation/assets/47445756/53f1fbc4-64d4-430b-8190-684d8a98b3e4)

| Running Time (seconds) | Accuracy over Real Training | Accuracy over Real Testing | Condensed Images per Class |
|--------------|-----------------------------|----------------------------|-----|
| 676.7407     | 43.0662                     | 29.9492                    | 10  |



## Condensation of Balanced Dataset

![image](https://github.com/dianujizer/MRI_condensation/assets/47445756/906d6429-d2fa-4557-8448-4f6e1a3ffbbc)

| Running Time (seconds) | Accuracy over Real Training | Accuracy over Real Testing | Condensed Images per Class |
|--------------|-----------------------------|----------------------------|-----|
| 649.1408     | 25.1282                     | 20.3046                    | 10  |




# Condensation of 20 MRI images/class into 2 images/class

![image](https://github.com/dianujizer/MRI_condensation/assets/47445756/c74a426c-bedb-4748-b446-8d2150349495)

| Running Time (seconds) | Accuracy over Real Training | Accuracy over Real Testing | Condensed Images per Class |
|--------------|-----------------------------|----------------------------|-----|
| 311.9385     | 52.5                        | 27.4112                    | 2   |









