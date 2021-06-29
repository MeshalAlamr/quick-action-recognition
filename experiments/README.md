## Experiments
| **Scenario** | **Model** | **Temporal** | **Downsample** | **Training Acc** | **Testing Acc** |
| :------ | :------: | :------: | :------: | :------: | :------: |
| 0 | Normal | 09 | 0.25 | 94.67% | 93.50% |  
| 1 | Normal | 13 | 0.25 | 94.87% | 92.79% | 
| 2 | Small | 09 | 0.25 | 94.02% | 93.27% |  
| 3 | Small | 13 | 0.25 | 95.17% | 93.55% |  
| 4 | Small | 09 | 0.33 | 93.88% | 93.50% |  
| 5 | Small | 13 | 0.33 | 94.86% | 92.54% |  
| 6 | Normal | 09 | 0.33 | 93.56% | 92.31% |  
| 7 | Normal | 13 | 0.33 | 96.18% | 94.73% |  
| 8 | Normal | 05 | 0.50 | 91.78% | 92.11% |  
| 9 | Small | 05 | 0.50 | 93.43% | 93.18% | 
| 10 | Normal | 07 | 0.50 | 95.74% | 95.24% |  
| 11 | Small | 07 | 0.50 | 93.97% | 93.12% |  
| 12 | Normal | 15 | 0.50 | 95.93% | 92.39% |  
| 13 | Small | 15 | 0.50 | 95.25% | 94.39% |  

### Notes: 
- All above experminets were done on the 10 actions dataset. The selected 10 actions of the NTU-RGB+D are as follows: _drink water, brush hair, hand waving, wipe face, rub two hands, jump up, staggering, falling down, walking towards and take off a shoe._
- "Normal" refers to **Model I**, Small refers to **Model II**.
- "Temporal" refers to the temporal kernel size.
- "Downsample" refers to the downsampling rate, for example: 0.25 means that the new number of frames is 1/4th of the original one.
