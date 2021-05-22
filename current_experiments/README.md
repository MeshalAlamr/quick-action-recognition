## Experiments

### 10 Actions Dataset

| **Scenario** | **Model** | **Temporal** | **Downsample** | **Completion** | Training Acc | Testing Acc |
| :------ | :------: | :------: | :------: | :------: | :------: | :------: |
| 0 | Normal | 09 | 0.25 | Running | - | - |  
| 1 | Normal | 13 | 0.25 | Done | 94.87% | 92.79% | 
| 2 | Small | 09 | 0.25 | Done | 94.02% | 93.27% |  
| 3 | Small | 13 | 0.25 | Done | 95.17% | 93.55% |  
| 4 | Small | 09 | 0.33 | Done | 93.88% | 93.50% |  
| 5 | Small | 13 | 0.33 | Done | 94.86% | 92.54% |  
| 6 | Normal | 09 | 0.33 | Running | - | - |  
| 7 | Normal | 13 | 0.33 | - | - | - |  

## Tasks
- [x] Create dataset with 0.25 downsampling.
- [x] Create dataset with 0.33 downsampling.
- [ ] Visualize other actions with the new downsampling rates (4 animations side by side).
- [x] Request NTU-RGB+D Dataset access.
- [ ] *Download NTU-RGB+D RGB Videos.*
- [ ] Show RGB video with visualization.
- [ ] Create a code to check minimum number of frames (non-zero) in the original dataset.
- [ ] *Train both models (10 Actions) for temporal 9 & 13 in the new downsampled data (0.25, 0.33)*.
- [ ] Train both models with 0.5 downsampling (10 Actions) for:
  - [ ] Temporal 5.
  - [ ] Temporal 7.
  - [ ] Temporal 15.
- [ ] Review & prepare a presentation for attention ST-GCN techniques.
- [ ] Retrain Model I (60 Actions) with temporal 9 for 0.5 downsample, **with batch size 4**.
