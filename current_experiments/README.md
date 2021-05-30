## Experiments

### 10 Actions Dataset

| **Scenario** | **Model** | **Temporal** | **Downsample** | **Completion** | Training Acc | Testing Acc |
| :------ | :------: | :------: | :------: | :------: | :------: | :------: |
| 0 | Normal | 09 | 0.25 | Done | 94.67% | 93.50% |  
| 1 | Normal | 13 | 0.25 | Done | 94.87% | 92.79% | 
| 2 | Small | 09 | 0.25 | Done | 94.02% | 93.27% |  
| 3 | Small | 13 | 0.25 | Done | 95.17% | 93.55% |  
| 4 | Small | 09 | 0.33 | Done | 93.88% | 93.50% |  
| 5 | Small | 13 | 0.33 | Done | 94.86% | 92.54% |  
| 6 | Normal | 09 | 0.33 | Done | 93.56% | 92.31% |  
| 7 | Normal | 13 | 0.33 | Done | 96.18% | 94.73% |  
| **=** | **=** | **=** | **=** | **=** | **=** | **=** |  
| 8 | Normal | 05 | 0.50 | Done | 91.78% | 92.11% |  
| 9 | Small | 05 | 0.50 | Done | 93.43% | 93.18% | 
| 10 | Normal | 07 | 0.50 | Done | 95.74% | 95.24% |  
| 11 | Small | 07 | 0.50 | Done | 93.97% | 93.12% |  
| 12 | Normal | 15 | 0.50 | Done | 95.93% | 92.39% |  
| 13 | Small | 15 | 0.50 | Done | 95.25% | 94.39% |  






## Tasks
- [x] Create dataset with 0.25 downsampling.
- [x] Create dataset with 0.33 downsampling.
- [x] Visualize other actions with the new downsampling rates (4 animations side by side).
- [x] Request NTU-RGB+D Dataset access.
- [x] Download NTU-RGB+D RGB Videos.
- [x] Show RGB video with visualization.
- [ ] Create a code to check minimum number of frames (non-zero) in the original dataset.
- [x] Train both models (10 Actions) for temporal 9 & 13 in the new downsampled data (0.25, 0.33).
- [x] Train both models with 0.5 downsampling (10 Actions) for:
  - [x] Temporal 5.
  - [x] Temporal 7.
  - [x] Temporal 15.
- [ ] Review & prepare a presentation for attention ST-GCN techniques.
- [ ] Retrain Model I (60 Actions) with temporal 9 for 0.5 downsample, **with batch size 4**.
