## Experiments

### 10 Actions Dataset

| **Scenario** | **Model** | **Temporal** | **Downsample** | **Completion** |
| :------ | :------: | :------: | :------: | :------: |
| 0 | Normal | 09 | 0.25 | - |  
| 1 | Normal | 13 | 0.25 | - |  
| 2 | Small | 09 | 0.25 | Done |  
| 3 | Small | 13 | 0.25 | Running |  
| 4 | Small | 09 | 0.33 | Running |  
| 5 | Small | 13 | 0.33 | - |  
| 6 | Small | 09 | 0.33 | - |  
| 7 | Small | 13 | 0.33 | - |  

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