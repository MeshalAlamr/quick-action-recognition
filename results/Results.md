# Parameters
- test_bach_size 8

# Pre-downsampling
## NTU RGB+D
### xview
[03.31.21|11:37:04] Model:   net.st_gcn.Model.  
[03.31.21|11:37:04] Weights: ./models/st_gcn.ntu-xview.pt.  
[03.31.21|11:37:04] Evaluation Start:  
[03.31.21|12:00:38]     mean_loss: 0.3523266639464899  
[03.31.21|12:00:38]     Top1: 88.76%  
[03.31.21|12:00:38]     Top5: 98.83%  
[03.31.21|12:00:38] Done.  

### xsub 
[03.31.21|12:52:41] Model:   net.st_gcn.Model.  
[03.31.21|12:52:41] Weights: ./models/st_gcn.ntu-xsub.pt.  
[03.31.21|12:52:41] Evaluation Start:  
[03.31.21|13:13:11]     mean_loss: 0.6566342396967307  
[03.31.21|13:13:11]     Top1: 81.57%  
[03.31.21|13:13:11]     Top5: 96.85%  
[03.31.21|13:13:11] Done.  

## Kinetics 
[03.31.21|13:19:12] Model:   net.st_gcn.Model.  
[03.31.21|13:19:12] Weights: ./models/st_gcn.kinetics.pt.  
[03.31.21|13:19:12] Evaluation Start:  
[03.31.21|13:37:17]     mean_loss: 3.233071251686173  
[03.31.21|13:37:17]     Top1: 31.60%  
[03.31.21|13:37:18]     Top5: 53.68%  
[03.31.21|13:37:18] Done.  


# Post-downsampling
## NTU RGB+D
### xview
[04.02.21|01:18:54] Model:   net.st_gcn.Model.  
[04.02.21|01:18:54] Weights: ./models/st_gcn.ntu-xview.pt.  
[04.02.21|01:18:54] Evaluation Start:  
[04.02.21|01:32:40]     mean_loss: 3.7022051528240993  
[04.02.21|01:34:30]     Top1: 47.13%  
[04.02.21|01:34:30]     Top5: 81.33%  
[04.02.21|01:34:30] Done.  

### xsub 
[04.02.21|01:39:47] Model:   net.st_gcn.Model.  
[04.02.21|01:39:47] Weights: ./models/st_gcn.ntu-xsub.pt.  
[04.02.21|01:39:47] Evaluation Start:  
[04.02.21|01:50:23]     mean_loss: 4.730176019350818  
[04.02.21|01:50:23]     Top1: 44.51%  
[04.02.21|01:50:23]     Top5: 77.33%  
[04.02.21|01:50:23] Done.  

## Kinetics 
