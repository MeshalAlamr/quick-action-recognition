%% Startup
clc
clear
close all

%%
normal_data_path = 'val_data.npy';
downsampled_data_path = 'val_data_downsampled.npy';

data = readNPY(normal_data_path);
data_ds = readNPY(downsampled_data_path);

n = 10; % video number
k = 1;
for i = [1:300]
    
    adj1 = []; adj2 = []; adj3= [];
    adj1_ds = []; adj2_ds = []; adj3_ds= [];
    
    j = data(n,:,i,:,1);
    j_ds = data_ds(n,:,k,:,1);
    
    figure(1)
    ax1 = subplot(1,2,1);
    ax2 = subplot(1,2,2);
    hold(ax1,'off')
    hold(ax2,'off')

    scatter3(ax1,j(1,1,1,1:25),j(1,2,1,1:25),j(1,3,1,1:25))
    hold(ax1,'on')
    scatter3(ax2,j_ds(1,1,1,1:25),j_ds(1,2,1,1:25),j_ds(1,3,1,1:25))
    hold(ax2,'on')
    pts = [];
    pts_ds = [];
    for j = [1:25]
        for x = [1:3]
            pts(j,x) = [data(n,x,i,j,1)];
            pts_ds(j,x) = [data_ds(n,x,k,j,1)];
        end
    end
    
    i = 1;
    for j = [16,15,14,13,1,2,21,5,6,7,8,23,22]
        adj1(i,1) = pts(j,1);
        adj1(i,2) = pts(j,2);
        adj1(i,3) = pts(j,3);
        
        adj1_ds(i,1) = pts_ds(j,1);
        adj1_ds(i,2) = pts_ds(j,2);
        adj1_ds(i,3) = pts_ds(j,3);
        i = i+1;
    end
    
    line(ax1,adj1(:,1), adj1(:,2),adj1(:,3))
    plot3(ax1,adj1(:,1), adj1(:,2),adj1(:,3))
    hold(ax1,'on')
    line(ax2,adj1_ds(:,1), adj1_ds(:,2),adj1_ds(:,3))
    plot3(ax2,adj1_ds(:,1), adj1_ds(:,2),adj1_ds(:,3))
    hold(ax2,'on')
    i = 1;
    for j = [20,19,18,17,1]
        adj2(i,1) = pts(j,1);
        adj2(i,2) = pts(j,2);
        adj2(i,3) = pts(j,3);
        
        adj2_ds(i,1) = pts_ds(j,1);
        adj2_ds(i,2) = pts_ds(j,2);
        adj2_ds(i,3) = pts_ds(j,3);
        i = i+1;
        
        
    end

    line(ax1,adj2(:,1), adj2(:,2),adj2(:,3))
    plot3(ax1,adj2(:,1), adj2(:,2),adj2(:,3))
    hold(ax1,'on')

    line(ax2,adj2_ds(:,1), adj2_ds(:,2),adj2_ds(:,3))
    plot3(ax2,adj2_ds(:,1), adj2_ds(:,2),adj2_ds(:,3))
    hold(ax2,'on')
    i = 1;
    for j = [4,3,21,9,10,11,12,25,24]
        adj3(i,1) = pts(j,1);
        adj3(i,2) = pts(j,2);
        adj3(i,3) = pts(j,3);
        
        adj3_ds(i,1) = pts_ds(j,1);
        adj3_ds(i,2) = pts_ds(j,2);
        adj3_ds(i,3) = pts_ds(j,3);
        i = i+1;
        
    end

    line(ax1,adj3(:,1), adj3(:,2),adj3(:,3))
    plot3(ax1,adj3(:,1), adj3(:,2),adj3(:,3))
    hold(ax1,'on')

    line(ax2,adj3_ds(:,1), adj3_ds(:,2),adj3_ds(:,3))
    plot3(ax2,adj3_ds(:,1), adj3_ds(:,2),adj3_ds(:,3))
    hold(ax2,'on')
    
    title(ax1,'Normal')
    title(ax2,'Downsampled')
    
    # 2D-View
    view(ax1,0,90)
    view(ax2,0,90)
    
    pause(0.03)
    
    k = k+1;
    if k > 150
        k = 150;
    end
    
end
