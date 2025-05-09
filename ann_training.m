I=in1';
T=out1';
net1_1 = feedforwardnet([32, 16], 'trainlm'); 
net1_1.layers{1}.transferFcn = 'tansig'; 
net1_1.layers{2}.transferFcn = 'tansig'; 
net1_1.layers{3}.transferFcn = 'purelin'; 
net1_1.trainParam.epochs = 500; 
net1_1.trainParam.lr = 0.01; 
net1_1.trainParam.goal = 1e-6; 

net1_1.divideFcn = 'dividerand'; % Chia ngẫu nhiên
net1_1.divideParam.trainRatio = 0.7; % 70% dữ liệu để huấn luyện
net1_1.divideParam.valRatio = 0.15; % 15% để kiểm tra
net1_1.divideParam.testRatio = 0.15; % 15% để đánh giá

% Huấn luyện mạng
net1_1 = train(net1_1, I, T);


