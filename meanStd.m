%%read all images and find mean, std etc

trainFold = 'G:\athi\acads\kaggleUS\train';
nP = 47; nI = 40;
imgs = zeros(nP*nI, 420,580);
k = 1;

for p = 1:nP
    for i = 1:nI
        fname = strcat(trainFold,'\',num2str(p),'_',num2str(i),'.tif');
        img = imread(fname);
        imgs(k,:,:) = histeq(img);
        k = k+1;
    end
end
   
%%
z = zeros(420*580,100);
for k = 1:1000
temp = squeeze(imgs(k,:,:));
z(:,k) = temp(:);
end
 
