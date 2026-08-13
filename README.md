Encoder (with GlobalAveragePooling and Linear) and Decoder (billinear with BatchNorm) after 100k steps
*transformer operates on whole frames*

![alt text](image-3.png)

Encoder (without GlobaAveragePooling and Linear) and Decoder (billinear and without BatchNorm) H=3, W=6 after 100k steps
*transformer operates sequentially on patches*

![alt text](image.png)
![alt text](image-1.png)
![alt text](image-2.png)

With GAP after 150k steps
![alt text](image-4.png)
![alt text](image-5.png)
![alt text](image-6.png)

Encoder with linear but without GAP, Decoder with PixelShuffle, without BN
![alt text](image-7.png)
![alt text](image-8.png)
![alt text](image-9.png)