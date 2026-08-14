Encoder (with GlobalAveragePooling and Linear) and Decoder (billinear with BatchNorm) after 100k steps
*transformer operates on whole frames*

![alt text](images/image-3.png)

Encoder (without GlobaAveragePooling and Linear) and Decoder (billinear and without BatchNorm) H=3, W=6 after 100k steps
*transformer operates sequentially on patches*

![alt text](images/image.png)
![alt text](images/image-1.png)
![alt text](images/image-2.png)

With GAP after 150k steps
![alt text](images/image-4.png)
![alt text](images/image-5.png)
![alt text](images/image-6.png)

Encoder with linear but without GAP, Decoder with PixelShuffle, without BN after 150k steps
![alt text](images/image-7.png)
![alt text](images/image-8.png)
![alt text](images/image-9.png)

No shape_loss, bigger encoder (128 channels), dropout added, bigger emb, ater 2k steps
![alt text](images/image-10.png)
![alt text](images/image-11.png)


Encoder with FC layers, dropout in enc, entropy penalty
![alt text](images/image-12.png)
![alt text](images/image-13.png)
*generates only these two two types of images*
*gradient noise made the results worse*