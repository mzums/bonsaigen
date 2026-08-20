Encoder (with GlobalAveragePooling and Linear) and Decoder (billinear with BatchNorm) after 100k steps
_transformer operates on whole frames_

![alt text](images/image-3.png)

Encoder (without GlobaAveragePooling and Linear) and Decoder (billinear and without BatchNorm) H=3, W=6 after 100k steps
_transformer operates sequentially on patches_

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
_generates only these two two types of images_  
_gradient noise made the results worse_

`loss = main_loss + 0.3 * shape_loss + 0.1 * progress_loss + 0.2 * entropy_penalty + 5.0 * isolated_loss + 0.1 * large_wood_loss`  
![alt text](images/image-14.png)
![alt text](images/image-15.png)
![alt text](images/image-16.png)

`loss = main_loss + 0.0 * shape_loss + 0.1 * progress_loss + entropy_penalty + 0.1 * large_wood_loss`
large_wood_loss with 6x6 kernel
![alt text](images/image-17.png)
![alt text](images/image-18.png)
![alt text](images/image-19.png)
![alt text](images/image-20.png)
![alt text](images/image-21.png)
![alt text](images/image-22.png)
