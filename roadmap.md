## Encoder (with GlobalAveragePooling and Linear) and Decoder (billinear with BatchNorm) after 100k steps

\_\_transformer operates on whole frames\_\_

<img src="images/image-3.png" alt="alt text" height="300">

## Encoder (without GlobaAveragePooling and Linear) and Decoder (billinear and without BatchNorm) H=3, W=6 after 100k steps

\_\_transformer operates sequentially on patches\_\_

<img src="images/image.png" alt="alt text" height="300">
<img src="images/image-1.png" alt="alt text" height="300">
<img src="images/image-2.png" alt="alt text" height="300">

## With GAP after 150k steps

<img src="images/image-4.png" alt="alt text" height="300">
<img src="images/image-5.png" alt="alt text" height="300">
<img src="images/image-6.png" alt="alt text" height="300">

## Encoder with linear but without GAP, Decoder with PixelShuffle, without BN after 150k steps

<img src="images/image-7.png" alt="alt text" height="300">
<img src="images/image-8.png" alt="alt text" height="300">
<img src="images/image-9.png" alt="alt text" height="300">

## No shape_loss, bigger encoder (128 channels), dropout added, bigger emb, ater 2k steps

<img src="images/image-10.png" alt="alt text" height="300">
<img src="images/image-11.png" alt="alt text" height="300">

## Encoder with FC layers, dropout in enc, entropy penalty

\_\_generates only these two two types of images\_\_

\_\_gradient noise made the results worse\_\_

<img src="images/image-12.png" alt="alt text" height="300">
<img src="images/image-13.png" alt="alt text" height="300">

## `loss = main_loss + 0.3 * shape_loss + 0.1 * progress_loss + 0.2 * entropy_penalty + 5.0 * isolated_loss + 0.1 * large_wood_loss`

<img src="images/image-14.png" alt="alt text" height="300">
<img src="images/image-15.png" alt="alt text" height="300">
<img src="images/image-16.png" alt="alt text" height="300">

## `loss = main_loss + 0.0 * shape_loss + 0.1 * progress_loss + entropy_penalty + 0.1 * large_wood_loss`

large_wood_loss with 6x6 kernel

Generations are diverse with little noice and look like trees

<img src="images/image-17.png" alt="alt text" height="300">
<img src="images/image-18.png" alt="alt text" height="300">
<img src="images/image-19.png" alt="alt text" height="300">
<img src="images/image-20.png" alt="alt text" height="300">
<img src="images/image-21.png" alt="alt text" height="300">
<img src="images/image-22.png" alt="alt text" height="300">
