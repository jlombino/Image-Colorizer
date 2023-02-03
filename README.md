<h1>Image & Video Colorizer</h1>

<p>
For more information on this project, please see this <a href=./Colorizer_Analysis.ipynb>jupyter notebook</a> or the <a href=./Slides.pdf>presentation</a>.
</p>

<h3>Disclaimer</h3>

<p>
This project features a hypothetical business case and is for educational purposes only.
</p>

<h3>Repository Structure</h3>

```
├── resources                        <- Images and videos used in notebooks
├── Colorizer_Analysis.ipynb         <- Final notebook summarizing all work
├── Video_Colorizer.ipynb            <- Notebook for model trained on video data
├── GAN_Colorizer.ipynb              <- Notebook for model trained using adversarial loss
├── MSE_Colorizer.ipynb              <- Notebook for model trained using mean squared error loss
├── colorizerutils.py                <- Utility functions used in multiple notebooks
└── README.md                        <- Top-level README file for the repository
```
<hr>

<h1>Introduction</h1>

<img src=resources/bentocolor_logo.png> 

<p>
BentoColor is widely known for providing colorization services for both personal and industrial scale customers.
</p>

<p>
While BentoColor employs many talented technicians to provide colorization services, the recent advances in machine learning has led them to question whether it can be an effective tool for colorizing images and videos. A successful implementation of machine learning based colorization tools can assist technicians greatly with their colorization work. BentoColor aims to bolster their efforts in the field of colorization by leveraging their technicians' increased productivity. This will help attract large customers such as broadcasting companies and museums as well as individuals with small-scale colorization jobs.
</p>

<h3>Current Technology</h3>

<p>
Colorization is the process of converting a black and white image or video into one with color. There are many techniques available for use in colorization. For example, manual colorization is a laborious process where technicians color each image or frame individually using image editing software. While it is somewhat common for still images, advancements in image processing algorithms have resulted in manual colorization becoming mostly obsolete for videos. 
</p>

<p>
These algorithms allow for objects to be tracked between video frames. If one frame is colorized manually, it is often possible for many similar frames to be colorized automatically. It is even feasible to automatically account for changes in brightness and perspective. This process, while substantially easier than manually colorizing a video, is still very time consuming for technicians. They need to verify that the algorithms are tracking objects and applying colors correctly for each frame individually. In addition, technicians need to manually add colors whenever a new object enters the frame or the scene changes substantially.
</p>

<h3>BentoColor Technology</h3>

<p>
Despite the many challenges, BentoColor plans to utilize neural networks to automatically colorize images and video frames. Specifically, BentoColor plans to utilize a U-Net similar to the ones described in [<a href=https://arxiv.org/pdf/1505.04597.pdf>U-Net: Convolutional Networks for Biomedical Image Segmentation</a>] and [<a href=https://www.tensorflow.org/tutorials/generative/pix2pix>pix2pix: Image-to-image translation with a conditional GAN</a>] to accomplish this goal.
</p>

<h1>Recommendations</h1>

<h3>Partnerships for Data Acquisition</h3>

<p>
First, BentoColor should make partnerships with image and movie rightshholders to acquire more video data. BentoColor should subsidize colorization services in exchange for the right to train future models on shows and movies owned by these partners. This will help BentoColor acquire new customers and form relationships with them. In addition, it will provide the necessary data to be used when training the next generation of models.
</p>

<h3>Model Upgrades</h3>

<p>
When creating the next generation of models, BentoColor should consider incorporating a classifier into the generator. An example of this architecture is shown in [<a href=https://arxiv.org/pdf/1712.03400.pdf>Deep Koalarization: Image Colorization using CNNs and Inception-ResNet-v2</a>]. Currently, the model can not determine whether it is coloring a shirt or a hat, and is only selecting colors for the image based on patterns of pixels. Adding a classifier to the model would provide it with context and likely improve colorization. This could not be incorporated into current models because pre-trained classifiers require a larger image resolution than BentoColors' current hardware can process in a reasonable amount of time.
</p>

<h3>Current Benefits</h3>

<p>
Finally, while the colorization model needs more work before it can automatically colorize videos, it can be deployed alongside current technology. It can colorize individual frames reasonably well which can provide a useful starting point for technicians. It is likely that fully automatic video colorization will be possible in the near future, but even the limited version available today can be a useful tool in the hands of talented technicians.
</p>

<h1>Data</h1>

<h3>Image Dataset - Landscapes</h3>

<p>
The first dataset used to train a model is [<a href=https://www.kaggle.com/datasets/utkarshsaxenadn/landscape-recognition-image-dataset-12k-images>Landscape Recognition | Image Dataset | 12k Images</a>] found on Kaggle. The dataset contains 10,000 training images, 1,500 validation images, and 500 test images of landsacpes. These landscapes fall into five categories: Coast, Desert, Forest, Glacier, and Mountain. All five categories were combined to form one large cohesive dataset while training these models. While the original images were of varying resolutions, every image was resized to 64x64 pixels and randomly flipped and rotated.
</p>

<img src=resources/train_images.png width=500>

<h3>Video Dataset - Nature</h3>

<p>
The second dataset used to train a model is a collection of nature videos sourced from the <a href=https://dareful.com/videos/nature/>Dareful Nature Collection</a>. The collection contains twenty seven videos of which twenty five were used for training data. The remaining two were used as the test set. The videos were split into frames at a rate of two frames per second using FFmpeg. There were a total of nearly 1,200 frames in the training set which is significantly fewer than for the image colorization model.

<img src=resources/train_frames.png width=500></img>

<h3>Color Spaces - RGB vs. LAB</h3>

<p>
The LAB color space is much more convenient than the typical RGB representation for the task of colorization. This is because in the LAB color space, the entire grayscale part of the image is contained in just the L channel. Making gray using RGB would require mixing all three colors in an equal proportion. LAB is convenient because the model will only need to predict new values for the A and B color channels. The original L channel can just be combined with the predicted A and B to make a fully colored LAB image. In contrast, if RGB were used, the model would need to predict new values for all three color channels. This reduces the complexity of the model substantially.
</p>

<h1>Image Colorization</h1>

<h3>Generator U-Net Architecure</h3>

<p>
The (not to scale) diagram below shows the literature based U-Net architecture for the generator that is used to colorize the images. It accepts one grayscale (L channel from LAB) image as an input and predicts two color channels (A and B channels from LAB) for that grayscale image. The predicted channels can be recombined with the input grayscale image to form a fully colored image. The input and output image sizes are both 64x64 pixels. The generator is built using a series of downsampling blocks followed by a series of upsampling blocks with skip connections. The code used to make the generator is found in this <a href=./MSE_Colorizer.ipynb>jupyter notebook</a>.
</p>

<img src=resources/generator_arch.png>

<h3>Training with Mean Squared Error Loss</h3>

<p>
The first attempt at training the generator to colorize images used mean squared error loss. To calculate this loss, subtract the color values for a pixel in the generated image from the color values for the same pixel in the original color image and square the result. Repeat this for each pixel in the image and add all the squared values. Then divide that by the number of pixels in the image. This method results in loss getting smaller as the generated image's colors get closer to the original image's colors. Because the error is squared, pixels that are very far from their intended values are penalized more heavliy than pixels that are close to their intended values. More information on the model trained using mean squared error as well as the code used to train this model can be found in this <a href=./MSE_Colorizer.ipynb>jupyter notebook.</a>
</p>

<img src=resources/mse_graphic.png width=300></img>

<p>

</p>

<h3>Training Results</h3>

<p>
The chart below shows the model's mean squared error loss plotted against the number of steps the model was trained for. The clear downward trend signals that the model's output images are becoming closer to the original color images over time. The training was ended because the model's improvements in loss were slowing down and training the model more in this situation could easily lead to overfitting. Results for images colorized using this model can be found in the Image Colorization Results section below.
</p>

<img src=resources/MSE_Loss_TBoard.jpg width=600></img>

<h1>Generative Adversarial Networks</h1>

<p>
Following training using mean squared error loss, a separate model was trained using a combination of mean squared error and adversarial loss. A generative adversarial network architecutre based on [<a href=https://arxiv.org/pdf/1611.07004.pdf>Image-to-Image Translation with Conditional Adversarial Networks</a>] was used with a modified and simplified loss function for this purpose. 
</p>

<p>
Generative adversarial networks (GANs) actually consist of two separate neural networks: the generator and the discriminator. The generator is primarily responsible for colorizing the images while the discriminator is responsible for finding differences between the generated colors and the original image's colors. If the discriminator is able to find differences, the generator is trained to produce images without those differences. The discriminator is then tasked with finding even more differences for the generator to fix. This is the general training loop for a GAN and can be repeated until the discriminator can not find differences between the generated images and the original images. This training loop is pictured in the diagram below. More information on the model trained using adversarial loss as well as the code used to train this model can be found in this <a href=./GAN_Colorizer.ipynb>jupyter notebook.</a>
</p>

<img src=resources/Gan_Arch.png width=300></img>

<h3>Adding the Discriminator</h3>

<p>
The (not to scale) diagram below shows the discriminator which is used to train the generator adversarially. It accepts three image channels: one grayscale and two color. The output is a matrix that shows whether the discriminator thinks each section of the image was created by the generator (represented as 0) or an original image (represented as 1). The discriminator uses a standard convolutional neural network architecture. It is built using repeated downsampling blocks with an increasing number of filters designed to capture the increasing number of conbinations of patterns in the images.
</p>

<p>
The code used to make the discriminator is found in this <a href=./GAN_Colorizer.ipynb>jupyter notebook</a>.
</p>

<img src=resources/discriminator_arch.png></img>

<h1>Training with Adversarial Loss</h1>

<h3>Generator Training</h3>

<p>
The generator recieves a grayscale image as an input and predicts color channels based on that image. A partial loss is then calculated based on mean squared error as described above. In addition to this, the generated images are also sent to the discriminator. The generator wants to fool the discriminator, so the loss is calculated based on how many sections of the image the discriminator thinks are generated images. The loss is high if the discriminator thinks there are many generated sections, and low if the discriminator thinks there are many original colored sections. These discriminator predictions as well as the scaled mean squared error loss are then used to update the weights of the generator. The diagram below shows the generator training loop in detail.
</p>

<img src=resources/Gen_Training_Loop.png width=300></img>

<h3>Discriminator Training</h3>

<p>
The discriminator recieves either original color images or images colored by the generator as an input. The goal of the discriminator is to classify sections of images colored by the generator as fake (0) and classify sections of original colored images as real (1). The discriminator looks at sections of images rather than images as a whole to provide the generator with more granular feedback. Discriminator loss is low if it correctly identifies the original and generated colors, and high if it incorrectly identifies them. These predictions are then used to update the weights of the discriminator. The training loop is pictured in the diagram below.
</p>

<img src=resources/Disc_Training_Loop.png width=300></img>

<h3>Training Results</h3>

<p>
The four graphs below show the discriminator loss, generator loss (adversarial), generator loss (mean squared error), and total generator loss.
</p>

<ul>
<li>The discriminator loss starts high and decreases over time as the discriminator learns to distinguish between the generator's predicted colors and the original colored images.</li>
<li>The generator adversarial loss is the inverse of the discriminator loss. It increases over time because the discriminator improves more quickly than the generator.</li>
<li>The generator MSE loss decreases over time as the generator predicted images more closely resemble the original color images. This improvement slows over time and eventually reverses as the adversarial loss begins to dominate.</li>
<li>The generator total loss is the weighted sum of the adversarial and MSE loss components. The MSE component is larger initially, but the adversarial component eventually begins to dominate.</li>
</ul>

<img src=resources/GAN_Loss_TBoard.jpg width=600> </img>

<h1>Image Colorization Results</h1>

<p>
Here, results are shown for both models. The leftmost image is the grayscale image. The second image is the output from the generator trained using only MSE loss. The third image is the output from the generator trained using adversarial and MSE loss combined. The rightmost image is the original color image.
</p>

<h3>Colorized Images</h3>
<img src=resources/Colorization_Results.png width=600></img>

<h3>Interpretation</h3>

<p>
<b>The interpretation of these results and the relative success of each model is highly subjective. Each person viewing the results may prefer a different colorization.</b>
</p>

<p>
There are several important questions to ask when determining how well a given model colorizes images. These include:

<ul>
<li>How well do the generator colorized images match the original training and test images?</li>
<li>Are the differences between the original and generator colorized images expected or reasonable?</li>
<li>Are there any major defects or inconsistencies in the generator colorized images?</li>
<li>Are the generator colorized images vibrant and colorful when appropriate?</li>
<li>Do the predicted colors accentuate details in the images?</li>
</ul>

<p>
In this case, there are not many defects in the generator colorized images. The GAN produces more vibrant and detailed images while the generator trained using only MSE tends to use fewer colors overall. The GAN is also noticably better at capturing and accentuating the details in the images while the generator trained using MSE tends to apply colors across object boundaries more often. The GAN deviates more from the colors in the original image, but it often uses colors that could also be appropriate for the objects in the scene.
</p>

<h1>Video Colorization</h1>

<h3>Video Data</h3>

<p>
As mentioned previously, the video colorization model was trained using the <a href=https://dareful.com/videos/nature/>Dareful Nature Collection</a>
</p>

<img src=resources/train_frames.png width=500>

<h3>Model Selection</h3>

<p>
The GAN was selected as the model to colorize videos over the model trained purely using MSE. It was selected because it performed better than the MSE model in colorizing still images, producing more vibrant images with sharper details. The loss function for the GAN was also adjusted to have a lower weighting for the mean squared error component. This was done to reduce overfitting caused by the MSE component as there are many near duplicate frames in the dataset. More information on the GAN used to colorize videos as well as the code used to train this model can be found in this <a href=./Video_Colorizer.ipynb>jupyter notebook</a>.
</p>

<img src=resources/Gan_Arch.png width=300></img>

<h3>Training Results</h3>

<p>
The four charts below show the discriminator loss, generator loss(adversarial), generator loss (mean squared error), and total generator loss.
</p>

<ul>
<li>The discriminator loss starts high and decreases over time as the discriminator learns to distinguish between the generator's predicted colors and the original color images.</li>
<li>The generator adversarial loss is the inverse of the discriminator loss. It increases over time because the discriminator improves more quickly than the generator.</li>
<li>The generator MSE loss decreases over time as the generator predicted images more closely resemble the original color images. The overall MSE is lower than the image colorization model, likely because it is easier to fit to the many near duplicate frames.</li>
<li>The generator total loss is the weighted sum of the adversarial and MSE loss components. The MSE component is weighted lower than in the image colorization model. The MSE component is larger initially, but the adversarial component eventually begins to dominate.</li>
</ul>

<img src=resources/Video_Loss_TBoard.jpg></img>

<h1>Video Colorization Results</h1>

<p>
Here, results are shown for the video colorizing model. The leftmost video is the grayscale video. The center video is the one colorized by the model. The rightmost video is the original video.
</p>

<h3>Colorized Videos</h3>

<table>
    <tr>
        <th>Grayscale Video</th>
        <th>Colorized Video</th>
        <th>Original Video</th>
    <tr>
        <td><img src=resources/video/test_5_grayscale.gif width=256></img></td>
        <td><img src=resources/video/test_5_colorized.gif width=256></img></td>
        <td><img src=resources/video/test_5_original.gif width=256></img></td>
    </tr>
    <tr>
        <th>Grayscale Video</th>
        <th>Colorized Video</th>
        <th>Original Video</th>
    <tr>
    <tr>
        <td><img src=resources/video/test_10_grayscale.gif width=256></img></td>
        <td><img src=resources/video/test_10_colorized.gif width=256></img></td>
        <td><img src=resources/video/test_10_original.gif width=256></img></td>
    </tr>
    <tr>
        <th>Grayscale Video</th>
        <th>Colorized Video</th>
        <th>Original Video</th>
    <tr>
    <tr>
        <td><img src=resources/video/test_20_grayscale.gif width=256></img></td>
        <td><img src=resources/video/test_20_colorized.gif width=256></img></td>
        <td><img src=resources/video/test_20_original.gif width=256></img></td>
    </tr>
    <tr>
        <th>Grayscale Video</th>
        <th>Colorized Video</th>
        <th>Original Video</th>
    <tr>
    <tr>
        <td><img src=resources/video/test_25_grayscale.gif width=256></img></td>
        <td><img src=resources/video/test_25_colorized.gif width=256></img></td>
        <td><img src=resources/video/test_25_original.gif width=256></img></td>
    </tr>
</table>

<h3>Interpretation</h3>

<p>
<b>The interpretation of these results and the relative success of each model is highly subjective. Each person viewing the results may prefer a different colorization.</b>
</p>

<p >
The video colorizing model can be assessed by asking many of the same questions:
</p>

<ul>
<li>How well do the generator colorized videos match the original training and test videos?</li>
<li>Are the differences between the original and generator colorized videos expected or reasonable?</li>
<li>Are there any major defects or inconsistencies in the generator colorized videos?</li>
<li>Are the generator colorized videos vibrant and colorful when appropriate?</li>
<li>Do the predicted colors accentuate details in the videos?</li>
</ul>

<p>
As well as some new questions:
</p>

<ul>
<li>Are the colors consistent frame-to-frame in the videos?</li>
<li>Do the colors track the boundaries of objects well as they move around in frame?</li>
</ul>

<p>
While the colors in the videos are pretty consistent, they are definitely not as vibrant as the original video's colors. The model appears to be tracking large objects such as the sky and trees reasonably well, but has problems finding the smaller objects in each frame. There are no major defects such as random lines or patches, but the colorized videos still do not match the original videos well. This is likely the result of the lower quality of the training dataset with its nearly duplicate frames and smaller size. Using more short videos to allow the model to experience a wider variety of scenes would likely have produced a substantially better model.
</p>

<h1>Conclusion</h1>

<p>
Automatic image and video colorization should be possible with current machine learning technology. The developed model was capable of colorizing small images reasonably well. Colorizing videos using a similar model posed some unique challenges, but should be feasible with more data and development time. While automatic colorization is currently not ready for commercial deployment, any of these models can be used to assist technicians in their manual or assisted colorization efforts.
</p>