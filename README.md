# Tools for generating and detecting synthetic content

## Table of Contents

* [Generation of synthetic content](#generation-of-synthetic-content)
  * [Text](#generation-text)
  * [Audio](#generation-audio)
  * [Images](#generation-images)
  * [Video](#generation-videos)
* [Detection of synthetic content](#detection-of-synthetic-content)
  * [Text](#generation-text)
  * [Audio](#generation-audio)
  * [Images](#generation-images)
  * [Video](#generation-videos)

## Generation of synthetic content

### Generation Text

#### ⚒️ Tools

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
| [Grover](https://github.com/rowanz/grover) | Grover is a model for Neural Fake News -- both generation and detection. However, it probably can also be used for other generation tasks. | [https://grover.allenai.org/](https://grover.allenai.org/) | [![stars](https://badgen.net/github/stars/rowanz/grover)](https://github.com/rowanz/grover)|
[gpt-2xy](https://github.com/NaxAlpha/gpt-2xy) | GPT-2 User Interface based on HuggingFace's Pytorch Implementation | https://gpt2.ai-demo.xyz/ | [![stars](https://badgen.net/github/stars/NaxAlpha/gpt-2xy)](https://github.com/NaxAlpha/gpt-2xy) |
[CTRL](https://github.com/salesforce/ctrl) | Conditional Transformer Language Model for Controllable Generation | N/A | [![stars](https://badgen.net/github/stars/salesforce/ctrl)](https://github.com/salesforce/ctrl) |
[Talk to Transformer](https://talktotransformer.com/) | See how a modern neural network completes your text. Type a custom snippet or try one of the examples | https://talktotransformer.com | N/A |
[LEO](http://leoia.es) | First intelligent system for creating news in Spanish | N/A | N/A

#### 📃 Papers

* [Saliency Maps Generation for Automatic Text Summarization](https://arxiv.org/pdf/1907.05664.pdf)
* [Automatic Conditional Generation of Personalized Social Media Short Texts ](https://arxiv.org/pdf/1906.09324.pdf)
* [Neural Text Generation in Stories Using Entity Representations as Context](https://homes.cs.washington.edu/~eaclark7/www/naacl2018.pdf)
* [DeepTingle](https://arxiv.org/pdf/1705.03557.pdf)

#### 😎 Awesome

* [awesome-text-generation](https://github.com/ChenChengKuan/awesome-text-generation)

### Generation Audio

#### ⚒️ Tools

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
[Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) | Clone a voice in 5 seconds to generate arbitrary speech in real-time | https://www.youtube.com/watch?v=-O_hYhToKoA | [![stars](https://badgen.net/github/stars/CorentinJ/Real-Time-Voice-Cloning)](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
[Lyrebird](https://beta.myvoice.lyrebird.ai/) | Create your own vocal avatar! | N/A | N/A |
[Descrypt](https://www.descript.com/) | Record. Transcribe. Edit. Mix. As easy as typing. | N/A | N/A
[Common Voice](https://voice.mozilla.org/en) | Common Voice is Mozilla's initiative to help teach machines how real people speak. | N/A | N/A
[Resemble.ai](https://www.resemble.ai/) | Resemble can clone any voice so it sounds like a real human. | N/A | N/A


#### 📃 Papers
* [Neural Voice Cloning with a Few Samples](http://research.baidu.com/Blog/index-view?id=81)
* [Data Efficient Voice Cloning for Neural Singing Synthesis](https://mtg.github.io/singing-synthesis-demos/voice-cloning/)
* [Efficient Neural Audio Synthesis](https://arxiv.org/pdf/1802.08435v1.pdf)
* [Score and Lyrics-free Singing Voice Generation](https://arxiv.org/pdf/1912.11747.pdf)

### Generation Images

#### ⚒️ Tools

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
[StyleGAN](https://github.com/NVlabs/stylegan) | An alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. The new architecture leads to an automatically learned, unsupervised separation of high-level attributes (e.g., pose and identity when trained on human faces) and stochastic variation in the generated images (e.g., freckles, hair), and it enables intuitive, scale-specific control of the synthesis. The new generator improves the state-of-the-art in terms of traditional distribution quality metrics, leads to demonstrably better interpolation properties, and also better disentangles the latent factors of variation. | https://www.youtube.com/watch?v=kSLJriaOumA | [![stars](https://badgen.net/github/stars/NVlabs/stylegan)](https://github.com/NVlabs/stylegan)
[StyleGAN2](https://github.com/NVlabs/stylegan2) | Improved version for StyleGAN. | https://www.youtube.com/watch?v=c-NJtV9Jvp0 | [![stars](https://badgen.net/github/stars/NVlabs/stylegan2)](https://github.com/NVlabs/stylegan2)

#### 📃 Papers

* [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948.pdf)
* [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/pdf/1912.04958.pdf)
* [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/pdf/1711.11585.pdf)
* [Complement Face Forensic Detection and Localization with Facial Landmarks](https://arxiv.org/pdf/1910.05455.pdf)

### Generation Videos

#### ⚒️ Tools

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
| [FaceSwap](https://github.com/deepfakes/faceswap) | Grover is a model for Neural Fake News -- both generation and detection. However, it probably can also be used for other generation tasks. | https://www.youtube.com/watch?v=r1jng79a5xc | [![stars](https://badgen.net/github/stars/deepfakes/faceswap)](https://github.com/deepfakes/faceswap)|
[Face2Face](https://github.com/datitran/face2face-demo) | FaceSwap is a tool that utilizes deep learning to recognize and swap faces in pictures and videos. | N/A | [![stars](https://badgen.net/github/stars/datitran/face2face-demo)](https://github.com/datitran/face2face-demo) |
[Faceswap](https://github.com/MarekKowalski/FaceSwap) | FaceSwap is an app that I have originally created as an exercise for my students in "Mathematics in Multimedia" on the Warsaw University of Technology. | N/A | [![stars](https://badgen.net/github/stars/MarekKowalski/FaceSwap)](https://github.com/MarekKowalski/FaceSwap) |
[Faceswap-GAN](https://github.com/shaoanlu/faceswap-GAN) | Adding Adversarial loss and perceptual loss (VGGface) to deepfakes'(reddit user) auto-encoder architecture. | https://github.com/shaoanlu/faceswap-GAN/blob/master/colab_demo/faceswap-GAN_colab_demo.ipynb | [![stars](https://badgen.net/github/stars/shaoanlu/faceswap-GAN)](https://github.com/shaoanlu/faceswap-GAN) |


#### 📃 Papers

* [HeadOn: Real-time Reenactment of Human Portrait Videos](https://arxiv.org/pdf/1805.11729.pdf)
* [Face2Face: Real-time Face Capture and Reenactment of RGB Videos](http://gvv.mpi-inf.mpg.de/projects/MZ/Papers/CVPR2016_FF/page.html)
* [Synthesizing Obama: Learning Lip Sync from Audio](https://grail.cs.washington.edu/projects/AudioToObama/siggraph17_obama.pdf)

## Detection of synthetic content

### Detection Text

#### ⚒️ Tools
| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
| [Grover](https://github.com/rowanz/grover) | Grover is a model for Neural Fake News -- both generation and detection. However, it probably can also be used for other generation tasks. | [https://grover.allenai.org/](https://grover.allenai.org/) | [![stars](https://badgen.net/github/stars/rowanz/grover)](https://github.com/rowanz/grover)|
[GLTR](https://github.com/HendrikStrobelt/detecting-fake-text) | Detecting text that was generated from large language models (e.g. GPT-2). | http://gltr.io/dist/index.html | [![stars](https://badgen.net/github/stars/HendrikStrobelt/detecting-fake-text)](https://github.com/HendrikStrobelt/detecting-fake-text) |

#### 📃 Papers

* [GLTR: Statistical Detection and Visualization of Generated Text](https://arxiv.org/pdf/1906.04043.pdf)
* [Human and Automatic Detection of Generated Text](https://arxiv.org/pdf/1911.00650.pdf)
* [CTRL: A CONDITIONAL TRANSFORMER LANGUAGE MODEL FOR CONTROLLABLE GENERATION](https://arxiv.org/pdf/1909.05858.pdf)

### Detection Audio

#### ⚒️ Tools

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
| [Spooded speech detection](https://github.com/elleros/spoofed-speech-detection) | This work is part of the "DDoS Resilient Emergency Dispatch Center" project at the University of Houston, funded by the Department of Homeland Security (DHS). | N/A | [![stars](https://badgen.net/github/stars/elleros/spoofed-speech-detection)](https://github.com/elleros/spoofed-speech-detection)|
[Fake voice detection](https://github.com/dessa-public/fake-voice-detection) | This repository provides the code for a fake audio detection model built using Foundations Atlas. It also includes a pre-trained model and inference code, which you can test on any of your own audio files. | N/A | [![stars](https://badgen.net/github/stars/dessa-public/fake-voice-detection)](https://github.com/dessa-public/fake-voice-detection)
[Fake Voice Detector](https://github.com/kstoneriv3/Fake-Voice-Detection) | For "Deep Learning class" at ETHZ. Evaluate how well the fake voice of Barack Obama 1. confuses the voice verification system, 2. can be detected. | N/A | [![stars](https://badgen.net/github/stars/kstoneriv3/Fake-Voice-Detection)](https://github.com/kstoneriv3/Fake-Voice-Detection)
[CycleGAN Voice Converter](https://leimao.github.io/project/Voice-Converter-CycleGAN/) | An implementation of CycleGAN on human speech conversions | https://leimao.github.io/project/Voice-Converter-CycleGAN/ | [![stars](https://badgen.net/github/stars/leimao/Voice_Converter_CycleGAN)](https://github.com/leimao/Voice_Converter_CycleGAN)


#### 📃 Papers

* [Can We Detect Fake Voice Generated by GANs?](https://github.com/kstoneriv3/Fake-Voice-Detection/blob/master/DLproject_fake_voice_detection.pdf)
* [CycleGAN Voice Converter](https://leimao.github.io/project/Voice-Converter-CycleGAN/)

### Detection Images

#### ⚒️ Tools
| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
| [FALdetector](https://github.com/peterwang512/FALdetector) | Detecting Photoshopped Faces by Scripting Photoshop. | N/A| [![stars](https://badgen.net/github/stars/peterwang512/FALdetector)](https://github.com/peterwang512/FALdetector)|

#### 📃 Papers

* [Detecting Photoshopped Faces by Scripting Photoshop](https://arxiv.org/pdf/1906.05856.pdf)

### Detection Videos

#### ⚒️ Tools

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
| [FaceForensics++](https://github.com/ondyari/FaceForensics) | FaceForensics++ is a forensics dataset consisting of 1000 original video sequences that have been manipulated with four automated face manipulation methods: Deepfakes, Face2Face, FaceSwap and NeuralTextures. | https://www.youtube.com/watch?v=x2g48Q2I2ZQ | [![stars](https://badgen.net/github/stars/ondyari/FaceForensics)](https://github.com/ondyari/FaceForensics)|
| [Face Artifacts](https://github.com/danmohaha/CVPRW2019_Face_Artifacts) | Our method is based on the observations that current DeepFake algorithm can only generate images of limited resolutions, which need to be further warped to match the original faces in the source video. | N/A | [![stars](https://badgen.net/github/stars/danmohaha/CVPRW2019_Face_Artifacts)](https://github.com/danmohaha/CVPRW2019_Face_Artifacts)|


#### 📃 Papers

* [Exposing DeepFake Videos By Detecting Face Warping Artifacts](http://www.cs.albany.edu/~lsw/papers/cvprw19a.pdf)
* [DeepFakes: a New Threat to Face Recognition? Assessment and Detection](https://arxiv.org/pdf/1812.08685.pdf)
* [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/pdf/1901.08971.pdf)
* [Deepfake Video Detection Using Recurrent Neural Networks](https://engineering.purdue.edu/~dgueraco/content/deepfake.pdf)
* [Deep Learning for Deepfakes Creation and Detection](https://arxiv.org/pdf/1909.11573.pdf)
* [Protecting World Leaders Against Deep Fakes](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Media%20Forensics/Agarwal_Protecting_World_Leaders_Against_Deep_Fakes_CVPRW_2019_paper.pdf)
* [CAPSULE-FORENSICS: USING CAPSULE NETWORKS TO DETECT FORGED IMAGES AND VIDEOS](https://arxiv.org/pdf/1810.11215.pdf)

#### 😎 Awesome

* [Awesome-Deepfakes-Materials](https://github.com/datamllab/awesome-deepfakes-materials)
