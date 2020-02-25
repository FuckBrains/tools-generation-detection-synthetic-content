# Tools for generating and detecting synthetic content

<p align="center">
  <img src="logo.png" />
</p>

## 🔖 Table of contents


* [Generation of synthetic content](#generation-of-synthetic-content)
  * [🔤 Text](#generation-text)
  * [🔊 Audio](#generation-audio)
  * [📷 Images](#generation-images)
  * [🎥 Video](#generation-videos)
* [Detection of synthetic content](#detection-of-synthetic-content)
  * [🔤 Text](#generation-text)
  * [🔊 Audio](#generation-audio)
  * [📷 Images](#generation-images)
  * [🎥 Video](#generation-videos)
* [Misc](#misc)
  * [Articles](#articles)
  * [Talks](#talks)
  * [Challenges](#challenges)
  * [Forums](#forums)


## Generation of synthetic content

### Generation Text

#### ⚒️ Tools ⚒️

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
| [Grover](https://github.com/rowanz/grover) | Grover is a model for Neural Fake News -- both generation and detection. However, it probably can also be used for other generation tasks. | [https://grover.allenai.org/](https://grover.allenai.org/) | [![stars](https://badgen.net/github/stars/rowanz/grover)](https://github.com/rowanz/grover)|
[gpt-2xy](https://github.com/NaxAlpha/gpt-2xy) | GPT-2 User Interface based on HuggingFace's Pytorch Implementation | https://gpt2.ai-demo.xyz/ | [![stars](https://badgen.net/github/stars/NaxAlpha/gpt-2xy)](https://github.com/NaxAlpha/gpt-2xy) |
[CTRL](https://github.com/salesforce/ctrl) | Conditional Transformer Language Model for Controllable Generation | N/A | [![stars](https://badgen.net/github/stars/salesforce/ctrl)](https://github.com/salesforce/ctrl) |
[Talk to Transformer](https://talktotransformer.com/) | See how a modern neural network completes your text. Type a custom snippet or try one of the examples | https://talktotransformer.com | N/A |
[LEO](http://leoia.es) | First intelligent system for creating news in Spanish | N/A | N/A
[Big Bird](https://bigbird.dev/) | Bird Bird uses State of the Art (SOTA) Natural Language Processing to aid your fact-checked and substantive content. | [BigBirdDemo](https://run.bigbird.dev/auth/login)| N/A

#### 📃 Papers 📃

* [Saliency Maps Generation for Automatic Text Summarization](https://arxiv.org/pdf/1907.05664.pdf)
* [Automatic Conditional Generation of Personalized Social Media Short Texts ](https://arxiv.org/pdf/1906.09324.pdf)
* [Neural Text Generation in Stories Using Entity Representations as Context](https://homes.cs.washington.edu/~eaclark7/www/naacl2018.pdf)
* [DeepTingle](https://arxiv.org/pdf/1705.03557.pdf)

#### 🌐 Webs 🌐

* [NotRealNews](https://notrealnews.net/)
* [BotPoet](http://botpoet.com/vote/sign-post/)
* [TheseLyricsDoNotExist](https://theselyricsdonotexist.com/)
* [ThisResumeDoesNotExist](https://thisresumedoesnotexist.com/)
* [NotRealNews](https://notrealnews.net/)

#### 😎 Awesome 😎

* [awesome-text-generation](https://github.com/ChenChengKuan/awesome-text-generation)

### Generation Audio

#### ⚒️ Tools ⚒️

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
[Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) | Clone a voice in 5 seconds to generate arbitrary speech in real-time | https://www.youtube.com/watch?v=-O_hYhToKoA | [![stars](https://badgen.net/github/stars/CorentinJ/Real-Time-Voice-Cloning)](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
[Lyrebird](https://beta.myvoice.lyrebird.ai/) | Create your own vocal avatar! | N/A | N/A |
[Descrypt](https://www.descript.com/) | Record. Transcribe. Edit. Mix. As easy as typing. | N/A | N/A
[Common Voice](https://voice.mozilla.org/en) | Common Voice is Mozilla's initiative to help teach machines how real people speak. | N/A | N/A
[Resemble.ai](https://www.resemble.ai/) | Resemble can clone any voice so it sounds like a real human. | N/A | N/A
[TacoTron](https://google.github.io/tacotron/) | Tacotron (/täkōˌträn/): An end-to-end speech synthesis system by Google. | [Demo](https://google.github.io/tacotron/publications/prosody_prior/index.html) | [![stars](https://badgen.net/github/stars/google/tacotron)](https://github.com/google/tacotron)


#### 📃 Papers 📃

* [Neural Voice Cloning with a Few Samples](http://research.baidu.com/Blog/index-view?id=81)
* [Data Efficient Voice Cloning for Neural Singing Synthesis](https://mtg.github.io/singing-synthesis-demos/voice-cloning/)
* [Efficient Neural Audio Synthesis](https://arxiv.org/pdf/1802.08435v1.pdf)
* [Score and Lyrics-free Singing Voice Generation](https://arxiv.org/pdf/1912.11747.pdf)
* [Generating diverse and natural Text-to-Speech samples using a quantized fine-grained vae and autoregressive prosody prior](https://arxiv.org/pdf/2002.03788.pdf)

### Generation Images

#### ⚒️ Tools ⚒️

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
[StyleGAN](https://github.com/NVlabs/stylegan) | An alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. The new architecture leads to an automatically learned, unsupervised separation of high-level attributes (e.g., pose and identity when trained on human faces) and stochastic variation in the generated images (e.g., freckles, hair), and it enables intuitive, scale-specific control of the synthesis. The new generator improves the state-of-the-art in terms of traditional distribution quality metrics, leads to demonstrably better interpolation properties, and also better disentangles the latent factors of variation. | https://www.youtube.com/watch?v=kSLJriaOumA | [![stars](https://badgen.net/github/stars/NVlabs/stylegan)](https://github.com/NVlabs/stylegan)
[StyleGAN2](https://github.com/NVlabs/stylegan2) | Improved version for StyleGAN. | https://www.youtube.com/watch?v=c-NJtV9Jvp0 | [![stars](https://badgen.net/github/stars/NVlabs/stylegan2)](https://github.com/NVlabs/stylegan2)
| [DG-Net](https://github.com/NVlabs/DG-Net) | Joint Discriminative and Generative Learning for Person Re-identification | https://www.youtube.com/watch?v=ubCrEAIpQs4 | [![stars](https://badgen.net/github/stars/NVlabs/DG-Net)](https://github.com/NVlabs/DG-Net)

#### 📃 Papers 📃

* [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948.pdf)
* [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/pdf/1912.04958.pdf)
* [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/pdf/1711.11585.pdf)
* [Complement Face Forensic Detection and Localization with Facial Landmarks](https://arxiv.org/pdf/1910.05455.pdf)
* [Joint Discriminative and Generative Learning for Person Re-identification](https://arxiv.org/pdf/1904.07223.pdf)
* [Image2StyleGAN++: How to Edit the Embedded Images?](https://arxiv.org/pdf/1911.11544.pdf)

#### 🌐 Webs 🌐

* [ThisPersonDoesNotExist](http://www.thispersondoesnotexist.com/)
* [WhichFaceIsReal](http://www.whichfaceisreal.com/)
* [ThisRentalDoesNotExist](https://thisrentaldoesnotexist.com/)
* [ThisCatDoesNotExist](https://thiscatdoesnotexist.com/)
* [ThisWaifuDoesNotExist](https://www.thiswaifudoesnotexist.net/)
* [thispersondoesnotexist](http://www.thispersondoesnotexist.com/)

### Generation Videos

#### ⚒️ Tools ⚒️

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
| [FaceSwap](https://github.com/deepfakes/faceswap) | Grover is a model for Neural Fake News -- both generation and detection. However, it probably can also be used for other generation tasks. | https://www.youtube.com/watch?v=r1jng79a5xc | [![stars](https://badgen.net/github/stars/deepfakes/faceswap)](https://github.com/deepfakes/faceswap)|
[Face2Face](https://github.com/datitran/face2face-demo) | FaceSwap is a tool that utilizes deep learning to recognize and swap faces in pictures and videos. | N/A | [![stars](https://badgen.net/github/stars/datitran/face2face-demo)](https://github.com/datitran/face2face-demo) |
[Faceswap](https://github.com/MarekKowalski/FaceSwap) | FaceSwap is an app that I have originally created as an exercise for my students in "Mathematics in Multimedia" on the Warsaw University of Technology. | N/A | [![stars](https://badgen.net/github/stars/MarekKowalski/FaceSwap)](https://github.com/MarekKowalski/FaceSwap) |
[Faceswap-GAN](https://github.com/shaoanlu/faceswap-GAN) | Adding Adversarial loss and perceptual loss (VGGface) to deepfakes'(reddit user) auto-encoder architecture. | https://github.com/shaoanlu/faceswap-GAN/blob/master/colab_demo/faceswap-GAN_colab_demo.ipynb | [![stars](https://badgen.net/github/stars/shaoanlu/faceswap-GAN)](https://github.com/shaoanlu/faceswap-GAN) |
[DeepFaceLab](https://github.com/iperov/DeepFaceLab) | DeepFaceLab is a tool that utilizes machine learning to replace faces in videos. | https://www.youtube.com/watch?v=um7q--QEkg4 | [![stars](https://badgen.net/github/stars/iperov/DeepFaceLab)](https://github.com/iperov/DeepFaceLab)|
[Vid2Vid](https://github.com/NVIDIA/vid2vid) | Pytorch implementation for high-resolution (e.g., 2048x1024) photorealistic video-to-video translation.  | https://www.youtube.com/watch?v=5zlcXTCpQqM | [![stars](https://badgen.net/github/stars/NVIDIA/vid2vid)](https://github.com/NVIDIA/vid2vid)|
[DFaker](https://github.com/dfaker/df) | Pytorch implementation for high-resolution (e.g., 2048x1024) photorealistic video-to-video translation.  | N/A | [![stars](https://badgen.net/github/stars/dfaker/df)](https://github.com/dfaker/df)|


#### 📃 Papers 📃

* [HeadOn: Real-time Reenactment of Human Portrait Videos](https://arxiv.org/pdf/1805.11729.pdf)
* [Face2Face: Real-time Face Capture and Reenactment of RGB Videos](http://gvv.mpi-inf.mpg.de/projects/MZ/Papers/CVPR2016_FF/page.html)
* [Synthesizing Obama: Learning Lip Sync from Audio](https://grail.cs.washington.edu/projects/AudioToObama/siggraph17_obama.pdf)

#### 🌐 Webs 🌐

* [DeepFake中文网](https://www.deepfaker.xyz/) :cn:
* [Website for creating deepfake videos with learning](https://deepfakesapp.online/)
* [Deep Fakes Net - Deepfakes Network](https://deep-fakes.net/)
* [Faceswap is the leading free and Open Source multi-platform Deepfakes software](https://faceswap.dev/)

## Detection of synthetic content

### Detection Text

#### ⚒️ Tools ⚒️

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
| [Grover](https://github.com/rowanz/grover) | Grover is a model for Neural Fake News -- both generation and detection. However, it probably can also be used for other generation tasks. | [https://grover.allenai.org/](https://grover.allenai.org/) | [![stars](https://badgen.net/github/stars/rowanz/grover)](https://github.com/rowanz/grover)|
[GLTR](https://github.com/HendrikStrobelt/detecting-fake-text) | Detecting text that was generated from large language models (e.g. GPT-2). | http://gltr.io/dist/index.html | [![stars](https://badgen.net/github/stars/HendrikStrobelt/detecting-fake-text)](https://github.com/HendrikStrobelt/detecting-fake-text) |
[fake news detection](https://github.com/nguyenvo09/fake_news_detection_deep_learning) | In this project, we aim to build state-of-the-art deep learning models to detect fake news based on the content of article itself. | [Demo](https://github.com/nguyenvo09/fake_news_detection_deep_learning/blob/master/biGRU_attention.ipynb) | [![stars](https://badgen.net/github/stars/nguyenvo09/fake_news_detection_deep_learning)](https://github.com/nguyenvo09/fake_news_detection_deep_learning) |

#### 📃 Papers 📃

* [GLTR: Statistical Detection and Visualization of Generated Text](https://arxiv.org/pdf/1906.04043.pdf)
* [Human and Automatic Detection of Generated Text](https://arxiv.org/pdf/1911.00650.pdf)
* [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/pdf/1909.05858.pdf)
* [The Limitations of Stylometry for Detecting Machine-Generated Fake News](https://arxiv.org/pdf/1908.09805.pdf)

### Detection Audio

#### ⚒️ Tools ⚒️

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
| [Spooded speech detection](https://github.com/elleros/spoofed-speech-detection) | This work is part of the "DDoS Resilient Emergency Dispatch Center" project at the University of Houston, funded by the Department of Homeland Security (DHS). | N/A | [![stars](https://badgen.net/github/stars/elleros/spoofed-speech-detection)](https://github.com/elleros/spoofed-speech-detection)|
[Fake voice detection](https://github.com/dessa-public/fake-voice-detection) | This repository provides the code for a fake audio detection model built using Foundations Atlas. It also includes a pre-trained model and inference code, which you can test on any of your own audio files. | N/A | [![stars](https://badgen.net/github/stars/dessa-public/fake-voice-detection)](https://github.com/dessa-public/fake-voice-detection)
[Fake Voice Detector](https://github.com/kstoneriv3/Fake-Voice-Detection) | For "Deep Learning class" at ETHZ. Evaluate how well the fake voice of Barack Obama 1. confuses the voice verification system, 2. can be detected. | N/A | [![stars](https://badgen.net/github/stars/kstoneriv3/Fake-Voice-Detection)](https://github.com/kstoneriv3/Fake-Voice-Detection)
[CycleGAN Voice Converter](https://leimao.github.io/project/Voice-Converter-CycleGAN/) | An implementation of CycleGAN on human speech conversions | https://leimao.github.io/project/Voice-Converter-CycleGAN/ | [![stars](https://badgen.net/github/stars/leimao/Voice_Converter_CycleGAN)](https://github.com/leimao/Voice_Converter_CycleGAN)


#### 📃 Papers 📃

* [Can We Detect Fake Voice Generated by GANs?](https://github.com/kstoneriv3/Fake-Voice-Detection/blob/master/DLproject_fake_voice_detection.pdf)
* [CycleGAN Voice Converter](https://leimao.github.io/project/Voice-Converter-CycleGAN/)

### Detection Images

#### ⚒️ Tools ⚒️

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
| [FALdetector](https://github.com/peterwang512/FALdetector) | Detecting Photoshopped Faces by Scripting Photoshop. | https://www.youtube.com/watch?v=TUootD36Xm0 | [![stars](https://badgen.net/github/stars/peterwang512/FALdetector)](https://github.com/peterwang512/FALdetector)|

#### 📃 Papers 📃

* [Detecting Photoshopped Faces by Scripting Photoshop](https://arxiv.org/pdf/1906.05856.pdf)

### Detection Videos

#### ⚒️ Tools ⚒️

| Name | Description | Demo | Popularity |
| ---------- | :---------- | :---------- | :----------: |
| [FaceForensics++](https://github.com/ondyari/FaceForensics) | FaceForensics++ is a forensics dataset consisting of 1000 original video sequences that have been manipulated with four automated face manipulation methods: Deepfakes, Face2Face, FaceSwap and NeuralTextures. | https://www.youtube.com/watch?v=x2g48Q2I2ZQ | [![stars](https://badgen.net/github/stars/ondyari/FaceForensics)](https://github.com/ondyari/FaceForensics)|
| [Face Artifacts](https://github.com/danmohaha/CVPRW2019_Face_Artifacts) | Our method is based on the observations that current DeepFake algorithm can only generate images of limited resolutions, which need to be further warped to match the original faces in the source video. | N/A | [![stars](https://badgen.net/github/stars/danmohaha/CVPRW2019_Face_Artifacts)](https://github.com/danmohaha/CVPRW2019_Face_Artifacts)|
[DeepFake-Detection](https://github.com/dessa-public/DeepFake-Detection) | Our Pytorch implementation, conducts extensive experiments to demonstrate that the datasets produced by Google and detailed in the FaceForensics++ paper are not sufficient for making neural networks generalize to detect real-life face manipulation techniques. | http://deepfake-detection.dessa.com/projects | [![stars](https://badgen.net/github/stars/dessa-public/DeepFake-Detection)](https://github.com/dessa-public/DeepFake-Detection)|
[Capsule-Forensics-v2](https://github.com/nii-yamagishilab/Capsule-Forensics-v2) | Implementation of the paper: Use of a Capsule Network to Detect Fake Images and Videos. | N/A | [![stars](https://badgen.net/github/stars/nii-yamagishilab/Capsule-Forensics-v2)](https://github.com/nii-yamagishilab/Capsule-Forensics-v2)|
[ClassNSeg](https://github.com/nii-yamagishilab/ClassNSeg) | Implementation of the paper: Multi-task Learning for Detecting and Segmenting Manipulated Facial Images and Videos (BTAS 2019). | N/A | [![stars](https://badgen.net/github/stars/nii-yamagishilab/ClassNSeg)](https://github.com/nii-yamagishilab/ClassNSeg)|
| [fakeVideoForensics](https://github.com/next-security-lab/fakeVideoForensics) | Fake video detector | https://www.youtube.com/watch?v=8YYRT4lzQgY | [![stars](https://badgen.net/github/stars/next-security-lab/fakeVideoForensics)](https://github.com/next-security-lab/fakeVideoForensics)


#### 📃 Papers 📃

* [Exposing DeepFake Videos By Detecting Face Warping Artifacts](http://www.cs.albany.edu/~lsw/papers/cvprw19a.pdf)
* [DeepFakes: a New Threat to Face Recognition? Assessment and Detection](https://arxiv.org/pdf/1812.08685.pdf)
* [FaceForensics++: Learning to Detect Manipulated Facial Images](https://arxiv.org/pdf/1901.08971.pdf)
* [Deepfake Video Detection Using Recurrent Neural Networks](https://engineering.purdue.edu/~dgueraco/content/deepfake.pdf)
* [Deep Learning for Deepfakes Creation and Detection](https://arxiv.org/pdf/1909.11573.pdf)
* [Protecting World Leaders Against Deep Fakes](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Media%20Forensics/Agarwal_Protecting_World_Leaders_Against_Deep_Fakes_CVPRW_2019_paper.pdf)
* [Capsule-Forensics: Using Capsule Networks to Detect Forged Images and Videos](https://arxiv.org/pdf/1810.11215.pdf)
* [DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection](https://arxiv.org/abs/2001.00179)
* [Media Forensics and DeepFakes:
an overview](https://arxiv.org/pdf/2001.06564.pdf)
* [Everybody’s Talkin’: Let Me Talk as You Want](https://arxiv.org/pdf/2001.05201.pdf)
* [FSGAN: Subject Agnostic Face Swapping and Reenactment](https://arxiv.org/pdf/1908.05932.pdf)
* [Celeb-DF (v2): A New Dataset for DeepFake Forensics](https://arxiv.org/pdf/1909.12962.pdf)
* [Deepfake Video Detection through Optical Flow based CNN](http://openaccess.thecvf.com/content_ICCVW_2019/papers/HBU/Amerini_Deepfake_Video_Detection_through_Optical_Flow_Based_CNN_ICCVW_2019_paper.pdf)
* [MesoNet: a Compact Facial Video Forgery Detection Network](https://arxiv.org/pdf/1809.00888.pdf)

#### 😎 Awesome 😎

* [Awesome-Deepfakes-Materials](https://github.com/datamllab/awesome-deepfakes-materials)

## Misc

### Articles

* [2020 Guide to Synthetic Media](https://blog.paperspace.com/2020-guide-to-synthetic-media/)
* [Machine Learning Experiments](https://www.linkedin.com/posts/thiago-porto-24004ba8_machinelearning-experiments-deeplearning-ugcPost-6625473356533649408-sl9v/)
* [Building rules in public: Our approach to synthetic & manipulated media](https://blog.twitter.com/en_us/topics/company/2020/new-approach-to-synthetic-and-manipulated-media.html)
* [Contenido Sintético (parte I): generación y detección de audio y texto](https://www.bbvanexttechnologies.com/contenido-sintetico-parte-i-generacion-y-deteccion-de-audio-y-texto/) :es:

### Talks

* [ICML 2019 Synthetic Realities](https://sites.google.com/view/audiovisualfakes-icml2019/)
* [CCN-CERT: Automatizando la detección de contenido Deep Fake](https://www.youtube.com/watch?v=ist4Za3C2DY) :es:
* [TED Talk: Fake videos of real people](https://www.youtube.com/watch?v=o2DDU4g0PRo)
* [Hacking with Skynet](https://www.slideshare.net/GTKlondike/hacking-with-skynet-how-ai-is-empowering-adversaries)

### Challenges

* [NIST: Media Forensics Challenge 2019](https://www.nist.gov/itl/iad/mig/media-forensics-challenge-2019-0)
* [ASVspoof: Automatic Speaker Verification](https://www.asvspoof.org/)
* [Kaggle: DeepFake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge)
* [Fake News Challenge](http://www.fakenewschallenge.org/)
* [Xente: Fraud detection challenge](https://zindi.africa/competitions/xente-fraud-detection-challenge)

### Forums

* [Reddit: MediaSynthesis](https://www.reddit.com/r/MediaSynthesis/)
* [Reddit: Digital Manipulation](https://www.reddit.com/r/Digital_Manipulation/)
* [MrDeepFake Forums](https://mrdeepfakes.com/forums/) 🔞
* [AIVillage](aivillage.slack.com)

## Contributors

<table>
  <tr>
    <td align="center"><a href="https://github.com/Miguel000"><img src="https://avatars2.githubusercontent.com/u/13256426?s=460&v=4" width="150;" alt=""/><br /><sub><b>Miguel Hernández</b></sub></a></td>
    <td align="center"><a href="https://github.com/jiep"><img src="https://avatars2.githubusercontent.com/u/414463?s=460&v=4" width="150px;" alt=""/><br /><sub><b>José Ignacio Escribano</b></sub></a></td>
  </tr>
</table>

## License

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

* [Creative Commons Attribution Share Alike 4.0 International](LICENSE)

## Logo

* Made with [draw.io](https://www.draw.io).
