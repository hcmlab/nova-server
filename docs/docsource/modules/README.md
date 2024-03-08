# Nova Server Modules
## Description
This repository contains python modules that can be used together with [NOVA-Server](https://github.com/hcmlab/nova-server).

## Installation 
Every module is self_contained with a test script in the main function and a requirements.txt file in the respective folder.
Note that you might need update the requirements to your own needs (e.g. when using torch with cuda support).

## Available Models

### Processor

|                     Module                     |                Description                |        Input        |                      Output                      | URL                                                                       |
|:----------------------------------------------:|:-----------------------------------------:|:-------------------:|:------------------------------------------------:|:-------------------------------------------------------------------------:|
|  [germanSentiment](modules/german_sentiment)   | Calculating sentiment for german language |        Text         | Probabilities for pos, neg and neutral sentiment |          https://huggingface.co/oliverguhr/german-sentiment-bert          |
|          [whisperx](modules/whisperx)          |                                           |   Free Annotation   |                    Transcript                    |                    https://github.com/m-bain/whisperX                     |            
|            [emow2v](modules/emow2v)            |                                           |                     |                                                  |                 https://github.com/audeering/w2v2-how-to                  | 
|      [wav2vec_bert_2](modules/w2v_bert_2)      |                                           |                     |                                                  |                 https://huggingface.co/facebook/w2v-bert-2.0              | 
|         [blazeface](modules/blazeface)         |                                           |                     |                                                  |               https://github.com/hollance/BlazeFace-PyTorch               |
|          [facemesh](modules/facemesh)          |                                           |                     |                                                  |               https://github.com/tiqq111/mediapipe_pytorch               |
|      [diaristation](modules/diarisation)       |                                           |                     |                                                  |                                                                           |  
|            [emonet](modules/emonet)            |                                           |                     |                                                  |                  https://github.com/face-analysis/emonet                  |
|         [synchrony](modules/synchrony)         |                                           | Two feature streams |                                                  |                                                                           |
| [stablediffusioncL](modules/stablediffusionxl) |       Prompt based image generation       |        Text         |                                                  |                                                                           |
|         [opensmile](modules/opensmile)         |          Audio feature exraction          |    Audio strean     |                  Feature stream                  |               https://audeering.github.io/opensmile-python/               |
|      [nova_assistant](modules/nova_assistant)       |   Large Language Models for prediction    |     Transcript      |            Discrete / Free Annotation            |                             |


### Trainer
| Module                             | Status | Url                                |
|:-----------------------------------|:------:|:-----------------------------------|


### Utilities
|                          Module                           |                 Description                  |                   Input                    |                     Output                      |                                                      URL |
|:---------------------------------------------------------:|:--------------------------------------------:|:------------------------------------------:|:-----------------------------------------------:|---------------------------------------------------------:|
|           [UC1](chains/nova-server/utility/uc1)           |     Featurewise mean over sliding window     |               Feature stream               |                                                 |                                                          |
|           [UC2](chains/nova-server/utility/uc2)           |     Subtract stream two from stream one      |            Two feature streams             |                                                 |                                                          |
|           [UC3](chains/nova-server/utility/uc3)           |        Sentiment based on german text        |           Free scheme annotation           |                                                 |                                                          |
|          [UC4](chains/nova-server/utility/uc4 )           |           Two audio files separate           | Audio stream and two discrete annotations  |                                                 |                                                          |

