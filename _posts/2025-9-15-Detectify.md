---
title: Detectify - an app to detect deepfakes !!
date: 2025-09-15
categories: [project, active]
tags: [project, technology, detectify]
---
## Detectify - an app to detect deepfakes !!

Artificial Intelligence is a double-edged sword, as much as it can help us there are crucial problems with it.
One of them being deepfakes. Recently there has been a rise in multiple deepfake cases.

## Need for Detectify.

1) Organizations like news outlets are facing problems verifying if a video is real or a deepfake. For e.g. a recent case of the Pentagon being attacked made it to the news, only to be later debunked as a case of deepfake images.

2) With the rise in deepfakes, extortion/blackmailing cases have increased. You can go through a report that describes how a 16-year-old boy committed suicide because of realistic deepfakes
https://www.thesun.co.uk/news/36269842/ai-deepfake-schools-app-children/?utm_source=chatgpt.com

3) It will be harder for the police to maintain law and order as anyone can make horrifying real deepfakes of anyone.

4) Social media platforms like YouTube, Instagram, Facebook etc. need to start marking their content as real or deepfake for their audience to have a better understanding of the content they are watching

## How we plan to do it :
1) I won’t be going in depth as this is the first post but the basic principle is as follows :

2) We will analyze the video, extract audio and make a spectrogram of it.

3) We will divide the video into frames and extract information from each frame

4) We will look for anomalies between frame N and frame N-1, like movement of hands, fingers in hand, if it is natural or not

5) Based on our analysis we will conclude (in %) if a video is a deepfake or not.

In my future Posts I will be going deeper with what modules (OpenCV), learning patterns (RNN, CNN) etc. I will use in my project. Feel free to mail me (at the bottom left corner). Until then bye bye see ya.