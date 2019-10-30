# Dp4jPractice
Newbie practicing dp4j-1.0.0-beta5

Documents: [my gitio](https://fulltopic.github.io/)

## src/mjclassifier
A simple CNN practice to classify mahjong into categories of circle, bamboo and character.

Refer to [Report](src/main/java/dp4jpractice/org/mjclassifier/Tuning.md)

## src/mjsupervised
A CNN + LSTM + DENSE Computation Graph to learn parameter initiation from Tenhou 九段 player

Refer to [Report](src/main/java/dp4jpractice/org/mjsupervised/Report.md)

## src/mjdrl
Reinforcement learning through Tenhou

### nn/dqn
Dqn learning. Deprecated.

### nn/a3c
A3C learning, refer to [Report](src/main/java/dp4jpractice/org/mjdrl/A3CReport.md)

## Tools

### tools/dataprocess/dbprocess
Convert zip files downloaded from [Tenhou](http://tenhou.net/ranking.html) into Caffe2 readable lmdb files. 
Refer to [README](./src/main/java/dp4jpractice/org/tools/dataprocess/dbprocess/README.md) for details