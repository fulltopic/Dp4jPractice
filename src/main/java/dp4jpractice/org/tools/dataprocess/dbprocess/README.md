# Tenhou Log to LMDB Converter 

* The Caffe2 class generated from [caffe2 proto file](https://github.com/pytorch/pytorch/blob/master/caffe2/proto/caffe2.proto)
* Sequences: 
    1. Extract file
    2. Recover into XML file
    3. Generate sequence of states for each player * each game
    4. Convert state into TensorProtos
    5. Save it into lmdb files
* State

    | Player  | 0 ~ 33       | 34 ~ 67      | 68    | 69  | 70     | 71    |
    |---------|--------------|--------------|-------|-----|--------|-------|
    | Player0 | Remain tiles | Stolen tiles | Reach | Oya | Winner | Dummy |
    | Player0 | Dropped tiles| Dummy        | Dummy |Dummy| Dummy  | Dummy |
    | Player1 | Dropped tiles| Stolen tiles | Reach | Oya | Winner | Dummy |
    | Player2 | Dropped tiles| Stolen tiles | Reach | Oya | Winner | Dummy |
    | Player3 | Dropped tiles| Stolen tiles | Reach | Oya | Winner | Dummy |
     