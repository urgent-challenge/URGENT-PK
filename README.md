# URGENT-PK: Perceptually-Aligned Ranking Model Designed for SE Competition

>**Abstract:**<br>
The Mean Opinion Score (MOS) is fundamental to speech quality assessment. However, its acquisition requires significant human annotation. Although deep neural network approaches, such as DNSMOS and UTMOS, have been developed to predict MOS to avoid this issue, they often suffer from insufficient training data. Recognizing that the comparison of speech enhancement (SE) systems prioritizes a reliable system comparison over absolute scores, we propose URGENT-PK, a novel ranking approach leveraging pairwise comparisons. URGENT-PK takes homologous enhanced speech pairs as input to predict relative quality rankings. This pairwise paradigm efficiently utilizes limited training data, as all pairwise permutations of multiple systems constitute a training instance. Experiments across multiple open test sets demonstrate URGENT-PK's superior system-level ranking performance over state-of-the-art baselines, despite its simple network architecture and limited training data.

![image](local/model.png)

## To train the URGENT-PK model

### An example

```
python abtest_model/train_ab.py --dataset urgent24 --score_diff_thres 0.30 --encoder mel --backbone resnet34 --tune_utmos False
```

### Important arguments:

* `--dataset`: the training dataset, you can customize the names in `abtest_model/train_ab.py`
    
* `--score_diff_thres`: the 'MOS difference threshold' in data cleaning, only speech pairs with a MOS difference larger than this threshold are included in the training data.

* `--encoder`: the encoder in the model, so far you can choose `mel` `utmos` `stft`.

* `--backbone`: the comparing module in the model, currently we only use `resnet34`, you can build your own model.

* `--tune_utmos`: whether to freeze or finetune the utmos encoder, only works when `--encoder utmos`.

## To test the URGENT-PK model

### An example:

```
python abtest_model/rank.py --ckpt_path latest.ckpt --dataset urgent25_en --subset test --score_mode logit_binary_vote
```

### Important arguments:

* `--ckpt_path`: path to the checkpoint.

* `--subset`: on which subset to perform the inference, should be `valid` or `test`.

* `--dataset`: the testing dataset, you can customize the names in `abtest_model/rank.py`

* `--score_mode`: the strategy to assgin scores to each system, see the following options:

    * `logit_binary_vote`: the **Binary Scoring** Strategy

    * `logit_non_binary_vote`: the **non-Binary Scoring** Strategy

    * `get_mos_dup`: the **Replication Strategy** in the **ablation study** on the Predicted MOS

    * `get_mos_noise`: the **Noisy-Speech Strategy** in the **ablation study** on the Predicted MOS

## To build your own dataset for URGENT-PK

Each dataset is loaded by an abstract class `ABDatset()` in `abtest_model/ab_dataset.py`. This abstract class prepares data by returning a pair of dictionaries that contain the detailed information of the dataset, either for the training stage or the testing stage.

### Prepare the training dataset

The `ABDatset()` prepares the training data in the `__init__()` method and prepares the validation data in the `load_wavs_for_valid()` method. In `__init__()`, a pair of lists `self.wav_pairs` and `self.mos_pairs` are organized to prepare the training data. `self.wav_pairs` contains pairs of speech file paths and `self.mos_pairs` containins corresponding pairs of MOS. In `load_wavs_for_valid()`, a pair of dictionaries `teams_wav` and `teams_mos` are organized to prepare the validation data. `teams_wav` contains teams' IDs as keys and their scp dictionaries as values, like `{teamID: {uttID: file_path}}`. `teams_mos` is the corresponding dictionary of MOS, like `{teamID: {uttID: MOS}}`.

After preparing the `ABDatset()` class, remember to associate the the dataset names and your `ABDatset()` class in `abtest_model/train_ab.py`, as well as to associate the the dataset names and your `ABDatset().load_wavs_for_valid()` method in `abtest_model/rank.py`.

### Prepare the testing dataset

Preparing the testing dataset is just like pareparing the validation data, however used for testing. You can create a `load_wavs_for_test()` method and follow the same operations in preparing the validation data, organizing the pair of dictionaries `teams_wav` and `teams_mos`. After that, remember to associate the the dataset names and your `ABDatset().load_wavs_for_test()` method in `abtest_model/rank.py`.
