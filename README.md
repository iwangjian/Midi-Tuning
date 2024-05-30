# Midi-Tuning

[*Instruct Once, Chat Consistently in Multiple Rounds*: An Efficient Tuning Framework for Dialogue](https://arxiv.org/abs/2402.06967) (ACL 2024)

We propose an efficient Multi-round Interactive Dialogue Tuning (Midi-Tuning) framework. It models the agent and user individually with two adapters built upon large language models. The adapters make use of respective utterances round by round in alternating order and they are tuned via a round-level memory caching mechanism.

<p align="center">
<img src="figure/overview.png" width="98%" />
</p>


## Requirements
```bash
pip install -r requirements.txt
```

## Datasets

Coming soon ...


## Quickstart

Coming soon ...


## Citation
If you find our code useful for your work, please kindly cite our work as:
```bibtex
@inproceedings{wang-etal-2024-instruct,
  title={Instruct Once, Chat Consistently in Multiple Rounds: An Efficient Tuning Framework for Dialogue},
  author={Wang, Jian and 
  Leong, Chak Tou and 
  Wang, Jiashuo and 
  Lin, Dongding and 
  Li, Wenjie and 
  Wei, Xiao-Yong},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2024}
}
```