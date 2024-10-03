## 文件结构及说明
```sh
.
├── README.md
├── task1                           # 任务1
│   ├── gpt2-rlhf.ipynb             # 模型对齐
│   ├── sentiment.ipynb             # 情感分析微调
│   ├── transfer.ipynb              # 风格迁移数据集1
│   ├── transfer1.ipynb             # 风格迁移数据集2
│   └── translation.ipynb           # 机器翻译
└── task2                           # 任务1
    ├── GPT2Model.py                # 所有修改的模型代码，具体见代码中注释
    ├── logs                        # 第二部分log
    │   ├── attention
    │   │   ├── linear-rnn.log
    │   │   ├── linear-sim.log
    │   │   └── nystrom.log
    │   ├── pos embedding
    │   │   └── lpe.log
    │   └── token embedding
    │       ├── sentence.log
    │       └── word.log
    ├── sentiment.py                # 运行情感分类任务
    └── tools                       # mamba相关    
        ├── logger.py
        ├── mamba.py
        ├── misc.py
        └── utils_mamba.py
```

## 代码运行
### task1
- 运行`.ipynb`文件
### task2
- 运行以下命令
    ```sh
    python3 sentiment.py --log_path ./logs/attention/ --log_name nystrom.log --bz 16 --tokenizer gpt2 --attention nystrom 
    ```
    - 其他参数见代码中参数说明
    - 在运行不同的探索任务时需要根据需要注释`sentiment.py`中16-38行代码