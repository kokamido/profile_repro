Установка в три пипа, так оно не конфликтует по версиям -_-

```
pip install wheel && pip install cython
pip install torch==2.4.1 graphviz==0.20.3 numpy==2.1.1 matplotlib==3.9.2
pip install nemo-toolkit[asr]==1.23.0 huggingface_hub==0.23.2 numpy==1.26.4
```

```shell
python meme.py

> cat gpu_forward_profiling               | sed -E 's/[ ]{2,}/\t/g' |cut -f9 | grep -oP "\d+\.\d+%" | tr -d %| awk '{s+=$1} END {print s}'
199.99

> grep -v "aten::"  gpu_forward_profiling | sed -E 's/[ ]{2,}/\t/g' |cut -f9 | grep -oP "\d+\.\d+%" | tr -d %| awk '{s+=$1} END {print s}'
99.99
```