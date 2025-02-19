---
license: mit
task_categories:
- text-generation
language:
- en
pretty_name: mix-instruct
size_categories:
- 100K<n<1M
---
# MixInstruct

## Introduction
This is the official realease of dataset **MixInstruct** for project **LLM-Blender**.

This dataset contains 11 responses from the current popular instruction following-LLMs that includes:
1. [Stanford Alpaca](https://huggingface.co/chavinlo/alpaca-native)
2. [FastChat Vicuna](https://huggingface.co/eachadea/vicuna-13b-1.1)
3. [Dolly V2](https://huggingface.co/databricks/dolly-v2-12b)
4. [StableLM](https://huggingface.co/stabilityai/stablelm-tuned-alpha-7b)
5. [Open Assistant](https://huggingface.co/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5)
6. [Koala](https://huggingface.co/TheBloke/koala-7B-HF)
7. [Baize](https://huggingface.co/mosesjun0h/llama-7b-hf-baize-lora-bf16)
8. [Flan-T5](https://huggingface.co/google/flan-t5-xxl)
9. [ChatGLM](https://huggingface.co/THUDM/chatglm-6b)
10. [MOSS](https://huggingface.co/fnlp/moss-moon-003-sft)
11. [Moasic MPT](https://huggingface.co/mosaicml/mpt-7b-instruct)

We evaluate each response with auto metrics including BLEU, ROUGE, BERTScore, BARTScore. And provide pairwise comparison results by prompting ChatGPT for the $4771$ examples 
in the test split. (The rest $229$ examples contain contents filtered by the API).

## Data Format
```json
[
    {
        "id": "unified_chip2/69962",
        "instruction": "",
        "input": "I've always wondered what the difference is between a skeptic and a denier.",
        "output": "A skeptic is someone who questions the validity of something, while a denier is someone who outright rejects something without evidence or reason.",
        "candidates": [
            {
                "decoding_method": "top_p_sampling",
                "model": "oasst-sft-4-pythia-12b-epoch-3.5",
                "text": "A skeptic is someone who doubts or expresses ...",
                "scores": {
                    "logprobs": -0.02404022216796875,
                    "bleu": 5.656152750894142,
                    "bertscore": 0.7549101114273071,
                    "rouge1": 0.2857142857142857,
                    "rouge2": 0.1272727272727273,
                    "rougeL": 0.23214285714285715,
                    "rougeLsum": 0.23214285714285715
                }
            },
            ...
        ],
    },
    ...
]
```

Examples evaluted by ChatGPT will contain another filed **cmp_results**.
The options contains:
1. A is better
2. B is better
3. Same good
4. Same bad
```json
"cmp_results": {
    "model_A,model_B": "A is better",
    ...
}, 
```
Each cmp_results field is encoded into a str in a json format. Please first use `json.loads(item['cmp_results'])` to get the cmp_results for each item.
"null" denotes no cmp_results from ChatGPT avaliable.


## Eval Results

### Auto Metrics

- train

| Models (down) / Metircs (right)   | logprobs    | rougeL          | rouge2          | rougeLsum       | rouge1          | bleu            | bertscore       | bleurt          | bartscore    |
|:----------------------------------|:------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:-------------|
| alpaca-native                     | -6.1247     | 0.248           | 0.1414          | 0.2986          | 0.3347          | 8.057           | 0.7196          | -0.5092         | -3.5335      |
| chatglm-6b                        | -10.1263    | 0.2231          | 0.1212          | 0.2743          | 0.3074          | 6.2597          | 0.7043          | -0.6071         | -3.4975      |
| dolly-v2-12b                      | -24.8508    | 0.1245          | 0.0502          | 0.1625          | 0.1836          | 2.1062          | 0.6244          | -0.8562         | -3.8145      |
| flan-t5-xxl                       | -1.0717     | 0.1202          | 0.0456          | 0.1334          | 0.1489          | 1.8418          | 0.6514          | -1.2176         | -4.537       |
| koala-7B-HF                       | -10.8323    | 0.1533          | 0.0683          | 0.1909          | 0.2165          | 3.2848          | 0.6436          | -0.8284         | -3.8326      |
| llama-7b-hf-baize-lora-bf16       | -24.8867    | 0.1539          | 0.0797          | 0.2042          | 0.2276          | 3.4928          | 0.6564          | -0.6575         | -3.496       |
| moss-moon-003-sft                 | -796.1366   | 0.1599          | 0.0898          | 0.2135          | 0.236           | 3.944           | 0.6689          | -0.5617         | -3.3404      |
| mpt-7b                            | -174.1702   | 0.1118          | 0.0447          | 0.1517          | 0.1683          | 1.7698          | 0.618           | -0.9525         | -3.9119      |
| mpt-7b-instruct                   | -156.8005   | 0.1225          | 0.0538          | 0.1669          | 0.1861          | 2.1041          | 0.6327          | -0.8176         | -3.6996      |
| oasst-sft-4-pythia-12b-epoch-3.5  | -4.7714     | 0.2902          | 0.1763          | 0.3447          | 0.386           | 10.6599         | 0.748           | -0.3762         | -3.4221      |
| stablelm-tuned-alpha-7b           | -1268.9396  | 0.1336          | 0.0544          | 0.1714          | 0.1948          | 2.6348          | 0.6355          | -0.9585         | -4.0795      |
| vicuna-13b-1.1                    | -11.1528    | 0.211           | 0.1219          | 0.2671          | 0.3003          | 6.3697          | 0.6928          | -0.6194         | -3.4233      |
| Best Model Metric Perf            | -1.0717     | 0.2902          | 0.1763          | 0.3447          | 0.386           | 10.6599         | 0.748           | -0.3762         | -3.3404      |
| Oracle                            | 0.0         | 0.3611          | 0.2471          | 0.4242          | 0.4706          | 15.8557         | 0.7783          | 0.0723          | 0.0          |
| Oracle-Best_Model Gap             | 1.0717      | 0.0709          | 0.0708          | 0.0794          | 0.0846          | 5.1958          | 0.0303          | 0.4484          | 3.3404       |

- val
| Models (down) / Metircs (right)   | logprobs    | rouge1          | rouge2          | rougeLsum       | rougeL          | bleu            | bertscore       | bleurt          | bartscore      |
|:----------------------------------|:------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:---------------|
| alpaca-native                     | -3.3832     | 0.3342          | 0.1452          | 0.299           | 0.2503          | 8.1749          | 0.7198          | -0.5076         | -3.5517        |
| chatglm-6b                        | -4.7033     | 0.3066          | 0.1216          | 0.2743          | 0.2241          | 6.3323          | 0.7053          | -0.6091         | -3.51          |
| dolly-v2-12b                      | -9.1237     | 0.1843          | 0.0511          | 0.1633          | 0.1254          | 2.1368          | 0.6257          | -0.852          | -3.8121        |
| flan-t5-xxl                       | -1.0077     | 0.1497          | 0.0464          | 0.1342          | 0.1212          | 1.8653          | 0.652           | -1.2089         | -4.5407        |
| koala-7B-HF                       | -6.015      | 0.2154          | 0.068           | 0.1903          | 0.1538          | 3.2596          | 0.6425          | -0.8298         | -3.8456        |
| llama-7b-hf-baize-lora-bf16       | -12.2594    | 0.2261          | 0.0803          | 0.2034          | 0.1543          | 3.5462          | 0.6562          | -0.6604         | -3.4831        |
| moss-moon-003-sft                 | -357.3054   | 0.2053          | 0.0678          | 0.1851          | 0.1361          | 2.9639          | 0.648           | -0.7261         | -3.6317        |
| mpt-7b                            | -171.9416   | 0.1663          | 0.0447          | 0.1499          | 0.1111          | 1.7555          | 0.617           | -0.964          | -3.9189        |
| mpt-7b-instruct                   | -157.1143   | 0.1841          | 0.054           | 0.1652          | 0.1224          | 2.1252          | 0.6307          | -0.8275         | -3.7183        |
| oasst-ft-4-pythia-12b-epoch-3.5   | -1.6194     | 0.3835          | 0.1761          | 0.3434          | 0.2896          | 10.5858         | 0.7479          | -0.378          | -3.4366        |
| stablelm-tuned-alpha-7b           | -869.6767   | 0.192           | 0.0529          | 0.1688          | 0.1317          | 2.5687          | 0.6314          | -0.9618         | -4.1008        |
| vicuna-13b-1.1                    | -5.6143     | 0.3029          | 0.1242          | 0.2701          | 0.2142          | 6.5299          | 0.695           | -0.6212         | -3.4332        |
| Best Model Metric Perf            | -1.0077     | 0.3835          | 0.1761          | 0.3434          | 0.2896          | 10.5858         | 0.7479          | -0.378          | -3.4332        |
| Oracle                            | 0.0         | 0.4712          | 0.2488          | 0.4258          | 0.3642          | 15.9896         | 0.7794          | 0.0726          | 0.0            |
| Oracle-Best_Model Gap             | 1.0077      | 0.0877          | 0.0728          | 0.0824          | 0.0746          | 5.4038          | 0.0315          | 0.4506          | 3.4332         |

- test
| Models (down) / Metircs (right)   | logprobs    | rougeL          | rougeLsum       | rouge1          | rouge2          | bleu            | bertscore       | bleurt          | bartscore      |
|:----------------------------------|:------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:---------------|
| alpaca-native                     | -3.458      | 0.2421          | 0.2915          | 0.3276          | 0.1362          | 7.6478          | 0.7146          | -0.5307         | -3.5696        |
| chatglm-6b                        | -4.7418     | 0.2225          | 0.2734          | 0.3063          | 0.1192          | 6.0493          | 0.7038          | -0.6167         | -3.5193        |
| dolly-v2-12b                      | -9.1266     | 0.1236          | 0.1606          | 0.1811          | 0.0495          | 2.062           | 0.6226          | -0.8654         | -3.8331        |
| flan-t5-xxl                       | -0.9924     | 0.1172          | 0.1296          | 0.1444          | 0.0432          | 1.6066          | 0.6492          | -1.2288         | -4.5717        |
| koala-7B-HF                       | -6.1159     | 0.1507          | 0.1871          | 0.2131          | 0.0662          | 3.0983          | 0.6396          | -0.8354         | -3.8496        |
| llama-7b-hf-baize-lora-bf16       | -11.9519    | 0.1521          | 0.2022          | 0.2253          | 0.0781          | 3.4005          | 0.6557          | -0.663          | -3.526         |
| moss-moon-003-sft                 | -356.8774   | 0.1365          | 0.1863          | 0.2062          | 0.0686          | 2.9561          | 0.6485          | -0.7261         | -3.6461        |
| mpt-7b                            | -176.2144   | 0.1106          | 0.1498          | 0.1663          | 0.0439          | 1.7392          | 0.6165          | -0.9636         | -3.9419        |
| mpt-7b-instruct                   | -156.0153   | 0.121           | 0.1647          | 0.1837          | 0.0524          | 2.0692          | 0.6321          | -0.8232         | -3.7208        |
| oasst-sft-4-pythia-12b-epoch-3.5  | -1.6749     | 0.2873          | 0.341           | 0.3813          | 0.1738          | 10.5046         | 0.7468          | -0.3908         | -3.4486        |
| stablelm-tuned-alpha-7b           | -831.595    | 0.1306          | 0.1672          | 0.1904          | 0.0524          | 2.5044          | 0.6247          | -0.9832         | -4.1208        |
| vicuna-13b-1.1                    | -5.6914     | 0.2122          | 0.2677          | 0.3012          | 0.1223          | 6.3584          | 0.696           | -0.6146         | -3.4368        |
| Best Model Metric Perf            | -0.9924     | 0.2873          | 0.341           | 0.3813          | 0.1738          | 10.5046         | 0.7468          | -0.3908         | -3.4368        |
| Oracle                            | 0.0         | 0.3585          | 0.4201          | 0.466           | 0.2438          | 15.4971         | 0.7767          | 0.0679          | 0.0            |
| Oracle-Best_Model Gap             | 0.9924      | 0.0712          | 0.0791          | 0.0847          | 0.07            | 4.9925          | 0.0299          | 0.4587          | 3.4368         |

### ChatGPT CMPTS (4771 examples)
|    **Methods**    | BERTScore | BARTScore |   BLEURT  | GPT-Rank |  Beat Vic(%)  |   Beat OA(%)  |  Top-1(%)  |  Top-2(%)  |  Top-3(%)  |
|:-----------------:|:---------:|:---------:|:---------:|:--------:|:----------:|:----------:|:----------:|:----------:|:----------:|
|   Open Assistant  | **74.68** |   -3.45   | **-0.39** | **3.90** |  **62.78** |     N/A    |    17.35   |    35.67   |    51.98   |
|       Vicuna      |   69.60   | **-3.44** |   -0.61   |   4.13   |     N/A    |  **64.77** |  **25.47** |  **41.23** |  **52.88** |
|       Alpaca      |   71.46   |   -3.57   |   -0.53   |   4.62   |    56.70   |    61.35   |    15.41   |    29.81   |    44.46   |
|       Baize       |   65.57   |   -3.53   |   -0.66   |   4.86   |    52.76   |    56.40   |    14.23   |    26.91   |    38.80   |
|        moss       |   64.85   |   -3.65   |   -0.73   |   5.09   |    51.62   |    51.79   |    15.93   |    27.52   |    38.27   |
|      ChatGLM      |   70.38   |   -3.52   |   -0.62   |   5.63   |    44.04   |    45.67   |    9.41    |    19.37   |    28.78   |
|       Koala       |   63.96   |   -3.85   |   -0.84   |   6.76   |    39.93   |    39.01   |    8.15    |    15.72   |    22.55   |
|      Dolly v2     |   62.26   |   -3.83   |   -0.87   |   6.90   |    33.33   |    31.44   |    5.16    |    10.06   |    16.45   |
|     Mosaic MPT    |   63.21   |   -3.72   |   -0.82   |   7.19   |    30.87   |    30.16   |    5.39    |    10.61   |    16.24   |
|      StableLM     |   62.47   |   -4.12   |   -0.98   |   8.71   |    21.55   |    19.87   |    2.33    |    4.74    |    7.96    |
|      Flan-T5      |   64.92   |   -4.57   |   -1.23   |   8.81   |    23.89   |    19.93   |    1.30    |    2.87    |    5.32    |
| Oracle(BERTScore) | **77.67** |   -3.17   |   -0.27   |   3.88   |    54.41   |    38.84   |    20.16   |    38.11   |    53.49   |
|   Oracle(BLEURT)  |   75.02   |   -3.15   | **-0.15** |   3.77   |    55.61   |    45.80   |    21.48   |    39.84   |    55.36   |
| Oracle(BARTScore) |   73.23   | **-2.87** |   -0.38   |   3.69   |    50.32   |    57.01   |    26.10   |    43.70   |    57.33   |
|  Oracle(ChatGPT)  |   70.32   |   -3.33   |   -0.51   | **1.00** | **100.00** | **100.00** | **100.00** | **100.00** | **100.00** |
