# VLM-Grocery-Fine-Tuning

## Datasets
- Freiburg Groceries Dataset
- MVTec D2S Dataset
- Grocery Store Dataset
    - https://github.com/marcusklasson/GroceryStoreDataset?utm_source=chatgpt.com
- GroZi120
- Grozi-3.2K
- RPC: A Large-Scale Retail Product Checkout Dataset
    - https://rpc-dataset.github.io/
- Locount Dataset
    - https://isrc.iscas.ac.cn/gitlab/research/locount-dataset
- RP2K
    - https://www.pinlandata.com/rp2k_dataset/

Fruits and Vegetables:
- https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition?utm_source=chatgpt.com

## Image Labeling

- Pypi: labelImg

## Categories

- 

## Dataset Labeling and Pricing

- Adding category count for Freiburg Dataset with openai: 
    - prompt_tokens=8536
    - model: gpt-4o-mini
    - price per 256x256 image: $0.001275
    - 5000 * 0.001275 $ = 6.375

- Custom units of measure of groceries:
    - piece
    - gram
    - ml
    - can
    - bottle
    - pack
    - bag