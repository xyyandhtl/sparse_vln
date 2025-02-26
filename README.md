# vln/vlm & sparse reconstruction

applications to combine low-cost visual sfm and visual-language maps/navigation

### dependence weights
downlaod from [huggingface](https://huggingface.co/) and put them in `./weights/`
```shell
tree ./weights/ 
./weights/
├── clip-vit-base-patch32
├──├── config.json
├──├── merges.txt
├──├── preprocessor_config.json
├──├── pytorch_model.bin
├──├── special_tokens_map.json
├──├── tokenizer_config.json
├──├── tokenizer.json
├──└── vocab.json
├── depth_anything_v2
├──├── config.json
├──├── model.safetensors
├──├── preprocessor_config.json
├──└── README.md
├── dino_tiny
├──├── added_tokens.json
├──├── config.json
├──├── model.safetensors
├──├── preprocessor_config.json
├──├── pytorch_model.bin
├──├── special_tokens_map.json
├──├── tokenizer_config.json
├──├── tokenizer.json
├──└── vocab.txt
├── lseg
├──└── demo_e200.ckpt
└── sam_base
    ├── config.json
    ├── preprocessor_config.json
    └── pytorch_model.bin
```

todo: impl vln models on sparse vlmap refer to InstructNav/BevBert