# NSFW Image Classification Module

**What this is:**
A pretrained image classifier classfying an input image or images into different categories of explicitness. 

These categories include drawing, neutral, hentai, sexy and porn.

For more information about the details of the categories and how the data are collected, see: https://github.com/alex000kim/nsfw_data_scraper

**How to use:**

First, clone this repo, then:

```
pip install -r nsfw_module_requirements.txt

import nsfw_module

nsfw_output = nsfw_module.predict(image_path)

```

**References:**
https://github.com/GantMan/nsfw_model



