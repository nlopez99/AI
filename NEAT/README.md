# NEAT Mario
> Uses a Recurrent Neural Network and Neural Evolution of Augmenting Topologies to play Super Mario Bros (NES).

![python-image] 
![python-build]

This software utilizes the Super Mario Bros OpenAI gym environment which can be found here: https://github.com/Kautenja/gym-super-mario-bros. This implementation of NEAT is run with parallel instances to minimize training time. Each is parallel instance is created using the multiprocessing module, and is managed with a queue. In the mario_config_feedforward file, is all the settings for NEAT's algorithm which can be adjusted for experimentation and curiosity. The default population size for each generation is 150, and resets on extinction if the genomes stagnate.

## Installation

OS X & Linux:

```sh
git clone https://github.com/nlopez99/AI.git
cd NEAT/
pip install -r requirements.txt
```
## To Run The Software
```sh
python3 neat_mario.py
```

## Meta
Business Email –  antonino.lopez@spartans.ut.edu

Github - [https://github.com/nlopez99/](https://github.com/nlopez99/)

<!-- Markdown link & img dfn's -->
[python-image]: https://img.shields.io/pypi/pyversions/astsearch.svg
[python-build]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
