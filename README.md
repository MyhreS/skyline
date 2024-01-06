# Skyline
This is the codebase for my master thesis, following a naming scheme inspired by cloud formations and the sky.

## Setup
This project uses Python 3.10.11, a requirement due to server constraints for training the models. To ensure consistent results, it's recommended to use this specific version.

### Setting Up Python Version
First, install pyenv if you don't have it already:
```
brew install pyenv
```

Next, install and set the specific Python version:
```
pyenv install 3.10.11
pyenv global 3.10.11
```

Run the following to check the version:
```
python3.10 --version
```

If the command doesn't work, add pyenv to your path:
```
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```
Apply the changes (this example is for zsh):
```
source ~/.zshrc
```

### Setting Up the Virtual Environment
This project uses `venv`. Set it up as follows:
```
python3.10 -m venv skylinevenv
```

Activate the environment:
```
source skylinevenv/bin/activate
```

Install the dependencies:
```
pip install -r requirements.txt
```

### Specify the Path to the Data
To train new models, specify the path to your data. Create a `.env` file in the root directory, including the name and path to the dataset folders.

### Running a File
With the virtual environment activated, run any Python file:
```
python {filename}
```

## Theme: Cloud Formations
Consider these names inspired by cloud formations:
1. **Cirrus**:
   - Feathers
   - HaloEffects
   - JetStreams
2. **Cumulus**:
   - PuffyFeatures
   - Daydreams
   - SkyVistas
3. **Stratus**:
   - Overlays
   - Mists
   - RainDrizzles
4. **Nimbus**:
   - Downpours
   - Snowflakes
   - Thunderbolts
5. **Altocumulus or Altostratus**:
   - MixedBags
   - HorizonLines
   - SkyPatterns
