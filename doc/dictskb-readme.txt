
######## repo acquisition

git clone https://github.com/thunlp/DictSKB.git
cd DictSKB

######## enivronment setup

# mambaforge was used for environment management, but conda etc. should also work.
# add the following configurations to the conda configuration (.condarc):

channel_priority: strict
channels:
  - conda-forge

# dictskb-aa: used for following tasks:
# - adversarial attack (aa)
# - consistency check (cc)
# - sentiment analysis (cr)
# - language inference (nli)

mamba create --name dictskb-aa tensorflow-gpu=2.6.0 keras=2.6 scikit-learn=1.0.0 anytree=2.8.0 nltk=3.5 nltk_data=2019.07.04 loguru=0.3.2 pytorch-gpu=1.6.0
mamba activate dictskb-aa
pip install --upgrade pytorch-transformers==1.0.0
pip install --upgrade OpenHowNet==1.0
python -c "import OpenHowNet; OpenHowNet.download()"

# dictskb-lm: used for following tasks:
# - language modelling (lm)

mamba create --name dictskb-lm -c anaconda pytorch=1.0.1 scikit-learn=1.0.2 nltk=3.5 nltk_data=2019.07.04
mamba activate dictskb-lm

# some useful mamba commands

mamba info --envs
mamba remove --name "env-name-to-del" --all
mamba rename --name "env-name-old" "env-name-new"

######## external data setup

# dictskb-aa
subdir="AdversarialAttack/SST/"
urlbase="https://nlp.stanford.edu/software"
zipname="stanford-postagger-full-2018-10-16.zip"
wget -P "${subdir}" "${urlbase}/${zipname}"
unzip "${subdir}/${zipname}" -d "${subdir}"

# dictskb-nli
subdir="NLI/dataset/GloVe/"
urlbase="http://nlp.stanford.edu/data"
zipname="glove.840B.300d.zip"
wget -P "${subdir}" "${urlbase}/${zipname}"
unzip "${subdir}/${zipname}" -d "${subdir}"

######## patches

# commmand used for creating the patch file. only for reference.
git diff -- \
"AdversarialAttack/SST/SST_BERT.py" \
"AdversarialAttack/SST/train_model.py" \
"LM_SDLM/embed_regularize.py" \
"LM_SDLM/run_awd_lstm.py" \
"LM_SDLM/run_tied_lstm.py" \
> "dictskb.patch"

# apply the patch (using gnu-patch).
patch -p1 < "dictskb.patch"
# apply the patch (using git).
git apply "dictskb.patch"
# adding the following rename operation to the patch file was troublesome, so it is separated.
mv "LM_SDLM/data/sememe_dict.uncased.npy" "LM_SDLM/data/sememe_dict_uncased.npy"

######## experiments

# refer to the respective readme files under each task.

########
