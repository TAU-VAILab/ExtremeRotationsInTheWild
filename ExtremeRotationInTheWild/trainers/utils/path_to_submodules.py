import sys
import os.path as path
HERE_PATH = path.normpath(path.dirname(__file__))
LOFTR_PATH = path.normpath(path.join(HERE_PATH, '../../../LoFTR/'))
SEG_FORMER_PATH = path.normpath(path.join(HERE_PATH, '../../../semantic-segmentation/'))
SEGFORMER_MODEL_PATH = path.normpath(path.join(SEG_FORMER_PATH, 'checkpoints/segformer.b3.ade.pth'))
LOFTR_MODEL_PATH = path.normpath(path.join(LOFTR_PATH, 'pretrained/outdoor_ds.ckpt'))

for submodule_path in [LOFTR_PATH, SEG_FORMER_PATH]:
    if path.isdir(submodule_path):
        sys.path.insert(0, submodule_path)
    else:
        raise ImportError(f"submodule_path is not initialized, could not find: {submodule_path}.\n "
                          "Did you forget to run 'git submodule update --init --recursive' ?")

for model_path in [SEGFORMER_MODEL_PATH, LOFTR_MODEL_PATH]:
    if not path.isfile(model_path):
        raise ImportError(f"model_path is not initialized, could not find: {model_path}.\n "
                          "Did you forget to download the model weights?")