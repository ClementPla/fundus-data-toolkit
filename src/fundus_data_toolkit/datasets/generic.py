from nntools.dataset import ImageDataset

from fundus_data_toolkit.utils.composer import get_generic_composer


def get_image_dataset(root, shape, setup_composer=True, precise_autocrop=False):
    
    
    dataset = ImageDataset(root, shape, recursive_loading=True, auto_pad=True, auto_resize=True)
    
    if setup_composer:
        composer = get_generic_composer(shape, precise=precise_autocrop)
        dataset.composer = composer
    
    return dataset
