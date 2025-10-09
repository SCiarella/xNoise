"""
Label vocabularies for different datasets compatible with CLIP.
"""
import os
import ast


def _get_tiny_imagenet_labels():
    """
    Get the 200 class labels for Tiny ImageNet.
    These are WordNet synset IDs mapped to readable names.
    """
    # Mapping from WordNet ID to readable name
    # You can load this from your dataset's words.txt file
    tiny_imagenet_labels = [
        'goldfish', 'European fire salamander', 'bullfrog', 'tailed frog',
        'American alligator', 'boa constrictor', 'trilobite', 'scorpion',
        'black widow spider', 'tarantula', 'centipede', 'goose', 'koala',
        'jellyfish', 'brain coral', 'snail', 'slug', 'sea slug',
        'American lobster', 'spiny lobster', 'black stork', 'king penguin',
        'albatross', 'dugong', 'Chihuahua', 'Yorkshire terrier',
        'golden retriever', 'Labrador retriever', 'German shepherd',
        'standard poodle', 'tabby cat', 'Persian cat', 'Egyptian cat',
        'cougar', 'lion', 'brown bear', 'ladybug', 'fly', 'bee', 'grasshopper',
        'walking stick', 'cockroach', 'mantis', 'dragonfly',
        'monarch butterfly', 'sulphur butterfly', 'sea cucumber', 'guinea pig',
        'hog', 'ox', 'bison', 'bighorn sheep', 'gazelle', 'Arabian camel',
        'orangutan', 'chimpanzee', 'baboon', 'African elephant',
        'lesser panda', 'abacus', 'academic gown', 'aircraft carrier', 'altar',
        'apron', 'backpack', 'bannister', 'barbershop', 'barn', 'barrel',
        'basketball', 'bathtub', 'beach wagon', 'beacon', 'beaker',
        'beer bottle', 'bikini', 'binoculars', 'birdhouse', 'bow tie', 'brass',
        'broom', 'bucket', 'bullet train', 'butcher shop', 'cab', 'cannon',
        'cardigan', 'cash machine', 'CD player', 'chain', 'chest',
        'Christmas stocking', 'cliff dwelling', 'computer keyboard',
        'confectionery', 'convertible', 'crane', 'dam', 'desk', 'dining table',
        'drumstick', 'dumbbell', 'flagpole', 'fountain', 'freight car',
        'frying pan', 'fur coat', 'gasmask', 'go-kart', 'gondola', 'hourglass',
        'iPod', 'jinrikisha', 'kimono', 'lampshade', 'lawn mower', 'lifeboat',
        'limousine', 'magnetic compass', 'maypole', 'military uniform',
        'miniskirt', 'moving van', 'nail', 'neck brace', 'obelisk', 'oboe',
        'organ', 'parking meter', 'pay-phone', 'picket fence', 'pill bottle',
        'plunger', 'pole', 'police van', 'poncho', 'pop bottle',
        "potter's wheel", 'projectile', 'punching bag', 'reel', 'refrigerator',
        'remote control', 'rocking chair', 'rugby ball', 'sandal',
        'school bus', 'scoreboard', 'sewing machine', 'snorkel', 'sock',
        'sombrero', 'space heater', 'spider web', 'sports car',
        'steel arch bridge', 'stopwatch', 'sunglasses', 'suspension bridge',
        'swimming trunks', 'syringe', 'teapot', 'teddy bear', 'thatch',
        'torch', 'tractor', 'triumphal arch', 'trolleybus', 'turnstile',
        'umbrella', 'vestment', 'viaduct', 'volleyball', 'water jug',
        'water tower', 'wok', 'wooden spoon', 'comic book', 'plate',
        'guacamole', 'ice cream', 'ice lolly', 'pretzel', 'mashed potato',
        'cauliflower', 'bell pepper', 'mushroom', 'orange', 'lemon', 'banana',
        'pomegranate', 'meat loaf', 'pizza', 'potpie', 'espresso', 'alp',
        'cliff', 'lakeside', 'seashore', 'acorn'
    ]
    return tiny_imagenet_labels


def _get_cifar100_labels():
    """
    Get all 100 CIFAR-100 class labels.
    """
    cifar100_labels = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
        'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
        'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab',
        'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
        'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
        'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom',
        'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
        'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
        'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
        'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
        'willow_tree', 'wolf', 'woman', 'worm'
    ]
    return cifar100_labels


def _get_imagenet1k_labels():
    """
    Get all 1000 ImageNet-1k class labels.
    These are the standard ImageNet class names.
    Each label is split at commas to extract individual class names.
    """

    # Get the path to the labels file (in the same directory as this module)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    labels_path = os.path.join(current_dir, 'imagenet1K_labels.txt')

    # Read and parse the dictionary from the file
    with open(labels_path, 'r') as f:
        content = f.read()
        labels_dict = ast.literal_eval(content)

    # Convert dictionary to list and split each label at commas
    imagenet_labels = []
    for i in range(len(labels_dict)):
        # Split by comma and take all parts as separate labels
        label_parts = [part.strip() for part in labels_dict[i].split(',')]
        imagenet_labels.extend(label_parts)

    return imagenet_labels


def _get_openai_imagenet_labels():
    """
    Get OpenAI's curated ImageNet labels.
    These are cleaner, more human-readable versions.
    Source: https://github.com/openai/CLIP/blob/main/data/imagenet_classes.txt
    """
    print('Warning: this is not supported yet (it is only a boilerplate)')
    openai_labels = [
        'tench, Tinca tinca',
        'goldfish, Carassius auratus',
        'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
        'tiger shark, Galeocerdo cuvieri',
        # ... (full 1000 classes available at the GitHub link)
    ]
    return openai_labels


def load_tiny_imagenet_labels_from_file(dataset_root):
    """
    Load Tiny ImageNet labels directly from the dataset's wnids.txt and words.txt files.

    Parameters
    ----------
    dataset_root : str
        Path to the tiny-imagenet-200 directory

    Returns
    -------
    list of str
        List of 200 human-readable class names
    """
    import os

    # Load WordNet IDs
    wnids_path = os.path.join(dataset_root, 'wnids.txt')
    with open(wnids_path, 'r') as f:
        wnids = [line.strip() for line in f]

    # Load WordNet ID to name mapping
    words_path = os.path.join(dataset_root, 'words.txt')
    wnid_to_name = {}
    with open(words_path, 'r') as f:
        for line in f:
            wnid, name = line.strip().split('\t')
            wnid_to_name[wnid] = name

    # Get names for all classes in the dataset
    labels = [wnid_to_name[wnid] for wnid in wnids]
    return labels
