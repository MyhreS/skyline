import sys
sys.path.append('/Users/simonmyhre/workdir/gitdir/skyline')
from cirrus import Data

import os
from dotenv import load_dotenv
import os
load_dotenv()



if __name__ == "__main__":
    data = Data(os.getenv("DATA"))
    data.load_data()
    data.map_labels_to_classes({
        'drone': ['normal_drone', 'racing_drone', 'normal_fixedwing', 'petrol_fixedwing'],
        'non-drone': ['no_class']
    })

    data.define_test_dataset()
    data.define_training_dataset()
    data.add_augmentation_to_training_wavs(['low_pass'])
    data.print_data_stats()
    data.write_dataset(audio_format="stft", clean=True, file_type="npy")    












    








